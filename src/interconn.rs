use super::mem_fetch;
use console::style;
use std::collections::VecDeque;

/// Interconnect is a general interconnect
///
/// Functions are not mutable because the interface should
/// implement locking internally
pub trait Interconnect<P>: std::fmt::Debug + Send + Sync + 'static {
    fn busy(&self) -> bool;

    fn push(&self, _src: usize, _dest: usize, _packet: P, _size: u32);

    fn pop(&self, _dest: usize) -> Option<P>;

    fn has_buffer(&self, _dest: usize, _size: u32) -> bool;

    fn transfer(&self);
}

#[derive(Debug)]
pub struct SimpleInterconnect<P> {
    pub num_cores: usize,
    pub num_mems: usize,
    pub num_subnets: usize,
    pub num_nodes: usize,
    pub direct_connection: ndarray::Array2<shared::UnboundedChannel<P>>,
}

impl<P> SimpleInterconnect<P> {
    #[must_use]
    pub fn new(num_cores: usize, num_mems: usize) -> SimpleInterconnect<P> {
        let num_subnets = 2;
        let num_nodes = num_cores + num_mems;
        let direct_connection =
            ndarray::Array2::from_shape_simple_fn((num_subnets, num_nodes), || Default::default());

        Self {
            num_cores,
            num_mems,
            num_subnets,
            num_nodes,
            direct_connection,
        }
    }
}

impl<P> Interconnect<P> for SimpleInterconnect<P>
where
    P: Send + Sync + std::fmt::Display + std::fmt::Debug + 'static,
{
    fn busy(&self) -> bool {
        false
    }

    fn push(&self, src_device: usize, dest_device: usize, packet: P, size: u32) {
        assert!(self.has_buffer(src_device, size));

        let is_memory_node = self.num_subnets > 1 && dest_device >= self.num_cores;
        let subnet = usize::from(is_memory_node);
        log::debug!(
            "{}: {size} bytes from device {src_device} to {dest_device} ({}) (subnet {subnet})",
            style(format!("INTERCONN PUSH {packet}")).bold(),
            if is_memory_node {
                format!("subpartition={}", dest_device - self.num_cores)
            } else {
                format!("sm={}", dest_device)
            },
        );

        self.direct_connection[[subnet, dest_device]]
            .try_send(packet)
            .expect("failed to send packet");
    }

    fn pop(&self, src_device: usize) -> Option<P> {
        let subnet = usize::from(src_device >= self.num_cores);
        self.direct_connection[[subnet, src_device]].receive()
    }

    fn transfer(&self) {
        // do nothing
    }

    fn has_buffer(&self, _device: usize, _size: u32) -> bool {
        true
    }
}

/// Memory interconnect interface between components.
///
/// Functions are not mutable because the interface should
/// implement locking internally
pub trait MemFetchInterface: Send + Sync + std::fmt::Debug + 'static {
    fn is_full(&self, size: u32, write: bool) -> bool;

    fn push(&self, _fetch: mem_fetch::MemFetch, time: u64);
}

#[derive(Clone, PartialEq, Eq, Ord, PartialOrd)]
pub struct Packet<T> {
    pub fetch: T,
    pub time: u64,
}

impl<T> std::ops::Deref for Packet<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.fetch
    }
}

impl<T> std::ops::DerefMut for Packet<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.fetch
    }
}

impl<T> AsRef<T> for Packet<T> {
    fn as_ref(&self) -> &T {
        &self.fetch
    }
}

impl<T> Packet<T> {
    pub fn into_inner(self) -> T {
        self.fetch
    }
}

impl<T> std::fmt::Debug for Packet<T>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.fetch, f)
    }
}

impl<T> std::fmt::Display for Packet<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.fetch, f)
    }
}

/// A direct, owned connection between two components
pub trait Connection<P>: Sync + Send + 'static {
    /// If the connection can send a new message
    #[must_use]
    fn can_send(&self, packet_sizes: &[u32]) -> bool;

    /// Sends a packet to the connection
    fn send(&mut self, packet: P);
    // fn send(&mut self, packet: Packet<P>);

    /// Receive a packet from the connection
    fn receive(&mut self) -> Option<P>;
}

impl<P> Connection<P> for VecDeque<P>
where
    P: Send + Sync + 'static,
{
    fn can_send(&self, _packet_sizes: &[u32]) -> bool {
        true
    }

    fn send(&mut self, packet: P) {
        self.push_back(packet);
    }

    fn receive(&mut self) -> Option<P> {
        self.pop_front()
    }
}

/// A shared connection using internal mutability.
pub trait SharedConnection<P>: Sync + Send + 'static {
    /// Sends a packet to the connection
    fn try_send(&self, packet: P) -> Result<(), P>;

    /// Receives a packet from the connection
    fn receive(&self) -> Option<P>;
}

/// A bounded connection
pub trait BoundedConnection {
    fn is_full(&self) -> bool;

    fn capacity(&self) -> usize;

    fn len(&self) -> usize;
}

pub trait Iter<I, P>
where
    I: Iterator<Item = P>,
{
    fn iter(&self) -> I;
}

/// A buffered connection between two components
pub trait BufferedConnection<P>: Connection<P> {
    /// Iterator over buffered in-flight messages
    fn buffered(&self) -> Box<dyn Iterator<Item = &P> + '_>;

    #[must_use]
    fn num_buffered(&self) -> usize {
        self.buffered().count()
    }

    #[must_use]
    fn is_empty(&self) -> bool {
        self.buffered().count() == 0
    }

    fn drain(&mut self) -> Box<dyn Iterator<Item = P> + '_>;
}

impl<P> BufferedConnection<P> for VecDeque<P>
where
    P: Send + Sync + 'static,
{
    fn buffered(&self) -> Box<dyn Iterator<Item = &P> + '_> {
        Box::new(self.iter())
    }

    fn num_buffered(&self) -> usize {
        self.len()
    }

    fn drain(&mut self) -> Box<dyn Iterator<Item = P> + '_> {
        Box::new(self.drain(..))
    }
}

pub mod shared {
    #[derive(Debug, Clone)]
    pub struct UnboundedChannel<P>(
        (
            crossbeam::channel::Sender<P>,
            crossbeam::channel::Receiver<P>,
        ),
    );

    impl<P> Default for UnboundedChannel<P> {
        fn default() -> Self {
            Self(crossbeam::channel::unbounded())
        }
    }

    impl<P> UnboundedChannel<P> {
        pub fn new() -> Self {
            Self::default()
        }
    }

    impl<P> super::SharedConnection<P> for UnboundedChannel<P>
    where
        P: Send + Sync + 'static,
    {
        /// Sends a packet to the connection
        fn try_send(&self, packet: P) -> Result<(), P> {
            let (sender, _) = &self.0;
            // this does not block
            sender
                .try_send(packet)
                .map_err(crossbeam::channel::TrySendError::into_inner)
        }

        fn receive(&self) -> Option<P> {
            let (_, receiver) = &self.0;
            // this does not block
            receiver.try_recv().ok()
        }
    }

    #[derive(Debug)]
    pub struct UnboundedFifoQueue<P>(crossbeam::queue::SegQueue<P>);

    impl<P> Default for UnboundedFifoQueue<P> {
        fn default() -> Self {
            Self(crossbeam::queue::SegQueue::new())
        }
    }

    impl<P> UnboundedFifoQueue<P> {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn len(&self) -> usize {
            self.0.len()
        }
    }

    impl<P> super::SharedConnection<P> for UnboundedFifoQueue<P>
    where
        P: Send + Sync + 'static,
    {
        /// Sends a packet to the connection
        fn try_send(&self, packet: P) -> Result<(), P> {
            self.0.push(packet);
            Ok(())
        }

        fn receive(&self) -> Option<P> {
            self.0.pop()
        }
    }

    pub mod debug {
        use crate::sync::Mutex;

        #[derive(Debug)]
        pub struct UnboundedFifoQueueOld<P>(pub Mutex<crate::fifo::Fifo<P>>);

        impl<P> Default for UnboundedFifoQueueOld<P> {
            fn default() -> Self {
                Self(Mutex::new(crate::fifo::Fifo::new(None)))
            }
        }

        impl<P> UnboundedFifoQueueOld<P> {
            pub fn new() -> Self {
                Self::default()
            }

            pub fn len(&self) -> usize {
                self.0.lock().len()
            }
        }

        impl<P> crate::interconn::SharedConnection<P> for UnboundedFifoQueueOld<P>
        where
            P: Send + Sync + 'static,
        {
            /// Sends a packet to the connection
            fn try_send(&self, packet: P) -> Result<(), P> {
                let mut lock = self.0.lock();
                if lock.full() {
                    Err(packet)
                } else {
                    lock.enqueue(packet);
                    Ok(())
                }
            }

            fn receive(&self) -> Option<P> {
                self.0.lock().dequeue()
            }
        }
    }

    #[derive(Debug)]
    pub struct BoundedFifoQueue<P>(crossbeam::queue::ArrayQueue<P>);

    impl<P> BoundedFifoQueue<P> {
        pub fn new(capacity: usize) -> Self {
            Self(crossbeam::queue::ArrayQueue::new(capacity))
        }
    }

    impl<P> super::SharedConnection<P> for BoundedFifoQueue<P>
    where
        P: Send + Sync + 'static,
    {
        /// Sends a packet to the connection
        fn try_send(&self, packet: P) -> Result<(), P> {
            self.0.push(packet)
        }

        fn receive(&self) -> Option<P> {
            self.0.pop()
        }
    }

    impl<P> super::BoundedConnection for BoundedFifoQueue<P> {
        fn is_full(&self) -> bool {
            self.0.is_full()
        }

        fn capacity(&self) -> usize {
            self.0.capacity()
        }

        fn len(&self) -> usize {
            self.0.len()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::config;
    use color_eyre::eyre;

    use std::path::PathBuf;

    #[test]
    fn test_intersim_config() -> eyre::Result<()> {
        use playground::interconnect::IntersimConfig;

        let config_file = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("accelsim/gtx1080/config_fermi_islip.icnt");

        let config = IntersimConfig::from_file(&config_file)?;

        assert!(!config.get_bool("use_map"));
        assert_eq!(config.get_int("num_vcs"), 1); // this means vc can only ever be zero
        assert_eq!(config.get_int("ejection_buffer_size"), 0);
        assert_eq!(config.get_string("sim_type"), "gpgpusim");
        assert_eq!(config.get_string("topology"), "fly");
        Ok(())
    }

    #[ignore = "bridged box interconnect implementation can segfault"]
    #[test]
    fn test_box_interconnect() -> eyre::Result<()> {
        use playground::interconnect::{BoxInterconnect, Interconnect};

        let config = config::GPU::default();
        let num_clusters = config.num_simt_clusters;
        let num_mem_sub_partitions = config.total_sub_partitions();
        dbg!(&num_clusters);
        dbg!(&num_mem_sub_partitions);

        let config_file = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("accelsim/gtx1080/config_fermi_islip.icnt");
        dbg!(&config_file);

        let mut interconn: Interconnect<u32, _> = Interconnect::new(BoxInterconnect::new(
            &config_file,
            num_clusters as u32,
            num_mem_sub_partitions as u32,
        ));

        let num_shaders = interconn.num_shaders();

        let core_node = 0;
        let mem_node = num_shaders;

        // send from core to memory
        // let send_data = 42;
        interconn.push(core_node, mem_node, Box::new(42));
        interconn.push(core_node, mem_node, Box::new(43));
        interconn.push(core_node, mem_node, Box::new(44));
        let (_, recv_data) = interconn.must_pop(mem_node, Some(1))?;
        assert_eq!(42, *recv_data);
        let (_, recv_data) = interconn.must_pop(mem_node, Some(1))?;
        assert_eq!(43, *recv_data);
        let (_, recv_data) = interconn.must_pop(mem_node, Some(1))?;
        assert_eq!(44, *recv_data);

        // send memory to core back
        // let send_data = 24;
        interconn.push(mem_node, core_node, Box::new(32));
        interconn.push(mem_node, core_node, Box::new(33));
        interconn.push(mem_node, core_node, Box::new(34));
        let (_, recv_data) = interconn.must_pop(core_node, Some(1))?;
        assert_eq!(32, *recv_data);
        let (_, recv_data) = interconn.must_pop(core_node, Some(1))?;
        assert_eq!(33, *recv_data);
        let (_, recv_data) = interconn.must_pop(core_node, Some(1))?;
        assert_eq!(34, *recv_data);

        Ok(())
    }

    #[ignore = "race condition in interconnect"]
    #[test]
    fn test_interconnect_interface() -> eyre::Result<()> {
        use playground::interconnect::{Interconnect, InterconnectInterface};

        let config = config::GPU::default();
        let num_clusters = config.num_simt_clusters;
        let num_mem_sub_partitions = config.total_sub_partitions();
        dbg!(&num_clusters);
        dbg!(&num_mem_sub_partitions);

        let config_file = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("accelsim/gtx1080/config_fermi_islip.icnt");
        dbg!(&config_file);

        let mut interconn: Interconnect<u32, _> = Interconnect::new(InterconnectInterface::new(
            &config_file,
            num_clusters as u32,
            num_mem_sub_partitions as u32,
        ));

        let num_nodes = interconn.num_nodes();
        let num_shaders = interconn.num_shaders();
        let num_memories = interconn.num_memories();
        dbg!(&num_nodes);
        dbg!(&num_shaders);
        dbg!(&num_memories);

        let core_node = 0;
        let mem_node = num_shaders;

        // send from core to memory
        let send_data = 42;
        interconn.push(core_node, mem_node, Box::new(send_data));
        let (elapsed, recv_data) = interconn.must_pop(mem_node, None)?;
        dbg!(elapsed);
        assert!(elapsed > 1);
        assert_eq!(send_data, *recv_data);

        // send memory to core back
        let send_data = 24;
        interconn.push(mem_node, core_node, Box::new(send_data));
        let (elapsed, recv_data) = interconn.must_pop(core_node, None)?;
        assert!(elapsed > 0);
        assert_eq!(send_data, *recv_data);
        Ok(())
    }

    #[ignore = "todo"]
    #[test]
    fn test_interconnect() {}
}
