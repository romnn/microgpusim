use super::mem_fetch;
// use crate::sync::{Arc, Mutex, RwLock};
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

    // fn dest_queue(&self, _dest: usize) -> &Mutex<VecDeque<P>>;

    fn transfer(&self);
}

#[derive(Debug)]
pub struct SimpleInterconnect<P> {
    // pub capacity: Option<usize>,
    pub num_cores: usize,
    pub num_mems: usize,
    pub num_subnets: usize,
    pub num_nodes: usize,
    // pub num_classes: usize,

    // input_queue: Vec<Vec<Vec<Mutex<VecDeque<P>>>>>,
    // pub output_queue: Vec<Vec<Vec<Mutex<VecDeque<P>>>>>,

    // round_robin_turn: Vec<Vec<Mutex<usize>>>,
    // pub output_queue: Vec<Vec<Vec<Mutex<VecDeque<P>>>>>,
    // pub round_robin_turn: ndarray::Array2<Mutex<usize>>,
    // pub output_queue: ndarray::Array3<Mutex<VecDeque<P>>>,
    // pub direct_connection: ndarray::Array2<Mutex<VecDeque<P>>>,
    // pub direct_connection: ndarray::Array2<(channel::Sender<P>, channel::Receiver<P>)>,
    // pub direct_connection: ndarray::Array2<shared::UnboundedFifoQueue<P>>,
    pub direct_connection: ndarray::Array2<shared::UnboundedChannel<P>>,
    // pub in_flight: RwLock<u64>,
    // use ndarray here

    // deviceID to icntID map
    // deviceID : Starts from 0 for shaders and then continues until mem nodes
    // which starts at location n_shader and then continues to n_shader+n_mem (last device)
    // node_map: HashMap<usize, usize>,
}

impl<P> SimpleInterconnect<P> {
    #[must_use]
    pub fn new(num_cores: usize, num_mems: usize) -> SimpleInterconnect<P> {
        let num_subnets = 2;
        let num_nodes = num_cores + num_mems;
        // let num_classes = 1;

        // let mut output_queue: Vec<Vec<Vec<Mutex<VecDeque<P>>>>> = Vec::new();
        // let mut round_robin_turn: Vec<Vec<Mutex<usize>>> = Vec::new();
        //
        // for subnet in 0..num_subnets {
        //     // input_queue.push(Vec::new());
        //     output_queue.push(Vec::new());
        //     round_robin_turn.push(Vec::new());
        //
        //     for node in 0..num_nodes {
        //         // input_queue[subnet].push(Vec::new());
        //         output_queue[subnet].push(Vec::new());
        //         round_robin_turn[subnet].push(Mutex::new(0));
        //
        //         for _class in 0..num_classes {
        //             // input_queue[subnet][node].push(Mutex::new(VecDeque::new()));
        //             output_queue[subnet][node].push(Mutex::new(VecDeque::new()));
        //         }
        //     }
        // }

        // let round_robin_turn =
        //     ndarray::Array2::from_shape_simple_fn((num_subnets, num_nodes), || Mutex::new(0));

        let direct_connection =
            ndarray::Array2::from_shape_simple_fn((num_subnets, num_nodes), || {
                // Mutex::new(VecDeque::new())
                // channel::unbounded()
                // CrossbeamConnection::new()
                // shared::UnboundedFifoQueue::new()
                Default::default()
                // shared::UnboundedChannel::new()
            });

        Self {
            // capacity,
            // input_queue,
            num_cores,
            num_mems,
            num_subnets,
            num_nodes,
            // num_classes,
            // output_queue,
            // round_robin_turn,
            direct_connection,
            // round_robin_turn,
            // in_flight: RwLock::new(0),
        }
    }
}

impl<P> Interconnect<P> for SimpleInterconnect<P>
where
    P: Send + Sync + std::fmt::Display + std::fmt::Debug + 'static,
{
    // #[inline]
    fn busy(&self) -> bool {
        false
        // *self.in_flight.read() != 0
        // self.output_queue
        //     .iter()
        //     .flatten()
        //     .flatten()
        //     .any(|reqs: &Mutex<VecDeque<_>>| !reqslock().is_empty())
    }

    // #[inline]
    // fn dest_queue(&self, dest_device: usize) -> &Mutex<VecDeque<P>> {
    //     // assert!(self.has_buffer(src_device, size));
    //
    //     let is_memory_node = self.num_subnets > 1 && dest_device >= self.num_cores;
    //     let subnet = usize::from(is_memory_node);
    //     // log::debug!(
    //     //     "{}: {size} bytes from device {src_device} to {dest_device} (subnet {subnet})",
    //     //     style(format!("INTERCONN PUSH {packet}")).bold(),
    //     // );
    //
    //     // &self.output_queue[subnet][dest_device][0]
    //     &self.output_queue[[subnet, dest_device]]
    // }

    // #[inline]
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

        // *self.in_flight.write() += 1;
        // let mut connection = self.direct_connection[[subnet, dest_device]].lock();
        // connection.push_back(packet);

        self.direct_connection[[subnet, dest_device]]
            .try_send(packet)
            .expect("failed to send packet");
        // let (sender, _) = &self.direct_connection[[subnet, dest_device]];
        // sender.send(packet).unwrap();
    }

    // #[inline]
    fn pop(&self, src_device: usize) -> Option<P> {
        let subnet = usize::from(src_device >= self.num_cores);

        // let mut lock = self.round_robin_turn[[subnet, icnt_id]].lock();
        // let mut turn = *lock;
        // {
        //     let queue = self.direct_connection[[subnet, icnt_id]].lock();
        //     log::debug!(
        //         "{}: from device {device} (device={device}, id={icnt_id}, subnet={subnet}, buffer={:?})",
        //         style("INTERCONN POP").bold(),
        //         queue.iter().map(ToString::to_string).collect::<Vec<_>>(),
        //     );
        // }

        // for _ in 0..self.num_classes {

        // let mut connection = self.direct_connection[[subnet, icnt_id]].lock();
        // // turn = (turn + 1) % self.num_classes;
        // if let Some(packet) = connection.pop_front() {
        //     // *lock = turn;
        //     *self.in_flight.write() -= 1;
        //     return Some(packet);
        // }
        // // }
        // None

        self.direct_connection[[subnet, src_device]].receive()
        // let (_, receiver) = &self.direct_connection[[subnet, src_device]];
        // receiver.try_recv().ok()
    }

    // #[inline]
    fn transfer(&self) {
        // do nothing
    }

    // #[inline]
    fn has_buffer(&self, _device: usize, _size: u32) -> bool {
        true
        // let Some(capacity) = self.capacity else {
        //     return true;
        // };
        //
        // // TODO: using input queue makes no sense as we push into output directly
        // let subnet = usize::from(device >= self.num_cores);
        // let queue = self.input_queue[subnet][device][0]lock();
        // queue.len() <= capacity
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

// #[derive()]
// pub struct L2Interface<Q> {
//     pub l2_to_dram_queue: Arc<Mutex<Q>>,
// }
//
// impl<Q> std::fmt::Debug for L2Interface<Q> {
//     fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
//         f.debug_struct("L2Interface").finish()
//     }
// }

// pub trait Network<P> {
//     // todo
// }

// pub type Port = Arc<Mutex<VecDeque<(usize, mem_fetch::MemFetch, u32)>>>;
// pub type Port<P> = Arc<Mutex<dyn Connection<Packet<P>>>>;

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
    // #[inline]
    fn can_send(&self, _packet_sizes: &[u32]) -> bool {
        true
    }

    // #[inline]
    fn send(&mut self, packet: P) {
        self.push_back(packet);
    }

    fn receive(&mut self) -> Option<P> {
        self.pop_front()
    }
}

/// A shared connection using internal mutability.
pub trait SharedConnection<P>: Sync + Send + 'static {
    /// If the connection can send a new message
    // #[must_use]
    // fn can_send(&self, packet_sizes: &[u32]) -> bool;

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
    // #[inline]
    fn buffered(&self) -> Box<dyn Iterator<Item = &P> + '_> {
        Box::new(self.iter())
    }

    // #[inline]
    fn num_buffered(&self) -> usize {
        self.len()
    }

    // #[inline]
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
            // Self(crossbeam::channel::unbounded())
        }
    }

    impl<P> super::SharedConnection<P> for UnboundedChannel<P>
    where
        P: Send + Sync + 'static,
    {
        // If the connection can send a new message
        // #[must_use]
        // fn can_send(&self, _packet_sizes: &[u32]) -> bool {
        //     true
        // }

        /// Sends a packet to the connection
        fn try_send(&self, packet: P) -> Result<(), P> {
            let (sender, _) = &self.0;
            // this does not block
            sender
                .try_send(packet)
                .map_err(crossbeam::channel::TrySendError::into_inner)
            // let err = sender.try_send(packet).err()?;
            // give ownership of packet back to caller
            // Some(err.into_inner())
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
            // Self(crossbeam::queue::SegQueue::new())
        }

        pub fn len(&self) -> usize {
            self.0.len()
        }

        // pub fn drain(&self) -> usize {
        //     self.0.len()
        // }
    }

    impl<P> super::SharedConnection<P> for UnboundedFifoQueue<P>
    where
        P: Send + Sync + 'static,
    {
        // // If the connection can send a new message
        // #[must_use]
        // fn can_send(&self, _packet_sizes: &[u32]) -> bool {
        //     true
        // }
        //
        // // Sends a packet to the connection
        // fn send(&self, packet: P) {
        //     self.0.push(packet);
        // }

        /// Sends a packet to the connection
        fn try_send(&self, packet: P) -> Result<(), P> {
            self.0.push(packet);
            Ok(())
            // let returned_packet = self.0.push(packet).err()?;
            // SOme(returned_packet)
            // give ownership of packet back to caller
            // Some(err.into_inner())
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
                // let returned_packet = self.0.push(packet).err()?;
                // SOme(returned_packet)
                // give ownership of packet back to caller
                // Some(err.into_inner())
            }

            fn receive(&self) -> Option<P> {
                self.0.lock().dequeue()
            }
        }
    }

    // impl<P> super::Iter<<crossbeam::queue::SegQueue<P> as IntoIterator>::IntoIter, P>
    //     for UnboundedFifoQueue<P>
    // {
    //     fn iter(&self) -> <crossbeam::queue::SegQueue<P> as IntoIterator>::IntoIter {
    //         self.0.clone().into_iter()
    //     }
    // }

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
        // /// If the connection can send a new message
        // #[must_use]
        // fn can_send(&self, _packet_sizes: &[u32]) -> bool {
        //     self.0.is_full()
        // }
        //
        // /// Sends a packet to the connection
        // fn send(&self, packet: P) {
        //     self.0.push(packet);
        // }

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

    // impl<P> super::Iter<String> for BoundedFifoQueue<P> {
    //     fn iter(&self) -> String {
    //         "test".to_string()
    //     }
    // }
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
