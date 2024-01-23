use super::mem_fetch;
use crate::sync::{Arc, Mutex, RwLock};
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

    fn dest_queue(&self, _dest: usize) -> &Mutex<VecDeque<P>>;

    fn transfer(&self);
}

#[derive(Debug)]
pub struct ToyInterconnect<P> {
    // pub capacity: Option<usize>,
    pub num_cores: usize,
    pub num_mems: usize,
    pub num_subnets: usize,
    pub num_nodes: usize,
    pub num_classes: usize,
    round_robin_turn: Vec<Vec<Mutex<usize>>>,
    // input_queue: Vec<Vec<Vec<Mutex<VecDeque<P>>>>>,
    pub output_queue: Vec<Vec<Vec<Mutex<VecDeque<P>>>>>,
    pub in_flight: RwLock<u64>,
    // deviceID to icntID map
    // deviceID : Starts from 0 for shaders and then continues until mem nodes
    // which starts at location n_shader and then continues to n_shader+n_mem (last device)
    // node_map: HashMap<usize, usize>,
}

impl<P> ToyInterconnect<P> {
    #[must_use]
    pub fn new(num_cores: usize, num_mems: usize) -> ToyInterconnect<P> {
        let num_subnets = 2;
        let num_nodes = num_cores + num_mems;
        let num_classes = 1;

        // let mut input_queue: Vec<Vec<Vec<Mutex<VecDeque<P>>>>> = Vec::new();
        let mut output_queue: Vec<Vec<Vec<Mutex<VecDeque<P>>>>> = Vec::new();
        let mut round_robin_turn: Vec<Vec<Mutex<usize>>> = Vec::new();

        for subnet in 0..num_subnets {
            // input_queue.push(Vec::new());
            output_queue.push(Vec::new());
            round_robin_turn.push(Vec::new());

            for node in 0..num_nodes {
                // input_queue[subnet].push(Vec::new());
                output_queue[subnet].push(Vec::new());
                round_robin_turn[subnet].push(Mutex::new(0));

                for _class in 0..num_classes {
                    // input_queue[subnet][node].push(Mutex::new(VecDeque::new()));
                    output_queue[subnet][node].push(Mutex::new(VecDeque::new()));
                }
            }
        }
        Self {
            // capacity,
            num_cores,
            num_mems,
            num_subnets,
            num_nodes,
            num_classes,
            round_robin_turn,
            // input_queue,
            output_queue,
            in_flight: RwLock::new(0),
        }
    }
}

impl<P> Interconnect<P> for ToyInterconnect<P>
where
    P: Send + Sync + std::fmt::Display + std::fmt::Debug + 'static,
{
    // #[inline]
    fn busy(&self) -> bool {
        // todo: this is not efficient, could keep track of this with a variable
        *self.in_flight.read() != 0
        // self.output_queue
        //     .iter()
        //     .flatten()
        //     .flatten()
        //     .any(|reqs: &Mutex<VecDeque<_>>| !reqslock().is_empty())
    }

    // #[inline]
    fn dest_queue(&self, dest_device: usize) -> &Mutex<VecDeque<P>> {
        // assert!(self.has_buffer(src_device, size));

        let is_memory_node = self.num_subnets > 1 && dest_device >= self.num_cores;
        let subnet = usize::from(is_memory_node);
        // log::debug!(
        //     "{}: {size} bytes from device {src_device} to {dest_device} (subnet {subnet})",
        //     style(format!("INTERCONN PUSH {packet}")).bold(),
        // );

        &self.output_queue[subnet][dest_device][0]
    }

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

        *self.in_flight.write() += 1;
        let mut queue = self.output_queue[subnet][dest_device][0].lock();
        queue.push_back(packet);
    }

    // #[inline]
    fn pop(&self, device: usize) -> Option<P> {
        let icnt_id = device;
        let subnet = usize::from(device >= self.num_cores);

        let mut lock = self.round_robin_turn[subnet][icnt_id].lock();
        let mut turn = *lock;
        {
            let queue = self.output_queue[subnet][icnt_id][turn].lock();
            log::debug!(
                "{}: from device {device} (device={device}, id={icnt_id}, subnet={subnet}, turn={turn}, buffer={:?})",
                style("INTERCONN POP").bold(),
                queue.iter().map(ToString::to_string).collect::<Vec<_>>(),
            );
        }

        for _ in 0..self.num_classes {
            let mut queue = self.output_queue[subnet][icnt_id][turn].lock();
            turn = (turn + 1) % self.num_classes;
            if let Some(packet) = queue.pop_front() {
                *lock = turn;
                *self.in_flight.write() -= 1;
                return Some(packet);
            }
        }
        None
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
    fn full(&self, size: u32, write: bool) -> bool;

    fn push(&self, _fetch: mem_fetch::MemFetch, time: u64);
}

#[derive(Clone, PartialEq, Eq, Ord, PartialOrd)]
pub struct Packet<T> {
    pub data: T,
    // size: u32,
    // src: usize,
    // destination: usize,
    pub time: u64,
}

impl<T> std::ops::Deref for Packet<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.data
    }
}

impl<T> std::ops::DerefMut for Packet<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.data
    }
}

impl<T> AsRef<T> for Packet<T> {
    fn as_ref(&self) -> &T {
        &self.data
    }
}

impl<T> Packet<T> {
    pub fn into_inner(self) -> T {
        self.data
    }
}

impl<T> std::fmt::Debug for Packet<T>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.data, f)
    }
}

impl<T> std::fmt::Display for Packet<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.data, f)
    }
}

#[derive()]
pub struct L2Interface<Q> {
    pub l2_to_dram_queue: Arc<Mutex<Q>>,
}

impl<Q> std::fmt::Debug for L2Interface<Q> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("L2Interface").finish()
    }
}

pub trait Network<P> {
    // todo
}

// pub type Port = Arc<Mutex<VecDeque<(usize, mem_fetch::MemFetch, u32)>>>;
pub type Port<P> = Arc<Mutex<dyn Connection<Packet<P>>>>;

/// A connection between two components
pub trait Connection<P>: Sync + Send + 'static {
    /// If the connection can send a new message
    #[must_use]
    fn can_send(&self, packet_sizes: &[u32]) -> bool;

    /// Sends a packet to the connection
    fn send(&mut self, packet: P);
    // fn send(&mut self, packet: Packet<P>);
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
