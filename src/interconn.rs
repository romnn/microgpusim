use super::{config, mem_fetch, Packet};
use console::style;
use std::collections::VecDeque;

use std::sync::{Arc, Mutex};

/// Interconnect is a general interconnect
///
/// Functions are not mutable because the interface should
/// implement locking internally
pub trait Interconnect<P> {
    fn busy(&self) -> bool {
        todo!("interconn: busy");
    }

    fn push(&self, _src: usize, _dest: usize, _packet: P, _size: u32) {
        todo!("interconn: push");
    }
    fn pop(&self, _dest: usize) -> Option<P> {
        todo!("interconn: pop");
    }
    fn has_buffer(&self, _dest: usize, _size: u32) -> bool {
        todo!("interconn: has buffer");
    }
    fn transfer(&self) {
        todo!("interconn: transfer");
    }
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
    output_queue: Vec<Vec<Vec<Mutex<VecDeque<P>>>>>,
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
        }
    }
}

impl<P> Interconnect<P> for ToyInterconnect<P>
where
    P: std::fmt::Display + std::fmt::Debug,
{
    fn busy(&self) -> bool {
        // todo: this is not efficient, could keep track of this with a variable
        self.output_queue
            .iter()
            .flatten()
            .flatten()
            .any(|reqs: &Mutex<VecDeque<_>>| !reqs.lock().unwrap().is_empty())
    }

    fn push(&self, src_device: usize, dest_device: usize, packet: P, size: u32) {
        assert!(self.has_buffer(src_device, size));

        let is_memory_node = self.num_subnets > 1 && dest_device >= self.num_cores;
        let subnet = usize::from(is_memory_node);
        log::debug!(
            "{}: {size} bytes from device {src_device} to {dest_device} (subnet {subnet})",
            style(format!("INTERCONN PUSH {packet}")).bold(),
        );

        let mut queue = self.output_queue[subnet][dest_device][0].lock().unwrap();
        queue.push_back(packet);
    }

    fn pop(&self, device: usize) -> Option<P> {
        let icnt_id = device;
        let subnet = usize::from(device >= self.num_cores);

        let mut lock = self.round_robin_turn[subnet][icnt_id].lock().unwrap();
        let mut turn = *lock;
        log::debug!(
            "{}: from device {device} (device={device}, id={icnt_id}, subnet={subnet}, turn={turn})",
            style("INTERCONN POP").bold()
        );

        for _ in 0..self.num_classes {
            let mut queue = self.output_queue[subnet][icnt_id][turn].lock().unwrap();
            turn = (turn + 1) % self.num_classes;
            if let Some(packet) = queue.pop_front() {
                *lock = turn;
                return Some(packet);
            }
        }
        None
    }

    fn transfer(&self) {
        // do nothing
    }

    fn has_buffer(&self, _device: usize, _size: u32) -> bool {
        true
        // let Some(capacity) = self.capacity else {
        //     return true;
        // };
        //
        // // TODO: using input queue makes no sense as we push into output directly
        // let subnet = usize::from(device >= self.num_cores);
        // let queue = self.input_queue[subnet][device][0].lock().unwrap();
        // queue.len() <= capacity
    }
}

/// Memory interconnect interface between components.
///
/// Functions are not mutable because the interface should
/// implement locking internally
pub trait MemFetchInterface: std::fmt::Debug {
    fn full(&self, size: u32, write: bool) -> bool;

    fn push(&self, _fetch: mem_fetch::MemFetch) {
        todo!("mem fetch interface: full");
    }
}

#[derive()]
pub struct CoreMemoryInterface<P> {
    pub config: Arc<config::GPUConfig>,
    pub stats: Arc<Mutex<stats::Stats>>,
    pub cluster_id: usize,
    pub interconn: Arc<dyn Interconnect<P>>,
}

impl<P> std::fmt::Debug for CoreMemoryInterface<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("CoreMemoryInterface").finish()
    }
}

impl MemFetchInterface for CoreMemoryInterface<Packet> {
    fn full(&self, size: u32, write: bool) -> bool {
        let request_size = if write {
            size
        } else {
            u32::from(mem_fetch::READ_PACKET_SIZE)
        };
        !self.interconn.has_buffer(self.cluster_id, request_size)
    }

    fn push(&self, mut fetch: mem_fetch::MemFetch) {
        // self.core.interconn_simt_to_mem(fetch.get_num_flits(true));
        // self.cluster.interconn_inject_request_packet(fetch);

        {
            let mut stats = self.stats.lock().unwrap();
            let access_kind = *fetch.access_kind();
            debug_assert_eq!(fetch.is_write(), access_kind.is_write());
            stats.accesses.inc(access_kind, 1);
        }

        let dest_sub_partition_id = fetch.sub_partition_id();
        let mem_dest = self.config.mem_id_to_device_id(dest_sub_partition_id);

        log::debug!(
            "cluster {} icnt_inject_request_packet({}) dest sub partition id={} dest mem node={}",
            self.cluster_id,
            fetch,
            dest_sub_partition_id,
            mem_dest
        );

        // The packet size varies depending on the type of request:
        // - For write request and atomic request, packet contains the data
        // - For read request (i.e. not write nor atomic), packet only has control metadata
        let packet_size = if !fetch.is_write() && !fetch.is_atomic() {
            fetch.control_size
        } else {
            // todo: is that correct now?
            fetch.size()
            // fetch.data_size
        };
        // m_stats->m_outgoing_traffic_stats->record_traffic(mf, packet_size);
        fetch.status = mem_fetch::Status::IN_ICNT_TO_MEM;

        self.interconn
            .push(self.cluster_id, mem_dest, Packet::Fetch(fetch), packet_size);
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

impl<Q> MemFetchInterface for L2Interface<Q>
where
    Q: super::fifo::Queue<mem_fetch::MemFetch>,
{
    fn full(&self, _size: u32, _write: bool) -> bool {
        self.l2_to_dram_queue.lock().unwrap().full()
    }

    fn push(&self, mut fetch: mem_fetch::MemFetch) {
        fetch.set_status(mem_fetch::Status::IN_PARTITION_L2_TO_DRAM_QUEUE, 0);
        log::debug!("l2 interface push l2_to_dram_queue");
        self.l2_to_dram_queue.lock().unwrap().enqueue(fetch)
    }
}

#[cfg(test)]
mod tests {
    use crate::config::GPUConfig;
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

        let config = GPUConfig::default();
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
        let (_, recv_data) = interconn.must_pop(mem_node, Some(1)).unwrap();
        assert_eq!(42, *recv_data);
        let (_, recv_data) = interconn.must_pop(mem_node, Some(1)).unwrap();
        assert_eq!(43, *recv_data);
        let (_, recv_data) = interconn.must_pop(mem_node, Some(1)).unwrap();
        assert_eq!(44, *recv_data);

        // send memory to core back
        // let send_data = 24;
        interconn.push(mem_node, core_node, Box::new(32));
        interconn.push(mem_node, core_node, Box::new(33));
        interconn.push(mem_node, core_node, Box::new(34));
        let (_, recv_data) = interconn.must_pop(core_node, Some(1)).unwrap();
        assert_eq!(32, *recv_data);
        let (_, recv_data) = interconn.must_pop(core_node, Some(1)).unwrap();
        assert_eq!(33, *recv_data);
        let (_, recv_data) = interconn.must_pop(core_node, Some(1)).unwrap();
        assert_eq!(34, *recv_data);

        Ok(())
    }

    #[ignore = "race condition in interconnect"]
    #[test]
    fn test_interconnect_interface() -> eyre::Result<()> {
        use playground::interconnect::{Interconnect, InterconnectInterface};

        let config = GPUConfig::default();
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
        let (elapsed, recv_data) = interconn.must_pop(mem_node, None).unwrap();
        dbg!(elapsed);
        assert!(elapsed > 1);
        assert_eq!(send_data, *recv_data);

        // send memory to core back
        let send_data = 24;
        interconn.push(mem_node, core_node, Box::new(send_data));
        let (elapsed, recv_data) = interconn.must_pop(core_node, None).unwrap();
        assert!(elapsed > 0);
        assert_eq!(send_data, *recv_data);

        // assert!(false);
        Ok(())
    }

    #[ignore = "todo"]
    #[test]
    fn test_interconnect() {}
}
