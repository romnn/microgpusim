use super::{config, mem_fetch, Packet};
use console::style;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, Weak};

/// Interconnect is a general interconnect
///
/// Functions are not mutable because the interface should
/// implement locking internally
pub trait Interconnect<P> {
    fn busy(&self) -> bool {
        todo!("interconn: busy");
    }

    fn push(&self, src: usize, dest: usize, packet: P, size: u32) {
        todo!("interconn: push");
    }
    fn pop(&self, dest: usize) -> Option<P> {
        todo!("interconn: pop");
    }
    fn has_buffer(&self, dest: usize, size: u32) -> bool {
        todo!("interconn: has buffer");
    }
    fn transfer(&self) {
        todo!("interconn: transfer");
    }
}

#[derive(Debug)]
pub struct ToyInterconnect<P> {
    pub capacity: Option<usize>,
    pub num_cores: usize,
    pub num_mems: usize,
    pub num_subnets: usize,
    pub num_nodes: usize,
    pub num_classes: usize,
    round_robin_turn: Vec<Vec<Mutex<usize>>>,
    input_queue: Vec<Vec<Vec<Mutex<VecDeque<P>>>>>,
    output_queue: Vec<Vec<Vec<Mutex<VecDeque<P>>>>>,
    // deviceID to icntID map
    // deviceID : Starts from 0 for shaders and then continues until mem nodes
    // which starts at location n_shader and then continues to n_shader+n_mem (last device)
    node_map: HashMap<usize, usize>,
}

// impl<P> Default for ToyInterconnect<P> {
//     fn default() -> ToyInterconnect<P> {
//         Self {
//             capacity: Some(100),
//             num_cores: 1,
//             num_mems: 1,
//             queue: vec![vec![VecDeque::new()]],
//         }
//     }
// }

impl<P> ToyInterconnect<P> {
    pub fn new(num_cores: usize, num_mems: usize, capacity: Option<usize>) -> ToyInterconnect<P> {
        let num_subnets = 2;
        let num_nodes = num_cores + num_mems;
        let num_classes = 1;

        let mut input_queue: Vec<Vec<Vec<Mutex<VecDeque<P>>>>> = Vec::new();
        let mut output_queue: Vec<Vec<Vec<Mutex<VecDeque<P>>>>> = Vec::new();
        let mut round_robin_turn: Vec<Vec<Mutex<usize>>> = Vec::new();

        let node_map = HashMap::new();

        for subnet in 0..num_subnets {
            input_queue.push(Vec::new());
            output_queue.push(Vec::new());
            round_robin_turn.push(Vec::new());

            for node in 0..num_nodes {
                input_queue[subnet].push(Vec::new());
                output_queue[subnet].push(Vec::new());
                round_robin_turn[subnet].push(Mutex::new(0));

                for class in 0..num_classes {
                    input_queue[subnet][node].push(Mutex::new(VecDeque::new()));
                    output_queue[subnet][node].push(Mutex::new(VecDeque::new()));
                }
            }
        }
        Self {
            capacity,
            num_cores,
            num_mems,
            num_subnets,
            num_nodes,
            num_classes,
            input_queue,
            output_queue,
            node_map,
            round_robin_turn,
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
            .flat_map(|x| x)
            .flat_map(|x| x)
            .any(|reqs: &Mutex<VecDeque<_>>| !reqs.lock().unwrap().is_empty())
    }

    fn push(&self, src_device: usize, dest_device: usize, packet: P, size: u32) {
        assert!(self.has_buffer(src_device, size));

        // let is_memory_node = self.num_subnets > 1 && src_device >= self.num_cores;
        let is_memory_node = self.num_subnets > 1 && dest_device >= self.num_cores;
        let subnet = if is_memory_node { 1 } else { 0 };
        println!(
            "{}: {size} bytes from device {src_device} to {dest_device} (subnet {subnet})",
            style(format!("INTERCONN PUSH {}", packet)).bold(),
        );

        // let mut queue = self.output_queue[subnet][src_device][0].lock().unwrap();
        let mut queue = self.output_queue[subnet][dest_device][0].lock().unwrap();
        queue.push_back(packet);
    }

    fn pop(&self, device: usize) -> Option<P> {
        // let icnt_id = self.node_map[&device];
        let icnt_id = device;
        let subnet = if device >= self.num_cores { 1 } else { 0 };

        let mut lock = self.round_robin_turn[subnet][icnt_id].lock().unwrap();
        let mut turn = *lock;
        println!(
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

        // for (int vc=0;(vc<_vcs) && (data==NULL);vc++) {
        //   if (_boundary_buffer[subnet][icntID][turn].HasPacket()) {
        //     data = _boundary_buffer[subnet][icntID][turn].PopPacket();
        //   }
        //   turn++;
        //   if (turn == _vcs) turn = 0;
        // }
        // if (data) {
        //   _round_robin_turn[subnet][icntID] = turn;
        // }
        // let mut queue = self.queue[subnet][device][0].lock().unwrap();
        // queue.pop_front()
    }

    fn transfer(&self) {
        // do nothing
    }

    fn has_buffer(&self, device: usize, size: u32) -> bool {
        // todo!("interconn: has buffer");
        let Some(capacity) = self.capacity else {
            return true;
        };
        // InterconnectInterface::HasBuffer(unsigned deviceID, unsigned int size)
        // bool has_buffer = false;
        // unsigned int n_flits = size / _flit_size + ((size % _flit_size)? 1:0);
        // int icntID = _node_map.find(deviceID)->second;
        //
        // has_buffer = _traffic_manager->_input_queue[0][icntID][0].size() +n_flits <= _input_buffer_capacity;
        //
        // if ((_subnets>1) && deviceID >= _n_shader) // deviceID is memory node
        //   has_buffer = _traffic_manager->_input_queue[1][icntID][0].size() +n_flits <= _input_buffer_capacity;
        //
        // return has_buffer;
        // _traffic_manager->_input_queue[1][icntID][0].size() +n_flits <= _input_buffer_capacity;

        // TODO: using input queue makes no sense as we push into output directly
        let subnet = if device >= self.num_cores { 1 } else { 0 };
        let queue = self.input_queue[subnet][device][0].lock().unwrap();
        queue.len() <= capacity
    }
}

/// Memory interconnect interface between components.
///
/// Functions are not mutable because the interface should
/// implement locking internally
pub trait MemFetchInterface: std::fmt::Debug {
    fn full(&self, size: u32, write: bool) -> bool;

    fn push(&self, fetch: mem_fetch::MemFetch) {
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
            mem_fetch::READ_PACKET_SIZE as u32
        };
        !self.interconn.has_buffer(self.cluster_id, request_size)
    }

    fn push(&self, mut fetch: mem_fetch::MemFetch) {
        // self.core.interconn_simt_to_mem(fetch.get_num_flits(true));
        // self.cluster.interconn_inject_request_packet(fetch);

        {
            let mut stats = self.stats.lock().unwrap();
            // let counters = &mut stats.counters;
            // if fetch.is_write() {
            //     counters.num_mem_write += 1;
            // } else {
            //     counters.num_mem_read += 1;
            // }
            //
            let access_kind = *fetch.access_kind();
            debug_assert_eq!(fetch.is_write(), access_kind.is_write());
            stats.accesses.inc(access_kind, 1);
            // match fetch.access_kind() {
            //     mem_fetch::AccessKind::CONST_ACC_R => {
            //         counters.num_mem_const += 1;
            //     }
            //     mem_fetch::AccessKind::TEXTURE_ACC_R => {
            //         counters.num_mem_texture += 1;
            //     }
            //     mem_fetch::AccessKind::GLOBAL_ACC_R => {
            //         counters.num_mem_read_global += 1;
            //     }
            //     mem_fetch::AccessKind::GLOBAL_ACC_W => {
            //         counters.num_mem_write_global += 1;
            //     }
            //     mem_fetch::AccessKind::LOCAL_ACC_R => {
            //         counters.num_mem_read_local += 1;
            //     }
            //     mem_fetch::AccessKind::LOCAL_ACC_W => {
            //         counters.num_mem_write_local += 1;
            //     }
            //     mem_fetch::AccessKind::INST_ACC_R => {
            //         // TODO: this is wrong
            //         counters.num_mem_load_instructions += 1;
            //     }
            //     mem_fetch::AccessKind::L1_WRBK_ACC => {
            //         counters.num_mem_write_global += 1;
            //     }
            //     mem_fetch::AccessKind::L2_WRBK_ACC => {
            //         counters.num_mem_l2_writeback += 1;
            //     }
            //     mem_fetch::AccessKind::L1_WR_ALLOC_R => {
            //         counters.num_mem_l1_write_allocate += 1;
            //     }
            //     mem_fetch::AccessKind::L2_WR_ALLOC_R => {
            //         counters.num_mem_l2_write_allocate += 1;
            //     }
            //     _ => {}
            // }
        }

        let dest_sub_partition_id = fetch.sub_partition_id();
        let mem_dest = self
            .config
            .mem_id_to_device_id(dest_sub_partition_id as usize);

        println!(
            "cluster {} icnt_inject_request_packet({}) dest sub partition id={} dest mem node={}",
            self.cluster_id, fetch, dest_sub_partition_id, mem_dest
        );

        // The packet size varies depending on the type of request:
        // - For write request and atomic request, packet contains the data
        // - For read request (i.e. not write nor atomic), packet only has control metadata
        let packet_size = if !fetch.is_write() && !fetch.is_atomic() {
            fetch.control_size
        } else {
            // fetch.size()
            fetch.data_size // todo: is that correct now?
        };
        // m_stats->m_outgoing_traffic_stats->record_traffic(mf, packet_size);
        fetch.status = mem_fetch::Status::IN_ICNT_TO_MEM;

        self.interconn
            .push(self.cluster_id, mem_dest, Packet::Fetch(fetch), packet_size);
    }
}

use std::cell::RefCell;
use std::rc::Rc;

#[derive()]
// pub struct L2Interface<P> {
// pub struct L2Interface<I, Q> {
pub struct L2Interface<Q> {
    pub l2_to_dram_queue: Arc<Mutex<Q>>,
    // pub sub_partition_unit: Rc<RefCell<super::MemorySubPartition<I, Q>>>,
    // pub sub_partition_unit: Rc<RefCell<super::MemorySubPartition>>,
    // pub interconn: Arc<dyn Interconnect<P>>,
}

impl<Q> std::fmt::Debug for L2Interface<Q> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("L2Interface").finish()
    }
}

// impl MemFetchInterface for L2Interface<Packet> {
// impl<I, Q> MemFetchInterface for L2Interface<I, Q> {
// impl MemFetchInterface for L2Interface {
impl<Q> MemFetchInterface for L2Interface<Q>
where
    Q: super::Queue<mem_fetch::MemFetch>,
{
    fn full(&self, size: u32, write: bool) -> bool {
        use super::Queue;
        // todo!("l2 interface: full");
        // let request_size = if write { size } else { READ_PACKET_SIZE as u32 };
        // !self.interconn.has_buffer(self.cluster_id, request_size)
        // self.sub_partition_unit.borrow().l2_to_dram_queue.full()
        self.l2_to_dram_queue.lock().unwrap().full()
    }

    fn push(&self, mut fetch: mem_fetch::MemFetch) {
        use super::Queue;
        // todo!("l2 interface: push");
        fetch.set_status(mem_fetch::Status::IN_PARTITION_L2_TO_DRAM_QUEUE, 0);
        // self.sub_partition_unit
        //     .borrow_mut()
        // todo!("l2 interface push to dram queue");
        println!("l2 interface push l2_to_dram_queue");
        self.l2_to_dram_queue.lock().unwrap().enqueue(fetch)
    }
}

#[cfg(test)]
mod tests {
    use crate::config::GPUConfig;
    use color_eyre::eyre;
    use cxx::CxxString;
    use playground::{bindings, bridge};
    use std::ffi::CString;
    use std::path::PathBuf;
    use std::pin::Pin;
    use std::ptr;

    #[test]
    fn test_intersim_config() -> eyre::Result<()> {
        use bridge::interconnect::IntersimConfig;

        let config_file = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("accelsim/gtx1080/config_fermi_islip.icnt");

        let mut config = IntersimConfig::from_file(&config_file)?;

        assert_eq!(config.get_bool("use_map"), false);
        assert_eq!(config.get_int("num_vcs"), 1); // this means vc can only ever be zero
        assert_eq!(config.get_int("ejection_buffer_size"), 0);
        assert_eq!(config.get_string("sim_type"), "gpgpusim");
        assert_eq!(config.get_string("topology"), "fly");
        Ok(())
    }

    #[test]
    fn test_box_interconnect() -> eyre::Result<()> {
        use bridge::interconnect::{BoxInterconnect, Interconnect};

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
        use bridge::interconnect::{Interconnect, InterconnectInterface};

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
