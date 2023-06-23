use super::{config, mem_fetch, Packet};
use console::style;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, Weak};

/// Interconnect is a general interconnect
///
/// Functions are not mutable because the interface should
/// implement locking internally
pub trait Interconnect<P> {
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

impl<P> Interconnect<P> for ToyInterconnect<P> {
    fn push(&self, src_device: usize, dest_device: usize, packet: P, size: u32) {
        assert!(self.has_buffer(src_device, size));

        // let is_memory_node = self.num_subnets > 1 && src_device >= self.num_cores;
        let is_memory_node = self.num_subnets > 1 && dest_device >= self.num_cores;
        let subnet = if is_memory_node { 1 } else { 0 };
        println!(
            "{}: {size} bytes from device {src_device} to {dest_device} (subnet {subnet})",
            style("INTERCONN PUSH").bold(),
        );

        // let mut queue = self.output_queue[subnet][src_device][0].lock().unwrap();
        let mut queue = self.output_queue[subnet][dest_device][0].lock().unwrap();
        queue.push_back(packet);
    }

    fn pop(&self, device: usize) -> Option<P> {
        // let icnt_id = self.node_map[&device];
        let icnt_id = device;
        let subnet = if device >= self.num_cores { 1 } else { 0 };
        println!(
            "{}: from device {device} (device={device}, id={icnt_id}, subnet={subnet})",
            style("INTERCONN POP").bold()
        );

        let mut lock = self.round_robin_turn[subnet][icnt_id].lock().unwrap();
        let mut turn = *lock;
        for _ in 0..self.num_classes {
            dbg!(&turn);
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

// pub trait MemPort {
//     fn full(&self, size: u32, write: bool) -> bool;
//     fn push(&mut self, fetch: MemFetch);
//     fn pop(&mut self) -> Option<MemFetch>;
// }

/// Memory interconnect interface between components.
///
/// Functions are not mutable because the interface should
/// implement locking internally
pub trait MemFetchInterface: std::fmt::Debug {
    // fn new() -> Self;
    // fn full(&self, size: u32, write: bool) -> bool;
    // fn push(&self, fetch: MemFetch);
    fn full(&self, size: u32, write: bool) -> bool;
    // -> bool {
    //     todo!("mem fetch interface: full");
    // }

    fn push(&self, fetch: mem_fetch::MemFetch) {
        todo!("mem fetch interface: full");
    }
}

// #[derive(Debug)]
// pub struct PerfectMemoryInterface { }
//
// impl MemFetchInterface for PerfectMemoryInterface {
//     fn full(&self, size: u32, write: bool) -> bool {
//         todo!("core memory interface: full");
//         // self.cluster.interconn_injection_buffer_full(size, write)
//     }
//
//     fn push(&mut self, fetch: MemFetch) {
//         todo!("core memory interface: push");
//         // self.core.interconn_simt_to_mem(fetch.get_num_flits(true));
//         // self.cluster.interconn_inject_request_packet(fetch);
//     }
// }

// #[derive(Debug)]
// pub struct MockCoreMemoryInterface {}
// impl MemFetchInterface for MockCoreMemoryInterface {}

#[derive()]
pub struct CoreMemoryInterface<P> {
    // core: Arc<super::core::SIMTCore>,
    // cluster: Weak<super::core::SIMTCoreCluster<Self>>,
    pub config: Arc<config::GPUConfig>,
    pub cluster_id: usize,
    pub interconn: Arc<dyn Interconnect<P>>,
}

impl<P> std::fmt::Debug for CoreMemoryInterface<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("CoreMemoryInterface").finish()
    }
}

// impl<P> CoreMemoryInterface<P> {
//     // pub fn new(cluster: Weak<super::core::SIMTCoreCluster<Self>>) -> Self {
//     pub fn new(interconn: Arc<dyn Interconnect<P>>) -> Self {
//         Self { interconn }
//     }
// }

// impl MemFetchInterface for CoreMemoryInterface {
// impl<P> MemFetchInterface for CoreMemoryInterface<P> {
impl MemFetchInterface for CoreMemoryInterface<Packet> {
    fn full(&self, size: u32, write: bool) -> bool {
        // todo!("core memory interface: full");
        let request_size = if write {
            size
        } else {
            mem_fetch::READ_PACKET_SIZE as u32
        };
        !self.interconn.has_buffer(self.cluster_id, request_size)
    }

    fn push(&self, mut fetch: mem_fetch::MemFetch) {
        // todo!("core memory interface: push");
        // self.core.interconn_simt_to_mem(fetch.get_num_flits(true));
        // self.cluster.interconn_inject_request_packet(fetch);

        let packet_size = if fetch.is_write() && fetch.is_atomic() {
            fetch.control_size
        } else {
            fetch.data_size
        };
        // m_stats->m_outgoing_traffic_stats->record_traffic(mf, packet_size);
        let dest = fetch.sub_partition_id();
        fetch.status = mem_fetch::Status::IN_ICNT_TO_MEM;

        let packet = Packet::Fetch(fetch);

        // if !fetch.is_write() && !fetch.is_atomic() {
        self.interconn.push(
            self.cluster_id,
            self.config.mem_id_to_device_id(dest as usize),
            packet,
            packet_size,
        );
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

    // #[test]
    // fn test_intersim_config() -> eyre::Result<()> {
    //     use bridge::interconnect::{new_intersim_config, IntersimConfig};
    //     let mut config = new_intersim_config();
    //
    //     let config_file = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    //         .join("accelsim/gtx1080/config_fermi_islip.icnt");
    //
    //     let config_file = config_file.canonicalize()?.to_string_lossy().to_string();
    //     cxx::let_cxx_string!(config_file = config_file);
    //     dbg!(&config_file);
    //     config.pin_mut().ParseFile(&config_file);
    //
    //     cxx::let_cxx_string!(use_map_field = "use_map");
    //     dbg!(config.GetInt(&use_map_field) != 0);
    //     cxx::let_cxx_string!(num_vcs_field = "num_vcs");
    //     dbg!(config.GetInt(&num_vcs_field));
    //     cxx::let_cxx_string!(ejection_buffer_size_field = "ejection_buffer_size");
    //     dbg!(&config.GetInt(&ejection_buffer_size_field));
    //     cxx::let_cxx_string!(sim_type_field = "sim_type");
    //     dbg!(&config.GetStr(&sim_type_field));
    //     cxx::let_cxx_string!(topology_field = "topology");
    //     dbg!(&config.GetStr(&topology_field));
    //
    //     assert!(false);
    //     Ok(())
    // }

    fn gtx_1080_interconn_config() -> eyre::Result<CString> {
        let config_file = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("accelsim/gtx1080/config_fermi_islip.icnt");
        dbg!(&config_file);
        let config_file = CString::new(&*config_file.canonicalize()?.to_string_lossy())?;
        Ok(config_file)
    }

    // #[test]
    // fn test_box_interconnect() -> eyre::Result<()> {
    //     use bridge::interconnect::{new_box_interconnect, BoxInterconnect};
    //
    //     let config = GPUConfig::default();
    //     let num_clusters = config.num_simt_clusters;
    //     let num_mem_sub_partitions = config.total_sub_partitions();
    //     dbg!(&num_clusters);
    //     dbg!(&num_mem_sub_partitions);
    //
    //     let config_file = gtx_1080_interconn_config()?;
    //     let mut interconn = unsafe { new_box_interconnect(config_file.as_ptr()) };
    //
    //     interconn
    //         .pin_mut()
    //         .CreateInterconnect(num_clusters as u32, num_mem_sub_partitions as u32);
    //     interconn.pin_mut().Init();
    //
    //     let num_nodes = interconn.GetNumNodes();
    //     let num_shaders = interconn.GetNumShaders();
    //     let num_memories = interconn.GetNumMemories();
    //     dbg!(&num_nodes);
    //     dbg!(&num_shaders);
    //     dbg!(&num_memories);
    //
    //     // uses a k-ary n-fly bufferfly network
    //     // meaning k**n terminals
    //     //
    //     // for the GTX 1080: this means:
    //     // => k=50 port degree of switch
    //     // => n=1 number of stages
    //     // => so essentially a single switch for all 50 nodes?
    //
    //     // _k = config.GetInt("k");
    //     // _n = config.GetInt("n");
    //     //
    //     // gK = _k;
    //     // gN = _n;
    //     //
    //     // _nodes = powi(_k, _n);
    //     //
    //     // // n stages of k^(n-1) k x k switches
    //     // _size = _n * powi(_k, _n - 1);
    //     //
    //     // // n-1 sets of wiring between the stages
    //     // _channels = (_n - 1) * _nodes;
    //
    //     // _subnets = _icnt_config->GetInt("subnets");
    //     // _net[subnet_id] = Network::New(*_icnt_config, name.str());
    //     // _traffic_manager = static_cast<GPUTrafficManager *>(
    //     //       TrafficManager::New(*_icnt_config, _net));
    //     // _nodes = _net[0]->NumNodes();
    //
    //     // check sending and receiving
    //     // interconn.DisplayMap((num_nodes as f32).sqrt() as u32, num_nodes);
    //
    //     // _CreateNodeMap(_n_shader, _n_mem, _traffic_manager->_nodes,
    //     //          _icnt_config->GetInt("use_map"));
    //
    //     // dbg!(&interconn);
    //     // let core = ptr::null_mut();
    //     // let warp_size = 32;
    //     // let mut warp = unsafe { new_trace_shd_warp(core, warp_size) };
    //     // warp.pin_mut().reset();
    //     // dbg!(&warp.get_n_completed());
    //     // dbg!(&warp.hardware_done());
    //     // dbg!(&warp.functional_done());
    //     assert!(false);
    //     Ok(())
    // }

    #[test]
    fn test_interconnect_interface() -> eyre::Result<()> {
        // use bridge::interconnect::{c_void, new_interconnect_interface};
        use bridge::interconnect::{Interconnect, InterconnectInterface};

        let config = GPUConfig::default();
        let num_clusters = config.num_simt_clusters;
        let num_mem_sub_partitions = config.total_sub_partitions();
        dbg!(&num_clusters);
        dbg!(&num_mem_sub_partitions);

        // let config_file = gtx_1080_interconn_config()?;
        let config_file = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("accelsim/gtx1080/config_fermi_islip.icnt");
        dbg!(&config_file);

        let mut interconn: InterconnectInterface<u32> = InterconnectInterface::new(
            &config_file,
            num_clusters as u32,
            num_mem_sub_partitions as u32,
        );
        // let mut interconn = unsafe { new_interconnect_interface(config_file.as_ptr()) };

        // interconn
        //     .pin_mut()
        //     .CreateInterconnect(num_clusters as u32, num_mem_sub_partitions as u32);
        // interconn.pin_mut().Init();

        let num_nodes = interconn.num_nodes();
        let num_shaders = interconn.num_shaders();
        let num_memories = interconn.num_memories();
        dbg!(&num_nodes);
        dbg!(&num_shaders);
        dbg!(&num_memories);

        // send from core to memory
        let core_node = 0;
        let mem_node = num_shaders;

        let send_data = 42;
        interconn.push(core_node, mem_node, Box::new(send_data));
        // unsafe {
        //     interconn.pin_mut().Push(
        //         core_node,
        //         mem_node,
        //         (&mut send_data as *mut u32) as *mut c_void,
        //         std::mem::size_of_val(&send_data) as u32,
        //     )
        // };

        // let mut recv_data: *mut c_void = ptr::null_mut();
        // let mut recv_data: Option<Box<u32>> = None;
        // for _ in 0..100 {
        //     recv_data = interconn.pop(mem_node);
        //     if recv_data.is_some() {
        //         break;
        //     }
        //     interconn.advance();
        // }
        let (_, recv_data) = interconn.must_pop(mem_node).unwrap();
        // let recv_data =
        // assert!(recv_data.is_some());
        // for _ in 0..100 {
        //     recv_data = unsafe { interconn.pin_mut().Pop(mem_node) };
        //     if !recv_data.is_null() {
        //         break;
        //     }
        //     interconn.pin_mut().Advance();
        // }
        // assert!(!recv_data.is_null());
        // let recv_data: u32 = unsafe { *(recv_data as *mut u32) };
        // dbg!(&recv_data);

        assert_eq!(send_data, *recv_data);

        assert!(false);
        Ok(())
    }

    #[test]
    fn test_interconnect() {}
}
