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
