use super::{config, mem_fetch, Packet};
use crate::ported::ldst_unit::READ_PACKET_SIZE;
use std::collections::VecDeque;
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
    pub classes: usize,
    round_robin_turn: Vec<Vec<Mutex<usize>>>,
    queue: Vec<Vec<Vec<Mutex<VecDeque<P>>>>>,
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
        let subnets = 2;
        let nodes = num_cores + num_mems;
        let classes = 1;

        let mut queue: Vec<Vec<Vec<Mutex<VecDeque<P>>>>> = Vec::new();
        let mut round_robin_turn: Vec<Vec<Mutex<usize>>> = Vec::new();

        for subnet in 0..subnets {
            queue.push(Vec::new());
            round_robin_turn.push(Vec::new());

            for node in 0..nodes {
                queue[subnet].push(Vec::new());
                round_robin_turn[subnet].push(Mutex::new(0));

                for class in 0..classes {
                    queue[subnet][node].push(Mutex::new(VecDeque::new()));
                }
            }
        }
        Self {
            capacity,
            num_cores,
            num_mems,
            classes,
            queue,
            round_robin_turn,
        }
    }
}

impl<P> Interconnect<P> for ToyInterconnect<P> {
    fn push(&self, src_device: usize, dest_device: usize, packet: P, size: u32) {
        assert!(self.has_buffer(src_device, size));

        // _traffic_manager->_GeneratePacket( input_icntID, -1, 0 /*class*/, _traffic_manager->_time, subnet, n_flits, packet_type, data, output_icntID);

        let subnet = if src_device >= self.num_cores { 1 } else { 0 };
        let mut queue = self.queue[subnet][src_device][0].lock().unwrap();
        queue.push_back(packet);
    }

    fn pop(&self, device: usize) -> Option<P> {
        let subnet = if device >= self.num_cores { 1 } else { 0 };

        let mut lock = self.round_robin_turn[subnet][device].lock().unwrap();
        let mut turn = *lock;
        for _ in 0..self.classes {
            let mut queue = self.queue[subnet][device][turn].lock().unwrap();
            turn = (turn + 1) % self.classes;
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

        let subnet = if device >= self.num_cores { 1 } else { 0 };
        let queue = self.queue[subnet][device][0].lock().unwrap();
        queue.len() <= capacity
    }
}

// icnt_create = intersim2_create;
// icnt_init = intersim2_init;
// icnt_has_buffer = intersim2_has_buffer;
// icnt_push = intersim2_push;
// icnt_pop = intersim2_pop;
// icnt_transfer = intersim2_transfer;
// icnt_busy = intersim2_busy;
// icnt_display_stats = intersim2_display_stats;
// icnt_display_overall_stats = intersim2_display_overall_stats;
// icnt_display_state = intersim2_display_state;
// icnt_get_flit_size = intersim2_get_flit_size;

// pub trait MemPort {
//     fn full(&self, size: u32, write: bool) -> bool;
//     fn push(&mut self, fetch: MemFetch);
//     fn pop(&mut self) -> Option<MemFetch>;
// }

/// Memory interconnect interface between components.
///
/// Functions are not mutable because the interface should
/// implement locking internally
pub trait MemFetchInterface {
    // fn new() -> Self;
    // fn full(&self, size: u32, write: bool) -> bool;
    // fn push(&self, fetch: MemFetch);
    fn full(&self, size: u32, write: bool) -> bool {
        todo!("mem fetch interface: full");
    }

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

#[derive(Debug)]
pub struct MockCoreMemoryInterface {}
impl MemFetchInterface for MockCoreMemoryInterface {}

#[derive()]
pub struct CoreMemoryInterface<P> {
    // core: Arc<super::core::SIMTCore>,
    // cluster: Weak<super::core::SIMTCoreCluster<Self>>,
    pub config: Arc<config::GPUConfig>,
    pub cluster_id: usize,
    pub interconn: Arc<dyn Interconnect<P>>,
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
        let request_size = if write { size } else { READ_PACKET_SIZE as u32 };
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

// #[derive(Debug, Clone, Default)]
// pub struct Interconnect {}
//
// impl MemPort for Interconnect {
//     fn push(&mut self, fetch: MemFetch) {
//         todo!("interconnect: push fetch {:?}", fetch);
//     }
//
//     fn pop(&mut self) -> Option<MemFetch> {
//         // todo!("interconnect: pop");
//         None
//     }
//
//     fn full(&self, size: u32, write: bool) -> bool {
//         // todo!("interconnect: full");
//         false
//     }
// }

// static void intersim2_create(unsigned int n_shader, unsigned int n_mem) {
//   g_icnt_interface->CreateInterconnect(n_shader, n_mem);
// }
//
// static void intersim2_init() { g_icnt_interface->Init(); }
//
// static bool intersim2_has_buffer(unsigned input, unsigned int size) {
//   return g_icnt_interface->HasBuffer(input, size);
// }
//
// static void intersim2_push(unsigned input, unsigned output, void* data,
//                            unsigned int size) {
//   g_icnt_interface->Push(input, output, data, size);
// }
//
// static void* intersim2_pop(unsigned output) {
//   return g_icnt_interface->Pop(output);
// }

// functional interface to the interconnect

// typedef void (*icnt_create_p)(unsigned n_shader, unsigned n_mem);
// typedef void (*icnt_init_p)();
// typedef bool (*icnt_has_buffer_p)(unsigned input, unsigned int size);
// typedef void (*icnt_push_p)(unsigned input, unsigned output, void* data,
//                             unsigned int size);
// typedef void* (*icnt_pop_p)(unsigned output);
// typedef void (*icnt_transfer_p)();
// typedef bool (*icnt_busy_p)();
// typedef void (*icnt_drain_p)();
// typedef void (*icnt_display_stats_p)();
// typedef void (*icnt_display_overall_stats_p)();
// typedef void (*icnt_display_state_p)(FILE* fp);
// typedef unsigned (*icnt_get_flit_size_p)();
//
// void icnt_wrapper_init() {
//   switch (g_network_mode) {
//     case INTERSIM:
//       // FIXME: delete the object: may add icnt_done wrapper
//       g_icnt_interface = InterconnectInterface::New(g_network_config_filename);
//       icnt_create = intersim2_create;
//       icnt_init = intersim2_init;
//       icnt_has_buffer = intersim2_has_buffer;
//       icnt_push = intersim2_push;
//       icnt_pop = intersim2_pop;
//       icnt_transfer = intersim2_transfer;
//       icnt_busy = intersim2_busy;
//       icnt_display_stats = intersim2_display_stats;
//       icnt_display_overall_stats = intersim2_display_overall_stats;
//       icnt_display_state = intersim2_display_state;
//       icnt_get_flit_size = intersim2_get_flit_size;
//       break;
//     case LOCAL_XBAR:
//       g_localicnt_interface = LocalInterconnect::New(g_inct_config);
//       icnt_create = LocalInterconnect_create;
//       icnt_init = LocalInterconnect_init;
//       icnt_has_buffer = LocalInterconnect_has_buffer;
//       icnt_push = LocalInterconnect_push;
//       icnt_pop = LocalInterconnect_pop;
//       icnt_transfer = LocalInterconnect_transfer;
//       icnt_busy = LocalInterconnect_busy;
//       icnt_display_stats = LocalInterconnect_display_stats;
//       icnt_display_overall_stats = LocalInterconnect_display_overall_stats;
//       icnt_display_state = LocalInterconnect_display_state;
//       icnt_get_flit_size = LocalInterconnect_get_flit_size;
//       break;
//     default:
//       assert(0);
//       break;
//   }
// }
