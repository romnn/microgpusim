use super::MemFetch;
use std::sync::Arc;

pub trait MemPort {
    fn full(&self, size: u32, write: bool) -> bool;
    fn push(&mut self, fetch: MemFetch);
    fn pop(&mut self) -> Option<MemFetch>;
}

/// Memory interconnect interface between components.
///
/// Functions are not mutable because the interface should
/// implement locking internally
pub trait MemFetchInterface {
    fn new() -> Self;
    fn full(&self, size: u32, write: bool) -> bool;
    fn push(&self, fetch: MemFetch);
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
pub struct CoreMemoryInterface {
    // core: Arc<super::core::SIMTCore>,
    // cluster: Arc<super::core::SIMTCoreCluster>,
}

// impl CoreMemoryInterface {
//     pub fn new() -> Self {
//         Self {}
//     }
// }

impl MemFetchInterface for CoreMemoryInterface {
    fn new() -> Self {
        Self {}
    }

    fn full(&self, size: u32, write: bool) -> bool {
        todo!("core memory interface: full");
        // self.cluster.interconn_injection_buffer_full(size, write)
    }

    fn push(&self, fetch: MemFetch) {
        todo!("core memory interface: push");
        // self.core.interconn_simt_to_mem(fetch.get_num_flits(true));
        // self.cluster.interconn_inject_request_packet(fetch);
    }
}

#[derive(Debug, Clone, Default)]
pub struct Interconnect {}

impl MemPort for Interconnect {
    fn push(&mut self, fetch: MemFetch) {
        todo!("interconnect: push fetch {:?}", fetch);
    }

    fn pop(&mut self) -> Option<MemFetch> {
        // todo!("interconnect: pop");
        None
    }

    fn full(&self, size: u32, write: bool) -> bool {
        // todo!("interconnect: full");
        false
    }
}

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
