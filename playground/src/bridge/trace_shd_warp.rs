// use crate::bindings;
// use cxx::{type_id, ExternType};
//
// unsafe impl ExternType for bindings::cache_config_params {
//     type Id = type_id!("cache_config_params");
//     type Kind = cxx::kind::Trivial;
// }

#[cxx::bridge]
mod default {
    unsafe extern "C++" {
        include!("playground/src/bridge.hpp");

        // type cache_config_params = crate::bindings::cache_config_params;
        type warp_inst_t;
        type trace_kernel_info_t;
        type trace_shader_core_ctx;
        type trace_shd_warp_t;

        unsafe fn new_trace_shd_warp(
            core: *mut trace_shader_core_ctx,
            warp_size: u32,
        ) -> UniquePtr<trace_shd_warp_t>;

        fn reset(self: Pin<&mut trace_shd_warp_t>);

        fn functional_done(self: &trace_shd_warp_t) -> bool;
        fn waiting(self: Pin<&mut trace_shd_warp_t>) -> bool;
        fn hardware_done(self: &trace_shd_warp_t) -> bool;
        fn done_exit(self: &trace_shd_warp_t) -> bool;
        fn set_done_exit(self: Pin<&mut trace_shd_warp_t>);

        fn get_n_completed(self: &trace_shd_warp_t) -> u32;
        fn set_completed(self: Pin<&mut trace_shd_warp_t>, lane: u32);

        fn get_n_atomic(self: &trace_shd_warp_t) -> u32;
        fn inc_n_atomic(self: Pin<&mut trace_shd_warp_t>);
        fn dec_n_atomic(self: Pin<&mut trace_shd_warp_t>, n: u32);

        fn set_membar(self: Pin<&mut trace_shd_warp_t>);
        fn clear_membar(self: Pin<&mut trace_shd_warp_t>);
        fn get_membar(self: &trace_shd_warp_t) -> bool;

        fn get_pc(self: Pin<&mut trace_shd_warp_t>) -> u64;
        fn get_kernel_info(self: &trace_shd_warp_t) -> *mut trace_kernel_info_t;

        unsafe fn ibuffer_fill(
            self: Pin<&mut trace_shd_warp_t>,
            slot: u32,
            warp: *const warp_inst_t,
        );
        fn ibuffer_flush(self: Pin<&mut trace_shd_warp_t>);
        fn ibuffer_empty(self: &trace_shd_warp_t) -> bool;
        unsafe fn ibuffer_next_inst(self: &trace_shd_warp_t) -> *const warp_inst_t;
        fn ibuffer_next_valid(self: &trace_shd_warp_t) -> bool;
        fn ibuffer_free(self: Pin<&mut trace_shd_warp_t>);
        fn ibuffer_step(self: Pin<&mut trace_shd_warp_t>);

        fn imiss_pending(self: &trace_shd_warp_t) -> bool;
        fn set_imiss_pending(self: Pin<&mut trace_shd_warp_t>);
        fn clear_imiss_pending(self: Pin<&mut trace_shd_warp_t>);

        fn stores_done(self: &trace_shd_warp_t) -> bool;
        fn inc_store_req(self: Pin<&mut trace_shd_warp_t>);
        fn dec_store_req(self: Pin<&mut trace_shd_warp_t>);

        // fn num_inst_in_buffer(self: &trace_shd_warp_t) -> u32;
        // fn num_inst_in_pipeline(self: &trace_shd_warp_t) -> u32;
        // fn num_issued_inst_in_pipeline(self: &trace_shd_warp_t) -> u32;
        fn inst_in_pipeline(self: &trace_shd_warp_t) -> bool;
        fn inc_inst_in_pipeline(self: Pin<&mut trace_shd_warp_t>);
        fn dec_inst_in_pipeline(self: Pin<&mut trace_shd_warp_t>);

        fn get_cta_id(self: &trace_shd_warp_t) -> u32;
        fn get_dynamic_warp_id(self: &trace_shd_warp_t) -> u32;
        fn get_warp_id(self: &trace_shd_warp_t) -> u32;
    }
}

pub use default::*;
