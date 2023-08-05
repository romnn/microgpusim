#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
        include!("playground-sys/src/ref/bridge/warp_inst.hpp");

        type warp_inst_t = crate::bridge::types::warp_inst::warp_inst_t;

        type warp_inst_ptr;
        #[must_use] fn get(self: &warp_inst_ptr) -> *const warp_inst_t;

        type warp_inst_bridge;
        #[must_use]
        unsafe fn new_warp_inst_bridge(ptr: *const warp_inst_t) -> SharedPtr<warp_inst_bridge>;
        #[must_use]
        fn inner(self: &warp_inst_bridge) -> *const warp_inst_t;
    }

    // explicit instantiation for warp_inst_ptr to implement VecElement
    impl CxxVector<warp_inst_ptr> {}
}

pub use ffi::*;
