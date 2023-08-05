#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("playground-sys/src/ref/bridge/mem_fetch.hpp");

        type mem_fetch = crate::bridge::types::mem_fetch::mem_fetch;
        type mem_fetch_bridge;

        type mem_fetch_ptr_shim;
        #[must_use] fn get(self: &mem_fetch_ptr_shim) -> *const mem_fetch;

        #[must_use] unsafe fn new_mem_fetch_bridge(ptr: *const mem_fetch) -> SharedPtr<mem_fetch_bridge>;
        #[must_use]
        fn inner(self: &mem_fetch_bridge) -> *const mem_fetch;
    }

    // explicit instantiation for mem_fetch_ptr_shim to implement VecElement
    impl CxxVector<mem_fetch_ptr_shim> {}
}

pub use ffi::*;
