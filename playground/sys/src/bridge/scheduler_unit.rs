#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("playground-sys/src/ref/bridge/core.hpp");

        type scheduler_unit_ptr;
        #[must_use]
        fn get(self: &scheduler_unit_ptr) -> *const scheduler_unit;

        type scheduler_unit;

        type scheduler_unit_bridge;
        #[must_use] unsafe fn new_scheduler_unit_bridge(
            ptr: *const scheduler_unit,
        ) -> SharedPtr<scheduler_unit_bridge>;
        #[must_use]
        fn inner(self: &scheduler_unit_bridge) -> *const scheduler_unit;
        #[must_use] fn get_prioritized_warp_ids(self: &scheduler_unit_bridge) -> UniquePtr<CxxVector<u32>>;
        #[must_use] fn get_prioritized_dynamic_warp_ids(
            self: &scheduler_unit_bridge,
        ) -> UniquePtr<CxxVector<u32>>;
    }

    // explicit instantiation for scheduler_unit_ptr to implement VecElement
    impl CxxVector<scheduler_unit_ptr> {}
}

pub use ffi::*;
