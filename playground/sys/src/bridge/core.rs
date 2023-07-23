#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("playground-sys/src/ref/bridge/core.hpp");

        type register_set_ptr = crate::bridge::register_set::register_set_ptr;
        type scheduler_unit_ptr = crate::bridge::scheduler_unit::scheduler_unit_ptr;
        type operand_collector_bridge = crate::bridge::operand_collector::operand_collector_bridge;

        type core_bridge;
        #[must_use]
        fn get_register_sets(self: &core_bridge) -> UniquePtr<CxxVector<register_set_ptr>>;
        fn get_scheduler_units(self: &core_bridge) -> UniquePtr<CxxVector<scheduler_unit_ptr>>;
        fn get_operand_collector(self: &core_bridge) -> SharedPtr<operand_collector_bridge>;
    }

    // explicit instantiation for core_bridge to implement VecElement
    impl CxxVector<core_bridge> {}
}

pub use ffi::*;
