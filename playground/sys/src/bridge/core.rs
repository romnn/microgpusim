use crate::bindings;

super::extern_type!(bindings::pending_register_writes, "pending_register_writes");

#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("playground-sys/src/ref/bridge/core.hpp");

        type register_set = crate::bridge::types::register_set::register_set;
        type register_set_ptr = crate::bridge::register_set::register_set_ptr;
        type register_set_bridge = crate::bridge::register_set::register_set_bridge;

        type scheduler_unit_ptr = crate::bridge::scheduler_unit::scheduler_unit_ptr;
        type operand_collector_bridge = crate::bridge::operand_collector::operand_collector_bridge;

        type core_bridge;

        #[must_use]
        fn get_functional_unit_issue_register_sets(
            self: &core_bridge,
        ) -> UniquePtr<CxxVector<register_set_ptr>>;

        #[must_use]
        fn get_functional_unit_simd_pipeline_register_sets(
            self: &core_bridge,
        ) -> UniquePtr<CxxVector<register_set_ptr>>;

        #[must_use]
        fn get_functional_unit_occupied_slots(
            self: &core_bridge,
        ) -> UniquePtr<CxxVector<CxxString>>;

        #[must_use]
        fn get_scheduler_units(self: &core_bridge) -> UniquePtr<CxxVector<scheduler_unit_ptr>>;
        #[must_use]
        fn get_operand_collector(self: &core_bridge) -> SharedPtr<operand_collector_bridge>;

        type pending_register_writes = crate::bindings::pending_register_writes;

        #[must_use]
        fn get_pending_register_writes(
            self: &core_bridge,
        ) -> UniquePtr<CxxVector<pending_register_writes>>;
    }

    // explicit instantiation for core_bridge to implement VecElement
    impl CxxVector<core_bridge> {}
    // explicit instantiation for pending_register_writes to implement VecElement
    impl CxxVector<pending_register_writes> {}
}

pub use ffi::*;
