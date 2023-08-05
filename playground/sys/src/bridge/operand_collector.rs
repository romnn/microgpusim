#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("playground-sys/src/ref/bridge/operand_collector.hpp");

        type register_set = crate::bridge::types::register_set::register_set;
        type warp_inst_t = crate::bridge::types::warp_inst::warp_inst_t;
        type input_port_t = crate::bridge::input_port::input_port_t;

        type opndcoll_rfu_t;

        type collector_unit_set;
        #[must_use] fn get_set(self: &collector_unit_set) -> u32;
        #[must_use] fn get_unit(self: &collector_unit_set) -> &collector_unit_t;

        type collector_unit_t;
        #[must_use] fn get_warp_instruction(self: &collector_unit_t) -> *mut warp_inst_t;
        #[must_use] fn is_free(self: &collector_unit_t) -> bool;
        #[must_use] fn get_warp_id(self: &collector_unit_t) -> u32;
        #[must_use] fn get_active_count(self: &collector_unit_t) -> u32;
        #[must_use] fn get_reg_id(self: &collector_unit_t) -> u32;
        #[must_use] fn get_output_register(self: &collector_unit_t) -> *mut register_set;
        #[must_use] fn get_not_ready_mask(self: &collector_unit_t) -> UniquePtr<CxxString>;

        type dispatch_unit_t;
        #[must_use] fn get_set_id(self: &dispatch_unit_t) -> u32;
        #[must_use] fn get_last_cu(self: &dispatch_unit_t) -> u32;
        #[must_use] fn get_next_cu(self: &dispatch_unit_t) -> u32;

        type arbiter_t;
        #[must_use] fn get_last_cu(self: &arbiter_t) -> u32;

        type operand_collector_bridge;
        #[must_use] fn inner(self: &operand_collector_bridge) -> *const opndcoll_rfu_t;
        #[must_use] fn get_arbiter(self: &operand_collector_bridge) -> &arbiter_t;
        #[must_use] fn get_input_ports(self: &operand_collector_bridge) -> &CxxVector<input_port_t>;
        #[must_use] fn get_dispatch_units(self: &operand_collector_bridge) -> &CxxVector<dispatch_unit_t>;
        #[must_use] fn get_collector_units(
            self: &operand_collector_bridge,
        ) -> UniquePtr<CxxVector<collector_unit_set>>;

    }

    // explicit instantiation for operand_collector_bridge to implement SharedPtrTarget
    impl SharedPtr<operand_collector_bridge> {}
}

pub use ffi::*;
