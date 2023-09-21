#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("playground-sys/src/ref/bridge/memory_partition_unit.hpp");

        type cache_bridge = crate::bridge::cache::cache_bridge;
        type mem_fetch = crate::bridge::mem_fetch::mem_fetch;
        type mem_fetch_ptr_shim = crate::bridge::mem_fetch::mem_fetch_ptr_shim;

        type memory_partition_unit_bridge;
        #[must_use]
        fn get_dram_latency_queue(
            self: &memory_partition_unit_bridge,
        ) -> UniquePtr<CxxVector<mem_fetch_ptr_shim>>;

        #[must_use]
        fn get_last_borrower(self: &memory_partition_unit_bridge) -> i32;

        #[must_use]
        fn get_shared_credit(self: &memory_partition_unit_bridge) -> i32;

        #[must_use]
        fn get_private_credit(self: &memory_partition_unit_bridge) -> &CxxVector<i32>;

        type memory_sub_partition_bridge;
        #[must_use]
        fn get_icnt_L2_queue(
            self: &memory_sub_partition_bridge,
        ) -> UniquePtr<CxxVector<mem_fetch_ptr_shim>>;
        #[must_use]
        fn get_L2_dram_queue(
            self: &memory_sub_partition_bridge,
        ) -> UniquePtr<CxxVector<mem_fetch_ptr_shim>>;
        #[must_use]
        fn get_dram_L2_queue(
            self: &memory_sub_partition_bridge,
        ) -> UniquePtr<CxxVector<mem_fetch_ptr_shim>>;
        #[must_use]
        fn get_L2_icnt_queue(
            self: &memory_sub_partition_bridge,
        ) -> UniquePtr<CxxVector<mem_fetch_ptr_shim>>;

        type rop_delay_shim;
        #[must_use]
        fn get_ready(self: &rop_delay_shim) -> u64;
        #[must_use]
        fn get_fetch(self: &rop_delay_shim) -> *const mem_fetch;

        #[must_use]
        fn get_rop_delay_queue(
            self: &memory_sub_partition_bridge,
        ) -> UniquePtr<CxxVector<rop_delay_shim>>;

        #[must_use]
        fn get_l2_data_cache(self: &memory_sub_partition_bridge) -> SharedPtr<cache_bridge>;

    }

    // explicit instantiation for memory_partition_unit_bridge to implement VecElement
    impl CxxVector<memory_partition_unit_bridge> {}

    // explicit instantiation for memory_sub_partition_bridge to implement VecElement
    impl CxxVector<memory_sub_partition_bridge> {}
}

pub use ffi::*;
