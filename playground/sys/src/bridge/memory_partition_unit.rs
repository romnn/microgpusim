#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("playground-sys/src/ref/bridge/memory_partition_unit.hpp");

        type mem_fetch_ptr_shim = crate::bridge::mem_fetch::mem_fetch_ptr_shim;

        type memory_partition_unit_bridge;
        #[must_use]
        fn get_dram_latency_queue(
            self: &memory_partition_unit_bridge,
        ) -> UniquePtr<CxxVector<mem_fetch_ptr_shim>>;

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
    }

    // explicit instantiation for memory_partition_unit_bridge to implement VecElement
    impl CxxVector<memory_partition_unit_bridge> {}

    // explicit instantiation for memory_sub_partition_bridge to implement VecElement
    impl CxxVector<memory_sub_partition_bridge> {}
}

pub use ffi::*;
