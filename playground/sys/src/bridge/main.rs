use std::io::Write;

use crate::bindings;

super::extern_type!(bindings::accelsim_config, "accelsim_config");

#[cxx::bridge]
mod default {
    // struct mem_fetch_bridge {
    //     ptr: *mut mem_fetch,
    // }

    // struct SharedMemorySubPartition {
    //     ptr: SharedPtr<memory_sub_partition>,
    // }
    //
    // struct MemorySubPartitionShim {
    //     ptr: *mut memory_sub_partition,
    //     // s: SharedPtr<memory_sub_partition>,
    // }

    unsafe extern "C++" {
        include!("playground-sys/src/ref/bridge/main.hpp");

        type mem_fetch = crate::bridge::mem_fetch::mem_fetch;
        type mem_fetch_bridge;

        fn get_mem_fetch(self: &mem_fetch_bridge) -> *mut mem_fetch;

        type memory_sub_partition_bridge;

        // fn get_icnt_L2_queue(self: &memory_sub_partition_bridge) -> Vec<mem_fetch_bridge>;
        fn get_icnt_L2_queue(
            self: &memory_sub_partition_bridge,
        ) -> UniquePtr<CxxVector<mem_fetch_bridge>>;

        // fn get_id(self: &memory_sub_partition) -> u32;
        // fn get_id(self: &memory_sub_partition) -> u32;

        type accelsim_bridge;
        type accelsim_config = crate::bindings::accelsim_config;

        #[must_use]
        fn new_accelsim_bridge(
            config: accelsim_config,
            argv: &[&str],
        ) -> UniquePtr<accelsim_bridge>;
        fn run_to_completion(self: Pin<&mut accelsim_bridge>);
        fn process_commands(self: Pin<&mut accelsim_bridge>);
        fn launch_kernels(self: Pin<&mut accelsim_bridge>);
        fn cycle(self: Pin<&mut accelsim_bridge>);
        fn cleanup_finished_kernel(self: Pin<&mut accelsim_bridge>, kernel_uid: u32);
        fn get_finished_kernel_uid(self: Pin<&mut accelsim_bridge>) -> u32;

        fn get_cycle(self: &accelsim_bridge) -> u64;
        fn active(self: &accelsim_bridge) -> bool;
        fn limit_reached(self: &accelsim_bridge) -> bool;
        fn commands_left(self: &accelsim_bridge) -> bool;
        fn active_kernels(self: &accelsim_bridge) -> bool;
        fn kernels_left(self: &accelsim_bridge) -> bool;

        // iterate over sub partitions
        // fn get_sub_partitions(self: &accelsim_bridge) -> *const *const memory_sub_partition;
        // fn get_sub_partitions_vec(self: &accelsim_bridge) -> &Vec<MemorySubPartitionShim>;
        // fn get_sub_partitions_vec(self: &accelsim_bridge) -> &Vec<memory_sub_partition_shim>;
        fn get_sub_partitions_vec(
            self: &accelsim_bridge,
        ) -> &CxxVector<memory_sub_partition_bridge>;

        // NOTE: stat transfer functions defined in stats.cc bridge
    }
}

pub use default::*;
