use std::io::Write;

use crate::bindings;

super::extern_type!(bindings::accelsim_config, "accelsim_config");

#[cxx::bridge]
mod default {
    unsafe extern "C++" {
        include!("playground-sys/src/ref/bridge/main.hpp");

        type mem_fetch = crate::bridge::mem_fetch::mem_fetch;
        type mem_fetch_bridge;

        #[must_use] fn get_mem_fetch(self: &mem_fetch_bridge) -> *mut mem_fetch;

        type memory_partition_unit_bridge;

        #[must_use] fn get_dram_latency_queue(
            self: &memory_partition_unit_bridge,
        ) -> UniquePtr<CxxVector<mem_fetch_bridge>>;

        type memory_sub_partition_bridge;

        #[must_use] fn get_icnt_L2_queue(
            self: &memory_sub_partition_bridge,
        ) -> UniquePtr<CxxVector<mem_fetch_bridge>>;
        #[must_use] fn get_L2_dram_queue(
            self: &memory_sub_partition_bridge,
        ) -> UniquePtr<CxxVector<mem_fetch_bridge>>;
        #[must_use] fn get_dram_L2_queue(
            self: &memory_sub_partition_bridge,
        ) -> UniquePtr<CxxVector<mem_fetch_bridge>>;
        #[must_use] fn get_L2_icnt_queue(
            self: &memory_sub_partition_bridge,
        ) -> UniquePtr<CxxVector<mem_fetch_bridge>>;

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
        #[must_use] fn get_finished_kernel_uid(self: Pin<&mut accelsim_bridge>) -> u32;

        #[must_use] fn get_cycle(self: &accelsim_bridge) -> u64;
        #[must_use] fn active(self: &accelsim_bridge) -> bool;
        #[must_use] fn limit_reached(self: &accelsim_bridge) -> bool;
        #[must_use] fn commands_left(self: &accelsim_bridge) -> bool;
        #[must_use] fn active_kernels(self: &accelsim_bridge) -> bool;
        #[must_use] fn kernels_left(self: &accelsim_bridge) -> bool;

        // iterate over sub partitions
        #[must_use] fn get_sub_partitions_vec(
            self: &accelsim_bridge,
        ) -> &CxxVector<memory_sub_partition_bridge>;

        // iterate over memory partitions
        #[must_use] fn get_partition_units_vec(
            self: &accelsim_bridge,
        ) -> &CxxVector<memory_partition_unit_bridge>;

        // NOTE: stat transfer functions defined in stats.cc bridge
    }
}

pub use default::*;
