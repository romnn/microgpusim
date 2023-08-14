use crate::bindings;

super::extern_type!(bindings::accelsim_config, "accelsim_config");
super::extern_type!(bindings::pipeline_stage_name_t, "pipeline_stage_name_t");

#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("playground-sys/src/ref/bridge/main.hpp");

        type core_bridge = crate::bridge::core::core_bridge;
        type cluster_bridge = crate::bridge::cluster::cluster_bridge;
        type memory_sub_partition_bridge =
            crate::bridge::memory_partition_unit::memory_sub_partition_bridge;
        type memory_partition_unit_bridge =
            crate::bridge::memory_partition_unit::memory_partition_unit_bridge;

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
        #[must_use]
        fn get_finished_kernel_uid(self: Pin<&mut accelsim_bridge>) -> u32;

        #[must_use]
        fn get_cycle(self: &accelsim_bridge) -> u64;
        #[must_use]
        fn active(self: &accelsim_bridge) -> bool;
        #[must_use]
        fn limit_reached(self: &accelsim_bridge) -> bool;
        #[must_use]
        fn commands_left(self: &accelsim_bridge) -> bool;
        #[must_use]
        fn active_kernels(self: &accelsim_bridge) -> bool;
        #[must_use]
        fn kernels_left(self: &accelsim_bridge) -> bool;

        // iterate over sub partitions
        #[must_use]
        fn get_sub_partitions(self: &accelsim_bridge) -> &CxxVector<memory_sub_partition_bridge>;

        // iterate over memory partitions
        #[must_use]
        fn get_partition_units(self: &accelsim_bridge) -> &CxxVector<memory_partition_unit_bridge>;

        // iterate over all cores
        #[must_use]
        fn get_cores(self: &accelsim_bridge) -> &CxxVector<core_bridge>;

        // iterate over all clusters
        #[must_use]
        fn get_clusters(self: &accelsim_bridge) -> &CxxVector<cluster_bridge>;

        #[must_use]
        fn get_last_cluster_issue(self: &accelsim_bridge) -> u32;

        // NOTE: stat transfer functions defined in stats.cc bridge
    }
}

pub use ffi::*;
