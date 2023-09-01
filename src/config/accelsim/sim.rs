use super::Boolean;
use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[clap()]
pub struct SimConfig {
    #[clap(
        long = "gpgpu_max_cycle",
        help = "terminates gpu simulation early (0 = no limit)",
        default_value = "0"
    )]
    pub gpu_max_cycle_opt: u64,
    #[clap(
        long = "gpgpu_max_insn",
        help = "terminates gpu simulation early (0 = no limit)",
        default_value = "0"
    )]
    pub gpu_max_insn_opt: u64,
    #[clap(
        long = "gpgpu_max_cta",
        help = "terminates gpu simulation early (0 = no limit)",
        default_value = "0"
    )]
    pub gpu_max_cta_opt: u32,
    #[clap(
        long = "gpgpu_max_completed_cta",
        help = "terminates gpu simulation early (0 = no limit)",
        default_value = "0"
    )]
    pub gpu_max_completed_cta_opt: u32,
    #[clap(
        long = "gpgpu_runtime_stat",
        help = "display runtime statistics such as dram utilization {<freq>:<flag>}",
        default_value = "10000:0"
    )]
    pub gpgpu_runtime_stat: String,
    #[clap(
        long = "liveness_message_freq",
        help = "Minimum number of seconds between simulation liveness messages (0 = always print)",
        default_value = "1"
    )]
    pub liveness_message_freq: u64,
    #[clap(
        long = "gpgpu_compute_capability_major",
        help = "Major compute capability version number",
        default_value = "7"
    )]
    pub gpgpu_compute_capability_major: u32,
    #[clap(
        long = "gpgpu_compute_capability_minor",
        help = "Minor compute capability version number",
        default_value = "0"
    )]
    pub gpgpu_compute_capability_minor: u32,
    #[clap(
        long = "gpgpu_flush_l1_cache",
        help = "Flush L1 cache at the end of each kernel call",
        default_value = "0"
    )]
    pub gpgpu_flush_l1_cache: Boolean,
    #[clap(
        long = "gpgpu_flush_l2_cache",
        help = "Flush L2 cache at the end of each kernel call",
        default_value = "0"
    )]
    pub gpgpu_flush_l2_cache: Boolean,
    #[clap(
        long = "gpgpu_deadlock_detect",
        help = "Stop the simulation at deadlock (1=on (default), 0=off)",
        default_value = "1"
    )]
    pub gpu_deadlock_detect: Boolean,
    #[clap(
        long = "gpgpu_ptx_instruction_classification",
        help = "if enabled will classify ptx instruction types per kernel (Max 255 kernels now)",
        default_value = "0"
    )]
    pub gpgpu_ptx_instruction_classification: u32,
    #[clap(
        long = "gpgpu_ptx_sim_mode",
        help = "Select between Performance (default) or Functional simulation (1)",
        default_value = "0"
    )]
    pub g_ptx_sim_mode: u32,
    #[clap(
        long = "gpgpu_clock_domains",
        help = "Clock Domain Frequencies in MhZ {<Core Clock>:<ICNT Clock>:<L2 Clock>:<DRAM Clock>}",
        default_value = "500.0:2000.0:2000.0:2000.0"
    )]
    pub gpgpu_clock_domains: String,
    #[clap(
        long = "gpgpu_max_concurrent_kernel",
        help = "maximum kernels that can run concurrently on GPU, set this value according to max resident grids for your compute capability",
        default_value = "32"
    )]
    pub max_concurrent_kernel: u32,
    #[clap(
        long = "gpgpu_cflog_interval",
        help = "Interval between each snapshot in control flow logger",
        default_value = "0"
    )]
    pub gpgpu_cflog_interval: u32,
    #[clap(
        long = "visualizer_enabled",
        help = "Turn on visualizer output (1=On, 0=Off)",
        default_value = "1"
    )]
    pub g_visualizer_enabled: Boolean,
    #[clap(
        long = "visualizer_outputfile",
        help = "Specifies the output log file for visualizer"
    )]
    pub g_visualizer_filename: Option<String>,
    #[clap(
        long = "visualizer_zlevel",
        help = "Compression level of the visualizer output log (0=no comp, 9=highest)",
        default_value = "6"
    )]
    pub g_visualizer_zlevel: u32,
    #[clap(
        long = "gpgpu_stack_size_limit",
        help = "GPU thread stack size",
        default_value = "1024"
    )]
    pub stack_size_limit: u32,
    #[clap(
        long = "gpgpu_heap_size_limit",
        help = "GPU malloc heap size",
        default_value = "8388608"
    )]
    pub heap_size_limit: u32,
    #[clap(
        long = "gpgpu_runtime_sync_depth_limit",
        help = "GPU device runtime synchronize depth",
        default_value = "2"
    )]
    pub runtime_sync_depth_limit: u32,
    #[clap(
        long = "gpgpu_runtime_pending_launch_count_limit",
        help = "GPU device runtime pending launch count",
        default_value = "2048"
    )]
    pub runtime_pending_launch_count_limit: u32,
    #[clap(long = "trace_enabled", help = "Turn on traces", default_value = "0")]
    pub trace_enabled: Boolean,
    #[clap(
        long = "trace_components",
        help = "comma seperated list of traces to enable. Complete list found in trace_streams.tup. Default none",
        default_value = "none"
    )]
    pub trace_config_str: String,
    #[clap(
        long = "trace_sampling_core",
        help = "The core which is printed using CORE_DPRINTF. Default 0",
        default_value = "0"
    )]
    pub trace_sampling_core: u32,
    #[clap(
        long = "trace_sampling_memory_partition",
        help = "The memory partition which is printed using MEMPART_DPRINTF. Default -1 (i.e. all)",
        default_value = "-1"
    )]
    pub trace_sampling_memory_partition: i32,
    //   gpgpu_ctx->stats->ptx_file_line_stats_options(opp);
    //
    #[clap(
        long = "gpgpu_kernel_launch_latency",
        help = "Kernel launch latency in cycles. Default: 0",
        default_value = "0"
    )]
    pub g_kernel_launch_latency: u32,
    #[clap(long = "gpgpu_cdp_enabled", help = "Turn on CDP", default_value = "0")]
    pub g_cdp_enabled: Boolean,
    #[clap(
        long = "gpgpu_TB_launch_latency",
        help = "thread block launch latency in cycles. Default: 0",
        default_value = "0"
    )]
    pub g_tb_launch_latency: u32,
    // gpgpu_functional_sim_config::reg_options(opp);
    //   m_shader_config.reg_options(opp);
    //   m_memory_config.reg_options(opp);
    //   power_config::reg_options(opp);
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            gpu_max_cycle_opt: 0,
            gpu_max_insn_opt: 0,
            gpu_max_cta_opt: 0,
            gpu_max_completed_cta_opt: 0,
            gpgpu_runtime_stat: "10000:0".to_string(),
            liveness_message_freq: 1,
            gpgpu_compute_capability_major: 7,
            gpgpu_compute_capability_minor: 0,
            gpgpu_flush_l1_cache: false.into(),
            gpgpu_flush_l2_cache: false.into(),
            gpu_deadlock_detect: true.into(),
            gpgpu_ptx_instruction_classification: 0,
            g_ptx_sim_mode: 0,
            gpgpu_clock_domains: "500.0:2000.0:2000.0:2000.0".to_string(),
            max_concurrent_kernel: 32,
            gpgpu_cflog_interval: 0,
            g_visualizer_enabled: true.into(),
            g_visualizer_filename: None,
            g_visualizer_zlevel: 6,
            stack_size_limit: 1024,
            heap_size_limit: 8_388_608,
            runtime_sync_depth_limit: 2,
            runtime_pending_launch_count_limit: 2048,
            trace_enabled: false.into(),
            trace_config_str: "none".to_string(),
            trace_sampling_core: 0,
            trace_sampling_memory_partition: -1,
            g_kernel_launch_latency: 0,
            g_cdp_enabled: false.into(),
            g_tb_launch_latency: 0,
        }
    }
}
