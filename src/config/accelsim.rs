use clap::Parser;
use color_eyre::eyre;

#[derive(Parser, Debug, Default, Clone, PartialEq, Eq)]
#[clap(
    // trailing_var_arg = true,
    // allow_hyphen_values = true,
    // arg_required_else_help = false
)]
pub struct ShaderCore {
    #[clap(
        long = "gpgpu_simd_model",
        help = "1 = post-dominator",
        default_value = "1"
    )]
    pub gpgpu_simd_model: u32,
    #[clap(
        long = "gpgpu_shader_core_pipeline",
        help = "shader core pipeline config, i.e., {<nthread>:<warpsize>}",
        default_value = "1024:32"
    )]
    pub gpgpu_shader_core_pipeline: String,
    #[clap(
        long = "gpgpu_tex_cache:l1",
        help = "per-shader L1 texture cache  (READ-ONLY) config {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>:<rf>}",
        default_value = "8:128:5,L:R:m:N,F:128:4,128:2"
    )]
    pub gpgpu_const_cache_l1: String,
    #[clap(
        long = "gpgpu_const_cache:l1",
        help = "per-shader L1 constant memory cache  (READ-ONLY) config {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>}",
        default_value = "64:64:2,L:R:f:N,A:2:32,4"
    )]
    pub gpgpu_tex_cache_l1: String,
    #[clap(
        long = "gpgpu_cache:il1",
        help = "shader L1 instruction cache config {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>}",
        default_value = "4:256:4,L:R:f:N,A:2:32,4"
    )]
    pub gpgpu_cache_il1: String,
    #[clap(
        long = "gpgpu_cache:dl1",
        help = "per-shader L1 data cache config {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq> | none}",
        default_value = "none"
    )]
    pub gpgpu_cache_dl1: String,
    #[clap(
        long = "gpgpu_l1_cache_write_ratio",
        help = "L1D write ratio",
        default_value = "0"
    )]
    pub gpgpu_l1_cache_write_ratio: u32,
    #[clap(
        long = "gpgpu_l1_banks",
        help = "The number of L1 cache banks",
        default_value = "1"
    )]
    pub gpgpu_l1_banks: u32,
    #[clap(
        long = "gpgpu_l1_banks_byte_interleaving",
        help = "l1 banks byte interleaving granularity",
        default_value = "32"
    )]
    pub gpgpu_l1_banks_byte_interleaving: u32,
    #[clap(
        long = "gpgpu_l1_banks_hashing_function",
        help = "l1 banks hashing function",
        default_value = "0"
    )]
    pub gpgpu_l1_banks_hashing_function: u32,
    #[clap(
        long = "gpgpu_l1_latency",
        help = "L1 Hit Latency",
        default_value = "1"
    )]
    pub gpgpu_l1_latency: u32,
    #[clap(
        long = "gpgpu_smem_latency",
        help = "smem Latency",
        default_value = "3"
    )]
    pub gpgpu_smem_latency: u32,
    #[clap(
        long = "gpgpu_cache:dl1PrefL1",
        help = "per-shader L1 data cache config {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq> | none}",
        default_value = "none"
    )]
    pub gpgpu_cache_dl1_pref_l1: String,
    #[clap(
        long = "gpgpu_cache:dl1PrefShared",
        help = "per-shader L1 data cache config {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq> | none}",
        default_value = "none"
    )]
    pub gpgpu_cache_dl1_pref_shared: String,
    #[clap(
        long = "gpgpu_gmem_skip_L1D:dl1PrefShared",
        help = "global memory access skip L1D cache (implements -Xptxas -dlcm=cg, default=no skip)",
        default_value = "false"
    )]
    pub gpgpu_gmem_skip_l1d: bool,
    #[clap(
        long = "gpgpu_perfect_mem",
        help = "enable perfect memory mode (no cache miss)",
        default_value = "false"
    )]
    pub gpgpu_perfect_mem: bool,
    #[clap(
        long = "n_regfile_gating_group",
        help = "group of lanes that should be read/written together)",
        default_value = "4"
    )]
    pub n_regfile_gating_group: u32,
    #[clap(
        long = "gpgpu_clock_gated_reg_file",
        help = "enable clock gated reg file for power calculations",
        default_value = "false"
    )]
    pub gpgpu_clock_gated_reg_file: bool,
    #[clap(
        long = "gpgpu_clock_gated_lanes",
        help = "enable clock gated lanes for power calculations",
        default_value = "false"
    )]
    pub gpgpu_clock_gated_lanes: bool,
    #[clap(
        long = "gpgpu_shader_registers",
        help = "Number of registers per shader core. Limits number of concurrent CTAs. (default 8192)",
        default_value = "8192"
    )]
    pub gpgpu_shader_registers: u32,
    #[clap(
        long = "gpgpu_registers_per_block",
        help = "Maximum number of registers per CTA. (default 8192)",
        default_value = "8192"
    )]
    pub gpgpu_registers_per_block: u32,
    #[clap(
        long = "gpgpu_ignore_resources_limitation",
        help = "gpgpu_ignore_resources_limitation (default 0)",
        default_value = "0"
    )]
    pub gpgpu_ignore_resources_limitation: bool,
    #[clap(
        long = "gpgpu_shader_cta",
        help = "Maximum number of concurrent CTAs in shader (default 32)",
        default_value = "32"
    )]
    pub gpgpu_shader_cta: u32,
    #[clap(
        long = "gpgpu_num_cta_barriers",
        help = "Maximum number of named barriers per CTA (default 16)",
        default_value = "16"
    )]
    pub gpgpu_num_cta_barriers: u32,
    #[clap(
        long = "gpgpu_n_clusters",
        help = "number of processing clusters",
        default_value = "10"
    )]
    pub gpgpu_n_clusters: u32,
    #[clap(
        long = "gpgpu_n_cores_per_cluster",
        help = "number of simd cores per cluster",
        default_value = "3"
    )]
    pub gpgpu_n_cores_per_cluster: u32,
    #[clap(
        long = "gpgpu_n_cluster_ejection_buffer_size",
        help = "number of packets in ejection buffer",
        default_value = "8"
    )]
    pub gpgpu_n_cluster_ejection_buffer_size: u32,
    #[clap(
        long = "gpgpu_n_ldst_response_buffer_size",
        help = "number of response packets in ld/st unit ejection buffer",
        default_value = "2"
    )]
    pub gpgpu_n_ldst_response_buffer_size: u32,
    #[clap(
        long = "gpgpu_shmem_per_block",
        help = "Size of shared memory per thread block or CTA (default 48kB)",
        default_value = "49152"
    )]
    pub gpgpu_shmem_per_block: u32,
    #[clap(
        long = "gpgpu_shmem_size",
        help = "Size of shared memory per shader core (default 16kB)",
        default_value = "16384"
    )]
    pub gpgpu_shmem_size: u32,
    #[clap(
        long = "gpgpu_shmem_option",
        help = "Option list of shared memory sizes",
        default_value = "0"
    )]
    pub gpgpu_shmem_option: u32,
    #[clap(
        long = "gpgpu_unified_l1d_size",
        help = "Size of unified data cache(L1D + shared memory) in KB",
        default_value = "0"
    )]
    pub gpgpu_unified_l1d_size: u32,
    #[clap(
        long = "gpgpu_adaptive_cache_config",
        help = "adaptive_cache_config",
        default_value = "false"
    )]
    pub gpgpu_adaptive_cache_config: bool,
    #[clap(
        long = "gpgpu_shmem_sizeDefault",
        help = "Size of shared memory per shader core (default 16kB)",
        default_value = "16384"
    )]
    pub gpgpu_shmem_size_default: u32,
    #[clap(
        long = "gpgpu_shmem_size_PrefL1",
        help = "Size of shared memory per shader core (default 16kB)",
        default_value = "16384"
    )]
    pub gpgpu_shmem_size_pref_l1: u32,
    #[clap(
        long = "gpgpu_shmem_size_PrefShared",
        help = "Size of shared memory per shader core (default 16kB)",
        default_value = "16384"
    )]
    pub gpgpu_shmem_size_pref_shared: u32,
    #[clap(
        long = "gpgpu_shmem_num_banks",
        help = "Number of banks in the shared memory in each shader core (default 16)",
        default_value = "16"
    )]
    pub gpgpu_shmem_num_banks: u32,
    #[clap(
        long = "gpgpu_shmem_limited_broadcast",
        help = "Limit shared memory to do one broadcast per cycle (default on)",
        default_value = "1"
    )]
    pub gpgpu_shmem_limited_broadcast: u32,
    #[clap(
        long = "gpgpu_shmem_warp_parts",
        help = "Number of portions a warp is divided into for shared memory bank conflict check",
        default_value = "2"
    )]
    pub gpgpu_shmem_warp_parts: u32,
    #[clap(
        long = "gpgpu_mem_unit_ports",
        help = "The number of memory transactions allowed per core cycle",
        default_value = "1"
    )]
    pub gpgpu_mem_unit_ports: u32,
    #[clap(
        long = "gpgpu_warpdistro_shader",
        help = "Specify which shader core to collect the warp size distribution from",
        default_value = "-1"
    )]
    pub gpgpu_warpdistro_shader: i32,
    #[clap(
        long = "gpgpu_warp_issue_shader",
        help = "Specify which shader core to collect the warp issue distribution from",
        default_value = "0"
    )]
    pub gpgpu_warp_issue_shader: i32,
    #[clap(
        long = "gpgpu_local_mem_map",
        help = "Mapping from local memory space address to simulated GPU physical address space (default = enabled)",
        default_value = "true"
    )]
    pub gpgpu_local_mem_map: bool,
    #[clap(
        long = "gpgpu_num_reg_banks",
        help = "Number of register banks (default = 8)",
        default_value = "8"
    )]
    pub gpgpu_num_reg_banks: u32,
    #[clap(
        long = "gpgpu_reg_bank_use_warp_id",
        help = "Use warp ID in mapping registers to banks (default = off)",
        default_value = "false"
    )]
    pub gpgpu_reg_bank_use_warp_id: bool,
    #[clap(
        long = "gpgpu_sub_core_model",
        help = "Sub Core Volta/Pascal model (default = off)",
        default_value = "false"
    )]
    pub gpgpu_sub_core_model: bool,
    #[clap(
        long = "gpgpu_enable_specialized_operand_collector",
        help = "enable_specialized_operand_collector",
        default_value = "true"
    )]
    pub gpgpu_enable_specialized_operand_collector: bool,
    #[clap(
        long = "gpgpu_operand_collector_num_units_sp",
        help = "number of collector units (default = 4)",
        default_value = "4"
    )]
    pub gpgpu_operand_collector_num_units_sp: u32,
    #[clap(
        long = "gpgpu_operand_collector_num_units_dp",
        help = "number of collector units (default = 0)",
        default_value = "0"
    )]
    pub gpgpu_operand_collector_num_units_dp: u32,
    #[clap(
        long = "gpgpu_operand_collector_num_units_sfu",
        help = "number of collector units (default = 4)",
        default_value = "4"
    )]
    pub gpgpu_operand_collector_num_units_sfu: u32,
    #[clap(
        long = "gpgpu_operand_collector_num_units_int",
        help = "number of collector units (default = 0)",
        default_value = "0"
    )]
    pub gpgpu_operand_collector_num_units_int: u32,
    #[clap(
        long = "gpgpu_operand_collector_num_units_tensor_core",
        help = "number of collector units (default = 4)",
        default_value = "4"
    )]
    pub gpgpu_operand_collector_num_units_tensor_core: u32,
    #[clap(
        long = "gpgpu_operand_collector_num_units_mem",
        help = "number of collector units (default = 2)",
        default_value = "2"
    )]
    pub gpgpu_operand_collector_num_units_mem: u32,
    #[clap(
        long = "gpgpu_operand_collector_num_units_gen",
        help = "number of collector units (default = 0)",
        default_value = "0"
    )]
    pub gpgpu_operand_collector_num_units_gen: u32,
    #[clap(
        long = "gpgpu_operand_collector_num_in_ports_sp",
        help = "number of collector unit in ports (default = 1)",
        default_value = "1"
    )]
    pub gpgpu_operand_collector_num_in_ports_sp: u32,
    #[clap(
        long = "gpgpu_operand_collector_num_in_ports_dp",
        help = "number of collector unit in ports (default = 0)",
        default_value = "0"
    )]
    pub gpgpu_operand_collector_num_in_ports_dp: u32,
    #[clap(
        long = "gpgpu_operand_collector_num_in_ports_sfu",
        help = "number of collector unit in ports (default = 1)",
        default_value = "1"
    )]
    pub gpgpu_operand_collector_num_in_ports_sfu: u32,
    #[clap(
        long = "gpgpu_operand_collector_num_in_ports_int",
        help = "number of collector unit in ports (default = 0)",
        default_value = "0"
    )]
    pub gpgpu_operand_collector_num_in_ports_int: u32,
    #[clap(
        long = "gpgpu_operand_collector_num_in_ports_tensor_core",
        help = "number of collector unit in ports (default = 1)",
        default_value = "1"
    )]
    pub gpgpu_operand_collector_num_in_ports_tensor_core: u32,
    #[clap(
        long = "gpgpu_operand_collector_num_in_ports_mem",
        help = "number of collector unit in ports (default = 1)",
        default_value = "1"
    )]
    pub gpgpu_operand_collector_num_in_ports_mem: u32,
    #[clap(
        long = "gpgpu_operand_collector_num_in_ports_gen",
        help = "number of collector unit in ports (default = 0)",
        default_value = "0"
    )]
    pub gpgpu_operand_collector_num_in_ports_gen: u32,
    #[clap(
        long = "gpgpu_operand_collector_num_out_ports_sp",
        help = "number of collector unit in ports (default = 1)",
        default_value = "1"
    )]
    pub gpgpu_operand_collector_num_out_ports_sp: u32,
    #[clap(
        long = "gpgpu_operand_collector_num_out_ports_dp",
        help = "number of collector unit in ports (default = 0)",
        default_value = "0"
    )]
    pub gpgpu_operand_collector_num_out_ports_dp: u32,
    #[clap(
        long = "gpgpu_operand_collector_num_out_ports_sfu",
        help = "number of collector unit in ports (default = 1)",
        default_value = "1"
    )]
    pub gpgpu_operand_collector_num_out_ports_sfu: u32,
    #[clap(
        long = "gpgpu_operand_collector_num_out_ports_int",
        help = "number of collector unit in ports (default = 0)",
        default_value = "0"
    )]
    pub gpgpu_operand_collector_num_out_ports_int: u32,
    #[clap(
        long = "gpgpu_operand_collector_num_out_ports_tensor_core",
        help = "number of collector unit in ports (default = 1)",
        default_value = "1"
    )]
    pub gpgpu_operand_collector_num_out_ports_tensor_core: u32,
    #[clap(
        long = "gpgpu_operand_collector_num_out_ports_mem",
        help = "number of collector unit in ports (default = 1)",
        default_value = "1"
    )]
    pub gpgpu_operand_collector_num_out_ports_mem: u32,
    #[clap(
        long = "gpgpu_operand_collector_num_out_ports_gen",
        help = "number of collector unit in ports (default = 0)",
        default_value = "0"
    )]
    pub gpgpu_operand_collector_num_out_ports_gen: u32,
    #[clap(
        long = "gpgpu_coalesce_arch",
        help = "Coalescing arch (GT200 = 13, Fermi = 20)",
        default_value = "13"
    )]
    pub gpgpu_coalesce_arch: u32,
    #[clap(
        long = "gpgpu_num_sched_per_core",
        help = "Number of warp schedulers per core",
        default_value = "1"
    )]
    pub gpgpu_num_sched_per_core: u32,
    #[clap(
        long = "gpgpu_max_insn_issue_per_warp",
        help = "Max number of instructions that can be issued per warp in one cycle by scheduler (either 1 or 2)",
        default_value = "2"
    )]
    pub gpgpu_max_insn_issue_per_warp: u32,
    #[clap(
        long = "gpgpu_dual_issue_diff_exec_units",
        help = "should dual issue use two different execution unit resources (Default = 1)",
        default_value = "true"
    )]
    pub gpgpu_dual_issue_diff_exec_units: bool,
    #[clap(
        long = "gpgpu_simt_core_sim_order",
        help = "Select the simulation order of cores in a cluster (0=Fix, 1=Round-Robin)",
        default_value = "1"
    )]
    pub gpgpu_simt_core_sim_order: u32,
    #[clap(
        long = "gpgpu_pipeline_widths",
        help = "Pipeline widths ID_OC_SP,ID_OC_DP,ID_OC_INT,ID_OC_SFU,ID_OC_MEM,OC_EX_SP,OC_EX_DP,OC_EX_INT,OC_EX_SFU,OC_EX_MEM,EX_WB,ID_OC_TENSOR_CORE,OC_EX_TENSOR_CORE",
        default_value = "1,1,1,1,1,1,1,1,1,1,1,1,1"
    )]
    pub gpgpu_pipeline_widths: String,
    #[clap(
        long = "gpgpu_tensor_core_avail",
        help = "Tensor Core Available (default=0)",
        default_value = "0"
    )]
    pub gpgpu_tensor_core_avail: u32,
    #[clap(
        long = "gpgpu_num_sp_units",
        help = "Number of SP units (default=1)",
        default_value = "1"
    )]
    pub gpgpu_num_sp_units: u32,
    #[clap(
        long = "gpgpu_num_dp_units",
        help = "Number of DP units (default=0)",
        default_value = "0"
    )]
    pub gpgpu_num_dp_units: u32,
    #[clap(
        long = "gpgpu_num_int_units",
        help = "Number of INT units (default=0)",
        default_value = "0"
    )]
    pub gpgpu_num_int_units: u32,
    #[clap(
        long = "gpgpu_num_sfu_units",
        help = "Number of SF units (default=1)",
        default_value = "1"
    )]
    pub gpgpu_num_sfu_units: u32,
    #[clap(
        long = "gpgpu_num_tensor_core_units",
        help = "Number of tensor_core units (default=1)",
        default_value = "0"
    )]
    pub gpgpu_num_tensor_core_units: u32,
    #[clap(
        long = "gpgpu_num_mem_units",
        help = "Number if ldst units (default=1) WARNING: not hooked up to anything",
        default_value = "1"
    )]
    pub gpgpu_num_mem_units: u32,
    #[clap(
        long = "gpgpu_scheduler",
        help = "Scheduler configuration: < lrr | gto | two_level_active > If two_level_active:<num_active_warps>:<inner_prioritization>:<outer_prioritization> For complete list of prioritization values see shader.h enum scheduler_prioritization_type Default: gto",
        default_value = "gto"
    )]
    pub gpgpu_scheduler: String,
    #[clap(
        long = "gpgpu_concurrent_kernel_sm",
        help = "Support concurrent kernels on a SM (default = disabled)",
        default_value = "false"
    )]
    pub gpgpu_concurrent_kernel_sm: bool,
    #[clap(
        long = "gpgpu_perfect_inst_const_cache",
        help = "perfect inst and const cache mode, so all inst and const hits in the cache(default = disabled)",
        default_value = "false"
    )]
    pub gpgpu_perfect_inst_const_cache: bool,
    #[clap(
        long = "gpgpu_inst_fetch_throughput",
        help = "the number of fetched intruction per warp each cycle",
        default_value = "1"
    )]
    pub gpgpu_inst_fetch_throughput: u32,
    #[clap(
        long = "gpgpu_reg_file_port_throughput",
        help = "the number ports of the register file",
        default_value = "1"
    )]
    pub gpgpu_reg_file_port_throughput: u32,

    #[clap(
        long = "specialized_unit_1",
        help = "{<enabled>,<num_units>:<latency>:<initiation>,<ID_OC_SPEC>:<OC_EX_SPEC>,<NAME>}",
        default_value = "0,4,4,4,4,BRA"
    )]
    pub specialized_unit_1: String,
    #[clap(
        long = "specialized_unit_2",
        help = "{<enabled>,<num_units>:<latency>:<initiation>,<ID_OC_SPEC>:<OC_EX_SPEC>,<NAME>}",
        default_value = "0,4,4,4,4,BRA"
    )]
    pub specialized_unit_2: String,
    #[clap(
        long = "specialized_unit_3",
        help = "{<enabled>,<num_units>:<latency>:<initiation>,<ID_OC_SPEC>:<OC_EX_SPEC>,<NAME>}",
        default_value = "0,4,4,4,4,BRA"
    )]
    pub specialized_unit_3: String,
    #[clap(
        long = "specialized_unit_4",
        help = "{<enabled>,<num_units>:<latency>:<initiation>,<ID_OC_SPEC>:<OC_EX_SPEC>,<NAME>}",
        default_value = "0,4,4,4,4,BRA"
    )]
    pub specialized_unit_4: String,
    #[clap(
        long = "specialized_unit_5",
        help = "{<enabled>,<num_units>:<latency>:<initiation>,<ID_OC_SPEC>:<OC_EX_SPEC>,<NAME>}",
        default_value = "0,4,4,4,4,BRA"
    )]
    pub specialized_unit_5: String,
    #[clap(
        long = "specialized_unit_6",
        help = "{<enabled>,<num_units>:<latency>:<initiation>,<ID_OC_SPEC>:<OC_EX_SPEC>,<NAME>}",
        default_value = "0,4,4,4,4,BRA"
    )]
    pub specialized_unit_6: String,
    #[clap(
        long = "specialized_unit_7",
        help = "{<enabled>,<num_units>:<latency>:<initiation>,<ID_OC_SPEC>:<OC_EX_SPEC>,<NAME>}",
        default_value = "0,4,4,4,4,BRA"
    )]
    pub specialized_unit_7: String,
    #[clap(
        long = "specialized_unit_8",
        help = "{<enabled>,<num_units>:<latency>:<initiation>,<ID_OC_SPEC>:<OC_EX_SPEC>,<NAME>}",
        default_value = "0,4,4,4,4,BRA"
    )]
    pub specialized_unit_8: String,
}

#[derive(Parser, Debug, Clone, PartialEq, Eq)]
#[clap(
    trailing_var_arg = true,
    // allow_hyphen_values = true,
    arg_required_else_help = false
)]
pub struct Config {
    // #[clap(short, help = "todo")]
    // pub gpgpu_ptx_instruction_classification: Option<usize>,
    #[clap(flatten)]
    pub shader_core: ShaderCore,

    #[clap(num_args(0..), allow_hyphen_values = true)]
    pub unknown: Vec<String>,
}

use once_cell::sync::Lazy;
use regex::Regex;

static ARGUMENT_REGEX: Lazy<Regex> = Lazy::new(|| {
    let arg = r"([\w\d\-:]+)";
    let single_quoted_string = "(?:'(?:[^\\']|\\.)*')";
    let double_quoted_string = r#"(?:"(?:[^\"]|\\.)*")"#;
    let value_excluding_comment = r"(?:[^#\n]+)";
    let trailing_comment = r"(?:#.*)?";
    let pattern = [
        r"^\s*-{1,2}",
        arg,
        r"\s+(",
        single_quoted_string,
        "|",
        double_quoted_string,
        "|",
        value_excluding_comment,
        ")",
        trailing_comment,
    ];
    let pattern = pattern.join("");
    regex::RegexBuilder::new(&pattern)
        .multi_line(true)
        .build()
        .unwrap()
});

pub fn extract_arguments(config: &str) -> impl Iterator<Item = (&str, &str)> {
    ARGUMENT_REGEX.captures_iter(config).filter_map(|cap| {
        let key = cap.get(1)?.as_str().trim();
        let value = cap.get(2)?.as_str().trim();
        Some((key, value))
    })
}

impl Config {
    pub fn from_config_str(config: impl AsRef<str>) -> eyre::Result<Self> {
        let args = extract_arguments(config.as_ref())
            .flat_map(|(key, value)| [format!("--{key}"), value.to_string()]);
        let args: Vec<String> = ["test".to_string()].into_iter().chain(args).collect();
        dbg!(&args);
        let config = Self::try_parse_from(&args)?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {

    use color_eyre::eyre;
    use similar_asserts as diff;
    use std::path::PathBuf;

    #[test]
    fn test_read_config_file_gtx1080() -> eyre::Result<()> {
        use clap::Parser;
        let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
        let config_path = manifest_dir.join("accelsim/gtx1080/gpgpusim.config");
        let _config = std::fs::read_to_string(config_path)?;
        let config = r"
# --gpgpu_shader_core_pipeline 2048:32
# --gpgpu_simd_model 1
        ";
        let args = super::extract_arguments(config)
            .flat_map(|(key, value)| [format!("--{key}"), value.to_string()]);
        let mut args: std::collections::VecDeque<String> = args.collect();
        args.push_front("test".to_string());

        dbg!(&args);
        // let config = super::Config::from_config_str(config)?;
        diff::assert_eq!(
            // config.shader_core,
            // config.shader_core,
            super::ShaderCore::try_parse_from(&args)?,
            super::ShaderCore {
                gpgpu_simd_model: 1,
                gpgpu_shader_core_pipeline: "1024:32".to_string(),
                gpgpu_const_cache_l1: "8:128:5,L:R:m:N,F:128:4,128:2".to_string(),
                gpgpu_tex_cache_l1: "64:64:2,L:R:f:N,A:2:32,4".to_string(),
                ..super::ShaderCore::default()
            },
        );

        // diff::assert_eq!(
        //     super::Config::from_config_str(config)?,
        //     super::Config {
        //         gpgpu_ptx_instruction_classification: Some(0),
        //         unknown: vec![],
        //     },
        // );
        Ok(())
    }

    #[test]
    fn test_extract_arguments() -> eyre::Result<()> {
        let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
        let config_path = manifest_dir.join("accelsim/gtx1080/gpgpusim.config");
        let config = std::fs::read_to_string(config_path)?;
        let arguments: Vec<_> = super::extract_arguments(&config).collect();
        let expected = vec![
            ("-gpgpu_ptx_instruction_classification", "0"),
            ("-gpgpu_ptx_sim_mode", "0"),
            ("-gpgpu_ptx_force_max_capability", "60"),
            ("-gpgpu_ptx_convert_to_ptxplus", "0"),
            ("-gpgpu_ptx_save_converted_ptxplus", "0"),
            ("-gpgpu_n_clusters", "20"),
            ("-gpgpu_n_cores_per_cluster", "1"),
            ("-gpgpu_n_mem", "8"),
            ("-gpgpu_n_sub_partition_per_mchannel", "2"),
            ("-gpgpu_clock_domains", "1607.0:1607.0:1607.0:2500.0"),
            ("-gpgpu_shader_registers", "65536"),
            ("-gpgpu_occupancy_sm_number", "60"),
            ("-gpgpu_shader_core_pipeline", "2048:32"),
            ("-gpgpu_shader_cta", "32"),
            ("-gpgpu_simd_model", "1"),
            ("-gpgpu_pipeline_widths", "4,0,0,1,1,4,0,0,1,1,6"),
            ("-gpgpu_num_sp_units", "4"),
            ("-gpgpu_num_sfu_units", "1"),
            ("-gpgpu_tensor_core_avail", "0"),
            ("-gpgpu_num_tensor_core_units", "0"),
            ("-ptx_opcode_latency_int", "4,13,4,5,145"),
            ("-ptx_opcode_initiation_int", "1,2,2,2,8"),
            ("-ptx_opcode_latency_fp", "4,13,4,5,39"),
            ("-ptx_opcode_initiation_fp", "1,2,1,1,4"),
            ("-ptx_opcode_latency_dp", "8,19,8,8,330"),
            ("-ptx_opcode_initiation_dp", "1,2,1,1,130"),
            ("-gpgpu_cache:dl1", "N:64:128:6,L:L:m:N:H,A:128:8,8"),
            ("-gpgpu_shmem_size", "98304"),
            ("-gpgpu_gmem_skip_L1D", "1"),
            (
                "-gpgpu_cache:dl2",
                "N:64:128:16,L:B:m:W:L,A:1024:1024,4:0,32",
            ),
            ("-gpgpu_cache:dl2_texture_only", "0"),
            ("-gpgpu_cache:il1", "N:8:128:4,L:R:f:N:L,A:2:48,4"),
            ("-gpgpu_tex_cache:l1", "N:16:128:24,L:R:m:N:L,F:128:4,128:2"),
            ("-gpgpu_const_cache:l1", "N:128:64:2,L:R:f:N:L,A:2:64,4"),
            ("-gpgpu_operand_collector_num_units_sp", "20"),
            ("-gpgpu_operand_collector_num_units_sfu", "4"),
            ("-gpgpu_operand_collector_num_units_mem", "8"),
            ("-gpgpu_operand_collector_num_in_ports_sp", "4"),
            ("-gpgpu_operand_collector_num_out_ports_sp", "4"),
            ("-gpgpu_operand_collector_num_in_ports_sfu", "1"),
            ("-gpgpu_operand_collector_num_out_ports_sfu", "1"),
            ("-gpgpu_operand_collector_num_in_ports_mem", "1"),
            ("-gpgpu_operand_collector_num_out_ports_mem", "1"),
            ("-gpgpu_num_reg_banks", "32"),
            ("-gpgpu_shmem_num_banks", "32"),
            ("-gpgpu_shmem_limited_broadcast", "0"),
            ("-gpgpu_shmem_warp_parts", "1"),
            ("-gpgpu_max_insn_issue_per_warp", "2"),
            ("-network_mode", "1"),
            ("-inter_config_file", "config_fermi_islip.icnt"),
            ("-gpgpu_l2_rop_latency", "120"),
            ("-dram_latency", "100"),
            ("-gpgpu_dram_scheduler", "1"),
            ("-gpgpu_frfcfs_dram_sched_queue_size", "64"),
            ("-gpgpu_dram_return_queue_size", "116"),
            ("-gpgpu_n_mem_per_ctrlr", "1"),
            ("-gpgpu_dram_buswidth", "4"),
            ("-gpgpu_dram_burst_length", "8"),
            ("-dram_data_command_freq_ratio", "4"),
            ("-gpgpu_mem_address_mask", "1"),
            (
                "-gpgpu_mem_addr_mapping",
                "dramid@8;00000000.00000000.00000000.00000000.0000RRRR.RRRRRRRR.RBBBCCCC.BCCSSSSS",
            ),
            (
                "-gpgpu_dram_timing_opt",
                r#""nbk=16:CCD=2:RRD=6:RCD=12:RAS=28:RP=12:RC=40:
                        CL=12:WL=4:CDLR=5:WR=12:nbkgrp=1:CCDL=0:RTPL=0""#,
            ),
            ("-gpgpu_num_sched_per_core", "2"),
            ("-gpgpu_scheduler", "gto"),
            ("-gpgpu_memlatency_stat", "14"),
            ("-gpgpu_runtime_stat", "500"),
            ("-enable_ptx_file_line_stats", "1"),
            ("-visualizer_enabled", "0"),
            ("-power_simulation_enabled", "0"),
        ];
        diff::assert_eq!(have: arguments, want: expected);
        Ok(())
    }
}
