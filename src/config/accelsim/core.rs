use super::Boolean;
use clap::Parser;

#[derive(Parser, Debug, Clone, PartialEq, Eq)]
#[clap(
    // trailing_var_arg = true,
    // allow_hyphen_values = true,
    // arg_required_else_help = false
)]
pub struct CoreConfig {
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
    pub gpgpu_tex_cache_l1: String,
    #[clap(
        long = "gpgpu_const_cache:l1",
        help = "per-shader L1 constant memory cache  (READ-ONLY) config {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>}",
        default_value = "64:64:2,L:R:f:N,A:2:32,4"
    )]
    pub gpgpu_const_cache_l1: String,
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
        long = "gpgpu_gmem_skip_L1D",
        help = "global memory access skip L1D cache (implements -Xptxas -dlcm=cg, default=no skip)",
        default_value = "false"
    )]
    pub gpgpu_gmem_skip_l1d: Boolean,
    #[clap(
        long = "gpgpu_perfect_mem",
        help = "enable perfect memory mode (no cache miss)",
        default_value = "false"
    )]
    pub gpgpu_perfect_mem: Boolean,
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
    pub gpgpu_clock_gated_reg_file: Boolean,
    #[clap(
        long = "gpgpu_clock_gated_lanes",
        help = "enable clock gated lanes for power calculations",
        default_value = "false"
    )]
    pub gpgpu_clock_gated_lanes: Boolean,
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
        // value_parser = super::BoolParser{},
        default_value = "0"
    )]
    pub gpgpu_ignore_resources_limitation: Boolean,
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
    pub gpgpu_adaptive_cache_config: Boolean,
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
    pub gpgpu_local_mem_map: Boolean,
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
    pub gpgpu_reg_bank_use_warp_id: Boolean,
    #[clap(
        long = "gpgpu_sub_core_model",
        help = "Sub Core Volta/Pascal model (default = off)",
        default_value = "false"
    )]
    pub gpgpu_sub_core_model: Boolean,
    #[clap(
        long = "gpgpu_enable_specialized_operand_collector",
        help = "enable_specialized_operand_collector",
        default_value = "true"
    )]
    pub gpgpu_enable_specialized_operand_collector: Boolean,
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
    pub gpgpu_dual_issue_diff_exec_units: Boolean,
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
    pub gpgpu_concurrent_kernel_sm: Boolean,
    #[clap(
        long = "gpgpu_perfect_inst_const_cache",
        help = "perfect inst and const cache mode, so all inst and const hits in the cache(default = disabled)",
        default_value = "false"
    )]
    pub gpgpu_perfect_inst_const_cache: Boolean,
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

impl Default for CoreConfig {
    fn default() -> Self {
        Self {
            gpgpu_simd_model: 1,
            gpgpu_shader_core_pipeline: "1024:32".to_string(),
            gpgpu_tex_cache_l1: "8:128:5,L:R:m:N,F:128:4,128:2".to_string(),
            gpgpu_const_cache_l1: "64:64:2,L:R:f:N,A:2:32,4".to_string(),
            gpgpu_cache_il1: "4:256:4,L:R:f:N,A:2:32,4".to_string(),
            gpgpu_cache_dl1: "none".to_string(),
            gpgpu_l1_cache_write_ratio: 0,
            gpgpu_l1_banks: 1,
            gpgpu_l1_banks_byte_interleaving: 32,
            gpgpu_l1_banks_hashing_function: 0,
            gpgpu_l1_latency: 1,
            gpgpu_smem_latency: 3,
            gpgpu_cache_dl1_pref_l1: "none".to_string(),
            gpgpu_cache_dl1_pref_shared: "none".to_string(),
            gpgpu_gmem_skip_l1d: false.into(),
            gpgpu_perfect_mem: false.into(),
            n_regfile_gating_group: 4,
            gpgpu_clock_gated_reg_file: false.into(),
            gpgpu_clock_gated_lanes: false.into(),
            gpgpu_shader_registers: 8192,
            gpgpu_registers_per_block: 8192,
            gpgpu_ignore_resources_limitation: false.into(),
            gpgpu_shader_cta: 32,
            gpgpu_num_cta_barriers: 16,
            gpgpu_n_clusters: 10,
            gpgpu_n_cores_per_cluster: 3,
            gpgpu_n_cluster_ejection_buffer_size: 8,
            gpgpu_n_ldst_response_buffer_size: 2,
            gpgpu_shmem_per_block: 49152,
            gpgpu_shmem_size: 16384,
            gpgpu_shmem_option: 0,
            gpgpu_unified_l1d_size: 0,
            gpgpu_adaptive_cache_config: false.into(),
            gpgpu_shmem_size_default: 16384,
            gpgpu_shmem_size_pref_l1: 16384,
            gpgpu_shmem_size_pref_shared: 16384,
            gpgpu_shmem_num_banks: 16,
            gpgpu_shmem_limited_broadcast: 1,
            gpgpu_shmem_warp_parts: 2,
            gpgpu_mem_unit_ports: 1,
            gpgpu_warpdistro_shader: -1,
            gpgpu_warp_issue_shader: 0,
            gpgpu_local_mem_map: true.into(),
            gpgpu_num_reg_banks: 8,
            gpgpu_reg_bank_use_warp_id: false.into(),
            gpgpu_sub_core_model: false.into(),
            gpgpu_enable_specialized_operand_collector: true.into(),
            gpgpu_operand_collector_num_units_sp: 4,
            gpgpu_operand_collector_num_units_dp: 0,
            gpgpu_operand_collector_num_units_sfu: 4,
            gpgpu_operand_collector_num_units_int: 0,
            gpgpu_operand_collector_num_units_tensor_core: 4,
            gpgpu_operand_collector_num_units_mem: 2,
            gpgpu_operand_collector_num_units_gen: 0,
            gpgpu_operand_collector_num_in_ports_sp: 1,
            gpgpu_operand_collector_num_in_ports_dp: 0,
            gpgpu_operand_collector_num_in_ports_sfu: 1,
            gpgpu_operand_collector_num_in_ports_int: 0,
            gpgpu_operand_collector_num_in_ports_tensor_core: 1,
            gpgpu_operand_collector_num_in_ports_mem: 1,
            gpgpu_operand_collector_num_in_ports_gen: 0,
            gpgpu_operand_collector_num_out_ports_sp: 1,
            gpgpu_operand_collector_num_out_ports_dp: 0,
            gpgpu_operand_collector_num_out_ports_sfu: 1,
            gpgpu_operand_collector_num_out_ports_int: 0,
            gpgpu_operand_collector_num_out_ports_tensor_core: 1,
            gpgpu_operand_collector_num_out_ports_mem: 1,
            gpgpu_operand_collector_num_out_ports_gen: 0,
            gpgpu_coalesce_arch: 13,
            gpgpu_num_sched_per_core: 1,
            gpgpu_max_insn_issue_per_warp: 2,
            gpgpu_dual_issue_diff_exec_units: true.into(),
            gpgpu_simt_core_sim_order: 1,
            gpgpu_pipeline_widths: "1,1,1,1,1,1,1,1,1,1,1,1,1".to_string(),
            gpgpu_tensor_core_avail: 0,
            gpgpu_num_sp_units: 1,
            gpgpu_num_dp_units: 0,
            gpgpu_num_int_units: 0,
            gpgpu_num_sfu_units: 1,
            gpgpu_num_tensor_core_units: 0,
            gpgpu_num_mem_units: 1,
            gpgpu_scheduler: "gto".to_string(),
            gpgpu_concurrent_kernel_sm: false.into(),
            gpgpu_perfect_inst_const_cache: false.into(),
            gpgpu_inst_fetch_throughput: 1,
            gpgpu_reg_file_port_throughput: 1,
            specialized_unit_1: "0,4,4,4,4,BRA".to_string(),
            specialized_unit_2: "0,4,4,4,4,BRA".to_string(),
            specialized_unit_3: "0,4,4,4,4,BRA".to_string(),
            specialized_unit_4: "0,4,4,4,4,BRA".to_string(),
            specialized_unit_5: "0,4,4,4,4,BRA".to_string(),
            specialized_unit_6: "0,4,4,4,4,BRA".to_string(),
            specialized_unit_7: "0,4,4,4,4,BRA".to_string(),
            specialized_unit_8: "0,4,4,4,4,BRA".to_string(),
        }
    }
}
