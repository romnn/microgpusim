use crate::Metric;

#[derive(PartialEq, Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Metrics {
    #[serde(rename = "Device")]
    pub device: Metric<String>,
    #[serde(rename = "Context")]
    pub context: Metric<usize>,
    #[serde(rename = "Stream")]
    pub stream: Metric<usize>,
    #[serde(rename = "Kernel")]
    pub kernel: Metric<String>,
    #[serde(rename = "Correlation_ID")]
    pub correlation_id: Metric<usize>,

    pub elapsed_cycles_sm: Metric<usize>,
    pub inst_per_warp: Metric<f32>,
    pub ipc: Metric<f32>,
    pub issued_ipc: Metric<f32>,
    pub issue_slots: Metric<usize>,
    pub issue_slot_utilization: Metric<f32>,
    pub eligible_warps_per_cycle: Metric<f32>,
    pub unique_warps_launched: Metric<usize>,

    pub pcie_total_data_transmitted: Metric<usize>,
    pub pcie_total_data_received: Metric<usize>,

    pub inst_replay_overhead: Metric<f32>,
    pub local_memory_overhead: Metric<f32>,

    // Cache hit rates
    pub tex_cache_hit_rate: Metric<f32>,
    pub l2_tex_read_hit_rate: Metric<f32>,
    pub l2_tex_write_hit_rate: Metric<f32>,
    pub l2_tex_hit_rate: Metric<f32>,
    pub global_hit_rate: Metric<f32>,
    pub local_hit_rate: Metric<f32>,

    pub inst_issued: Metric<usize>,
    pub inst_executed: Metric<usize>,

    // GPU stalls
    pub stall_inst_fetch: Metric<f32>,
    pub stall_exec_dependency: Metric<f32>,
    pub stall_memory_dependency: Metric<f32>,
    pub stall_texture: Metric<f32>,
    pub stall_sync: Metric<f32>,
    pub stall_other: Metric<f32>,
    pub stall_constant_memory_dependency: Metric<f32>,
    pub stall_pipe_busy: Metric<f32>,
    pub stall_memory_throttle: Metric<f32>,
    pub stall_not_selected: Metric<f32>,

    // Instruction count per kind
    pub inst_fp_16: Metric<usize>,
    pub inst_fp_32: Metric<usize>,
    pub inst_fp_64: Metric<usize>,
    pub inst_integer: Metric<usize>,
    pub inst_bit_convert: Metric<String>,
    pub inst_control: Metric<String>,
    pub inst_compute_ld_st: Metric<String>,
    pub inst_misc: Metric<String>,
    pub inst_inter_thread_communication: Metric<String>,

    // Number of instructions issued and executed
    pub ldst_issued: Metric<usize>,
    pub ldst_executed: Metric<usize>,
    pub cf_issued: Metric<String>,
    pub cf_executed: Metric<String>,

    // Transactions
    pub atomic_transactions: Metric<usize>,
    pub atomic_transactions_per_request: Metric<f32>,
    pub l2_atomic_transactions: Metric<usize>,
    pub l2_tex_read_transactions: Metric<usize>,
    pub l2_tex_write_transactions: Metric<usize>,
    pub ecc_transactions: Metric<usize>,
    pub dram_read_transactions: Metric<usize>,
    pub dram_write_transactions: Metric<usize>,
    pub shared_store_transactions: Metric<usize>,
    pub shared_store_transactions_per_request: Metric<f32>,
    pub shared_load_transactions: Metric<usize>,
    pub shared_load_transactions_per_request: Metric<f32>,
    pub local_load_transactions: Metric<usize>,
    pub local_load_transactions_per_request: Metric<f32>,
    pub local_store_transactions: Metric<usize>,
    pub local_store_transactions_per_request: Metric<f32>,
    pub gld_transactions: Metric<usize>,
    pub gld_transactions_per_request: Metric<f32>,
    pub gst_transactions: Metric<usize>,
    pub gst_transactions_per_request: Metric<f32>,
    pub sysmem_read_transactions: Metric<usize>,
    pub sysmem_write_transactions: Metric<usize>,
    pub l2_read_transactions: Metric<usize>,
    pub l2_write_transactions: Metric<usize>,
    pub tex_cache_transactions: Metric<usize>,

    // FLOP count
    pub flop_count_hp: Metric<usize>,
    pub flop_count_hp_add: Metric<usize>,
    pub flop_count_hp_mul: Metric<usize>,
    pub flop_count_hp_fma: Metric<usize>,
    pub flop_count_dp: Metric<usize>,
    pub flop_count_dp_add: Metric<usize>,
    pub flop_count_dp_fma: Metric<usize>,
    pub flop_count_dp_mul: Metric<usize>,
    pub flop_count_sp: Metric<usize>,
    pub flop_count_sp_add: Metric<usize>,
    pub flop_count_sp_fma: Metric<usize>,
    pub flop_count_sp_mul: Metric<usize>,
    pub flop_count_sp_special: Metric<usize>,

    // Num executed instructions
    pub inst_executed_global_loads: Metric<usize>,
    pub inst_executed_local_loads: Metric<usize>,
    pub inst_executed_shared_loads: Metric<usize>,
    pub inst_executed_surface_loads: Metric<usize>,
    pub inst_executed_global_stores: Metric<usize>,
    pub inst_executed_local_stores: Metric<usize>,
    pub inst_executed_shared_stores: Metric<usize>,
    pub inst_executed_surface_stores: Metric<usize>,
    pub inst_executed_global_atomics: Metric<usize>,
    pub inst_executed_global_reductions: Metric<usize>,
    pub inst_executed_surface_atomics: Metric<usize>,
    pub inst_executed_surface_reductions: Metric<String>,
    pub inst_executed_shared_atomics: Metric<String>,
    pub inst_executed_tex_ops: Metric<String>,

    // L2 loaded and stored number of bytes
    pub l2_global_load_bytes: Metric<usize>,
    pub l2_local_load_bytes: Metric<usize>,
    pub l2_surface_load_bytes: Metric<usize>,
    pub l2_local_global_store_bytes: Metric<usize>,
    pub l2_global_reduction_bytes: Metric<usize>,
    pub l2_global_atomic_store_bytes: Metric<usize>,
    pub l2_surface_store_bytes: Metric<usize>,
    pub l2_surface_reduction_bytes: Metric<usize>,
    pub l2_surface_atomic_store_bytes: Metric<usize>,

    // Number of bytes read and written
    pub sysmem_read_bytes: Metric<usize>,
    pub sysmem_write_bytes: Metric<usize>,
    pub dram_write_bytes: Metric<usize>,
    pub dram_read_bytes: Metric<usize>,

    // Number of requests per memory space
    pub global_load_requests: Metric<usize>,
    pub local_load_requests: Metric<usize>,
    pub surface_load_requests: Metric<usize>,
    pub global_store_requests: Metric<usize>,
    pub local_store_requests: Metric<usize>,
    pub surface_store_requests: Metric<usize>,
    pub global_atomic_requests: Metric<usize>,
    pub global_reduction_requests: Metric<usize>,
    pub surface_atomic_requests: Metric<usize>,
    pub surface_reduction_requests: Metric<usize>,
    pub texture_load_requests: Metric<usize>,

    // Utilization of functional units
    pub tex_utilization: Metric<String>,
    pub l2_utilization: Metric<String>,
    pub shared_utilization: Metric<String>,
    pub ldst_fu_utilization: Metric<String>,
    pub cf_fu_utilization: Metric<String>,
    pub special_fu_utilization: Metric<String>,
    pub tex_fu_utilization: Metric<String>,
    pub single_precision_fu_utilization: Metric<String>,
    pub double_precision_fu_utilization: Metric<String>,
    pub dram_utilization: Metric<String>,
    pub half_precision_fu_utilization: Metric<String>,
    pub sysmem_utilization: Metric<String>,
    pub sysmem_read_utilization: Metric<String>,
    pub sysmem_write_utilization: Metric<String>,

    // Efficiency
    pub sm_efficiency: Metric<f32>,
    pub shared_efficiency: Metric<f32>,
    pub flop_hp_efficiency: Metric<f32>,
    pub flop_sp_efficiency: Metric<f32>,
    pub flop_dp_efficiency: Metric<f32>,
    pub gld_efficiency: Metric<f32>,
    pub gst_efficiency: Metric<f32>,
    pub branch_efficiency: Metric<f32>,
    pub warp_execution_efficiency: Metric<f32>,
    pub warp_nonpred_execution_efficiency: Metric<f32>,

    // Throughput
    pub dram_read_throughput: Metric<f32>,
    pub dram_write_throughput: Metric<f32>,
    pub l2_atomic_throughput: Metric<f32>,
    pub ecc_throughput: Metric<f32>,
    pub tex_cache_throughput: Metric<f32>,
    pub l2_tex_read_throughput: Metric<f32>,
    pub l2_tex_write_throughput: Metric<f32>,
    pub l2_read_throughput: Metric<f32>,
    pub l2_write_throughput: Metric<f32>,
    pub sysmem_read_throughput: Metric<f32>,
    pub sysmem_write_throughput: Metric<f32>,
    pub local_load_throughput: Metric<f32>,
    pub local_store_throughput: Metric<f32>,
    pub shared_load_throughput: Metric<f32>,
    pub shared_store_throughput: Metric<f32>,
    pub gld_throughput: Metric<f32>,
    pub gst_throughput: Metric<f32>,
    pub gld_requested_throughput: Metric<f32>,
    pub gst_requested_throughput: Metric<f32>,
}

#[derive(PartialEq, Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Command {
    #[serde(rename = "Start")]
    pub start: Metric<f32>,
    #[serde(rename = "Duration")]
    pub duration: Metric<f32>,
    #[serde(rename = "Grid X", default)]
    pub grid_x: Metric<usize>,
    #[serde(rename = "Grid Y", default)]
    pub grid_y: Metric<usize>,
    #[serde(rename = "Grid Z", default)]
    pub grid_z: Metric<usize>,
    #[serde(rename = "Block X", default)]
    pub block_x: Metric<usize>,
    #[serde(rename = "Block Y", default)]
    pub block_y: Metric<usize>,
    #[serde(rename = "Block Z", default)]
    pub block_z: Metric<usize>,
    #[serde(rename = "Registers Per Thread")]
    pub registers_per_thread: Metric<usize>,
    #[serde(rename = "Static SMem")]
    pub static_shared_memory: Metric<usize>,
    #[serde(rename = "Dynamic SMem")]
    pub dynamic_shared_memory: Metric<usize>,
    #[serde(rename = "Size")]
    pub size: Metric<usize>,
    #[serde(rename = "Throughput")]
    pub throughput: Metric<f32>,
    #[serde(rename = "SrcMemType")]
    pub src_mem_type: Metric<String>,
    #[serde(rename = "DstMemType")]
    pub dest_mem_type: Metric<String>,
    #[serde(rename = "Device")]
    pub device: Metric<String>,
    #[serde(rename = "Context")]
    pub context: Metric<usize>,
    #[serde(rename = "Stream")]
    pub stream: Metric<usize>,
    #[serde(rename = "Name")]
    pub name: Metric<String>,
    #[serde(rename = "Correlation_ID")]
    pub correlation_id: Metric<usize>,
}
