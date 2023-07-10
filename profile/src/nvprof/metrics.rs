use crate::Metric;

#[derive(PartialEq, Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct NvprofMetrics {
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
    // pub elapsed_cycles_sm: Metric<String>,
    // pub inst_per_warp: Metric<String>,
    // pub branch_efficiency: Metric<String>,
    // pub warp_execution_efficiency: Metric<String>,
    // pub warp_nonpred_execution_efficiency: Metric<String>,
    // pub inst_replay_overhead: Metric<String>,
    // TEMP
    pub shared_load_transactions_per_request: Metric<f32>,
    pub shared_store_transactions_per_request: Metric<f32>,
    pub local_load_transactions_per_request: Metric<f32>,
    pub local_store_transactions_per_request: Metric<f32>,
    pub gld_transactions_per_request: Metric<f32>,
    pub gst_transactions_per_request: Metric<f32>,
    pub shared_store_transactions: Metric<usize>,
    pub shared_load_transactions: Metric<usize>,
    pub local_load_transactions: Metric<usize>,
    pub local_store_transactions: Metric<usize>,
    pub gld_transactions: Metric<usize>,
    pub gst_transactions: Metric<usize>,
    pub sysmem_read_transactions: Metric<usize>,
    pub sysmem_write_transactions: Metric<usize>,
    pub l2_read_transactions: Metric<usize>,
    pub l2_write_transactions: Metric<usize>,
    // TEMP
    // pub global_hit_rate: Metric<String>,
    // pub local_hit_rate: Metric<String>,
    // pub gld_requested_throughput: Metric<String>,
    // pub gst_requested_throughput: Metric<String>,
    // pub gld_throughput: Metric<String>,
    // pub gst_throughput: Metric<String>,
    // pub local_memory_overhead: Metric<String>,
    // pub tex_cache_hit_rate: Metric<String>,
    // pub l2_tex_read_hit_rate: Metric<String>,
    // pub l2_tex_write_hit_rate: Metric<String>,
    // pub tex_cache_throughput: Metric<String>,
    // pub l2_tex_read_throughput: Metric<String>,
    // pub l2_tex_write_throughput: Metric<String>,
    // pub l2_read_throughput: Metric<String>,
    // pub l2_write_throughput: Metric<String>,
    // pub sysmem_read_throughput: Metric<String>,
    // pub sysmem_write_throughput: Metric<String>,
    // pub local_load_throughput: Metric<String>,
    // pub local_store_throughput: Metric<String>,
    // pub shared_load_throughput: Metric<String>,
    // pub shared_store_throughput: Metric<String>,
    // pub gld_efficiency: Metric<String>,
    // pub gst_efficiency: Metric<String>,
    // pub tex_cache_transactions: Metric<String>,
    // pub flop_count_dp: Metric<String>,
    // pub flop_count_dp_add: Metric<String>,
    // pub flop_count_dp_fma: Metric<String>,
    // pub flop_count_dp_mul: Metric<String>,
    // pub flop_count_sp: Metric<String>,
    // pub flop_count_sp_add: Metric<String>,
    // pub flop_count_sp_fma: Metric<String>,
    // pub flop_count_sp_mul: Metric<String>,
    // pub flop_count_sp_special: Metric<String>,
    // pub inst_executed: Metric<String>,
    // pub inst_issued: Metric<String>,
    // pub sysmem_utilization: Metric<String>,
    // pub stall_inst_fetch: Metric<String>,
    // pub stall_exec_dependency: Metric<String>,
    // pub stall_memory_dependency: Metric<String>,
    // pub stall_texture: Metric<String>,
    // pub stall_sync: Metric<String>,
    // pub stall_other: Metric<String>,
    // pub stall_constant_memory_dependency: Metric<String>,
    // pub stall_pipe_busy: Metric<String>,
    // pub shared_efficiency: Metric<String>,
    // pub inst_fp_32: Metric<String>,
    // pub inst_fp_64: Metric<String>,
    // pub inst_integer: Metric<String>,
    // pub inst_bit_convert: Metric<String>,
    // pub inst_control: Metric<String>,
    // pub inst_compute_ld_st: Metric<String>,
    // pub inst_misc: Metric<String>,
    // pub inst_inter_thread_communication: Metric<String>,
    // pub issue_slots: Metric<String>,
    // pub cf_issued: Metric<String>,
    // pub cf_executed: Metric<String>,
    // pub ldst_issued: Metric<String>,
    // pub ldst_executed: Metric<String>,
    // TEMP
    pub atomic_transactions: Metric<usize>,
    pub atomic_transactions_per_request: Metric<f32>,
    // TEMP
    // pub l2_atomic_throughput: Metric<String>,
    // pub l2_atomic_transactions: Metric<String>,
    // pub l2_tex_read_transactions: Metric<String>,
    // pub stall_memory_throttle: Metric<String>,
    // pub stall_not_selected: Metric<String>,
    // pub l2_tex_write_transactions: Metric<String>,
    // pub flop_count_hp: Metric<String>,
    // pub flop_count_hp_add: Metric<String>,
    // pub flop_count_hp_mul: Metric<String>,
    // pub flop_count_hp_fma: Metric<String>,
    // pub inst_fp_16: Metric<String>,
    // pub sysmem_read_utilization: Option<String>,
    // pub sysmem_write_utilization: Option<String>,
    // pub pcie_total_data_transmitted: Option<String>,
    // pub pcie_total_data_received: Option<String>,
    // pub inst_executed_global_loads: Option<String>,
    // pub inst_executed_local_loads: Option<String>,
    // pub inst_executed_shared_loads: Option<String>,
    // pub inst_executed_surface_loads: Option<String>,
    // pub inst_executed_global_stores: Option<String>,
    // pub inst_executed_local_stores: Option<String>,
    // pub inst_executed_shared_stores: Option<String>,
    // pub inst_executed_surface_stores: Option<String>,
    // pub inst_executed_global_atomics: Option<String>,
    // pub inst_executed_global_reductions: Option<String>,
    // pub inst_executed_surface_atomics: Option<String>,
    // pub inst_executed_surface_reductions: Option<String>,
    // pub inst_executed_shared_atomics: Option<String>,
    // pub inst_executed_tex_ops: Option<String>,
    // TEMP
    pub l2_global_load_bytes: Metric<usize>,
    pub l2_local_load_bytes: Metric<usize>,
    pub l2_surface_load_bytes: Metric<usize>,
    pub l2_local_global_store_bytes: Metric<usize>,
    pub l2_global_reduction_bytes: Metric<usize>,
    pub l2_global_atomic_store_bytes: Metric<usize>,
    pub l2_surface_store_bytes: Metric<usize>,
    pub l2_surface_reduction_bytes: Metric<usize>,
    pub l2_surface_atomic_store_bytes: Metric<usize>,
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
    // TEMP
    // pub sysmem_read_bytes: Option<String>,
    // pub sysmem_write_bytes: Option<String>,
    // pub l2_tex_hit_rate: Option<String>,
    // pub texture_load_requests: Option<String>,
    // pub unique_warps_launched: Option<String>,
    // pub sm_efficiency: Option<String>,
    // pub achieved_occupancy: Option<String>,
    // pub ipc: Option<String>,
    // pub issued_ipc: Option<String>,
    // pub issue_slot_utilization: Option<String>,
    // pub eligible_warps_per_cycle: Option<String>,
    // pub tex_utilization: Option<String>,
    // pub l2_utilization: Option<String>,
    // pub shared_utilization: Option<String>,
    // pub ldst_fu_utilization: Option<String>,
    // pub cf_fu_utilization: Option<String>,
    // pub special_fu_utilization: Option<String>,
    // pub tex_fu_utilization: Option<String>,
    // pub single_precision_fu_utilization: Option<String>,
    // pub double_precision_fu_utilization: Option<String>,
    // pub flop_hp_efficiency: Option<String>,
    // pub flop_sp_efficiency: Option<String>,
    // pub flop_dp_efficiency: Option<String>,
    // TEMP
    pub dram_read_transactions: Metric<usize>,
    pub dram_write_transactions: Metric<usize>,
    pub dram_read_throughput: Metric<f32>,
    pub dram_write_throughput: Metric<f32>,
    // TEMP
    // pub dram_utilization: Option<String>,
    // pub half_precision_fu_utilization: Option<String>,
    // TEMP
    pub dram_write_bytes: Metric<usize>,
    // TEMP
    // pub ecc_transactions: Option<String>,
    // pub ecc_throughput: Option<String>,
    // TEMP
    pub dram_read_bytes: Metric<usize>,
    // TEMP
}