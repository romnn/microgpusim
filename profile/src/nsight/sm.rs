use super::metrics::Float;
use crate::Metric;

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SM {
    #[serde(rename = "sm__active_warps_avg_per_active_cycle")]
    pub active_warps_avg_per_active_cycle: Option<Metric<Float>>,
    #[serde(rename = "sm__active_warps_avg_per_active_cycle_pct")]
    pub active_warps_avg_per_active_cycle_pct: Option<Metric<Float>>,
    #[serde(rename = "sm__elapsed_cycles_avg")]
    pub elapsed_cycles_avg: Option<Metric<Float>>,
    #[serde(rename = "sm__elapsed_cycles_sum")]
    pub elapsed_cycles_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__active_cycles_avg")]
    pub active_cycles_avg: Option<Metric<Float>>,
    #[serde(rename = "sm__active_cycles_sum")]
    pub active_cycles_sum: Option<Metric<Float>>,

    // #[serde(rename = "sm__inst_executed_avg_per_active_cycle")]
    // pub inst_executed_avg_per_active_cycle: Option<Metric<Float>>,
    // #[serde(rename = "sm__inst_executed_avg_per_elapsed_cycle")]
    // pub inst_executed_avg_per_elapsed_cycle: Option<Metric<Float>>,
    // #[serde(rename = "sm__inst_executed_pipes_mem_per_active_cycle_sol_pct")]
    // pub inst_executed_pipes_mem_per_active_cycle_sol_pct: Option<Metric<Float>>,
    // #[serde(rename = "sm__inst_issued_avg_per_active_cycle")]
    // pub inst_issued_avg_per_active_cycle: Option<Metric<Float>>,
    // #[serde(rename = "sm__inst_issued_per_active_cycle_sol_pct")]
    // pub inst_issued_per_active_cycle_sol_pct: Option<Metric<Float>>,

    // instructions executed
    #[serde(rename = "sm__inst_executed_avg")]
    pub inst_executed_avg: Option<Metric<Float>>,
    #[serde(rename = "sm__inst_executed_avg_per_active_cycle")]
    pub inst_executed_avg_per_active_cycle: Option<Metric<Float>>,
    #[serde(rename = "sm__inst_executed_avg_per_elapsed_cycle")]
    pub inst_executed_avg_per_elapsed_cycle: Option<Metric<Float>>,
    #[serde(rename = "sm__inst_executed_max")]
    pub inst_executed_max: Option<Metric<Float>>,
    #[serde(rename = "sm__inst_executed_min")]
    pub inst_executed_min: Option<Metric<Float>>,
    #[serde(rename = "sm__inst_executed_per_active_cycle_sol_pct")]
    pub inst_executed_per_active_cycle_sol_pct: Option<Metric<Float>>,
    #[serde(rename = "sm__inst_executed_per_elapsed_cycle_sol_pct")]
    pub inst_executed_per_elapsed_cycle_sol_pct: Option<Metric<Float>>,
    #[serde(rename = "sm__inst_executed_pipes_mem_per_active_cycle_sol_pct")]
    pub inst_executed_pipes_mem_per_active_cycle_sol_pct: Option<Metric<Float>>,
    #[serde(rename = "sm__inst_executed_pipes_mem_per_elapsed_cycle_sol_pct")]
    pub inst_executed_pipes_mem_per_elapsed_cycle_sol_pct: Option<Metric<Float>>,
    #[serde(rename = "sm__inst_executed_sum")]
    pub inst_executed_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__inst_executed_sum_per_active_cycle")]
    pub inst_executed_sum_per_active_cycle: Option<Metric<Float>>,
    #[serde(rename = "sm__inst_executed_sum_per_elapsed_cycle")]
    pub inst_executed_sum_per_elapsed_cycle: Option<Metric<Float>>,
    // instructions issued
    /// The average number of instructions issued (may not retire) by all SMs.
    #[serde(rename = "sm__inst_issued_avg")]
    pub inst_issued_avg: Option<Metric<Float>>,
    /// The average number of instructions issued (may not retire) in all SMs per average active cycles.
    #[serde(rename = "sm__inst_issued_avg_per_active_cycle")]
    pub inst_issued_avg_per_active_cycle: Option<Metric<Float>>,
    /// The average number of instructions issued (may not retire) in all SMs per average elapsed cycles
    #[serde(rename = "sm__inst_issued_avg_per_elapsed_cycle")]
    pub inst_issued_avg_per_elapsed_cycle: Option<Metric<Float>>,
    /// The maximum number of instructions issued (may not retire) by any SM.
    #[serde(rename = "sm__inst_issued_max")]
    pub inst_issued_max: Option<Metric<Float>>,
    /// The minimum number of instructions issued (may not retire) by any SM.
    #[serde(rename = "sm__inst_issued_min")]
    pub inst_issued_min: Option<Metric<Float>>,
    /// The active SOL of instructions issued (may not retire) in the SM
    #[serde(rename = "sm__inst_issued_per_active_cycle_sol_pct")]
    pub inst_issued_per_active_cycle_sol_pct: Option<Metric<Float>>,
    /// The elapsed SOL of instructions issued (may not retire) in the SM.
    #[serde(rename = "sm__inst_issued_per_elapsed_cycle_sol_pct")]
    pub inst_issued_per_elapsed_cycle_sol_pct: Option<Metric<Float>>,
    /// The total number of instructions issued (may not retire) by all SMs.
    #[serde(rename = "sm__inst_issued_sum")]
    pub inst_issued_sum: Option<Metric<Float>>,
    /// The total number of instructions issued (may not retire) in all SMs per average active cycles.
    #[serde(rename = "sm__inst_issued_sum_per_active_cycle")]
    pub inst_issued_sum_per_active_cycle: Option<Metric<Float>>,
    /// The total number of instructions issued (may not retire) in all SMs per average elapsed cycles
    #[serde(rename = "sm__inst_issued_sum_per_elapsed_cycle")]
    pub inst_issued_sum_per_elapsed_cycle: Option<Metric<Float>>,
    #[serde(rename = "sm__maximum_warps_avg_per_active_cycle")]
    pub maximum_warps_avg_per_active_cycle: Option<Metric<Float>>,
    #[serde(rename = "sm__maximum_warps_per_active_cycle_pct")]
    pub maximum_warps_per_active_cycle_pct: Option<Metric<Float>>,
    #[serde(rename = "sm__shmem_ld_bank_conflict_sum")]
    pub shmem_ld_bank_conflict_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__shmem_ld_count_per_active_cycle_sol_pct")]
    pub shmem_ld_count_per_active_cycle_sol_pct: Option<Metric<Float>>,
    #[serde(rename = "sm__shmem_ld_count_sum")]
    pub shmem_ld_count_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__shmem_st_bank_conflict_sum")]
    pub shmem_st_bank_conflict_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__shmem_st_count_per_active_cycle_sol_pct")]
    pub shmem_st_count_per_active_cycle_sol_pct: Option<Metric<Float>>,
    #[serde(rename = "sm__shmem_st_count_sum")]
    pub shmem_st_count_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__sol_pct")]
    pub sol_pct: Option<Metric<Float>>,
    #[serde(rename = "sm__warps_active.avg.pct_of_peak_sustained_active")]
    pub warps_active_avg_pct_of_peak_sustained_active: Option<Metric<Float>>,
    #[serde(rename = "sm__pipe_alu_cycles_active.sum")]
    pub pipe_alu_cycles_active_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__pipe_fma_cycles_active.sum")]
    pub pipe_fma_cycles_active_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__pipe_fp64_cycles_active.sum")]
    pub pipe_fp64_cycles_active_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__pipe_shared_cycles_active.sum")]
    pub pipe_shared_cycles_active_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__pipe_tensor_cycles_active.sum")]
    pub pipe_tensor_cycles_active_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__pipe_tensor_op_hmma_cycles_active.sum")]
    pub pipe_tensor_op_hmma_cycles_active_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__cycles_elapsed.sum")]
    pub cycles_elapsed_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__cycles_active.sum")]
    pub cycles_active_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__cycles_active.avg")]
    pub cycles_active_avg: Option<Metric<Float>>,
    #[serde(rename = "sm__cycles_elapsed.avg")]
    pub cycles_elapsed_avg: Option<Metric<Float>>,
    #[serde(rename = "sm__sass_thread_inst_executed_op_integer_pred_on.sum")]
    pub sass_thread_inst_executed_op_integer_pred_on_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__sass_thread_inst_executed_ops_dadd_dmul_dfma_pred_on.sum")]
    pub sass_thread_inst_executed_ops_dadd_dmul_dfma_pred_on_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__sass_thread_inst_executed_ops_fadd_fmul_ffma_pred_on.sum")]
    pub sass_thread_inst_executed_ops_fadd_fmul_ffma_pred_on_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__sass_thread_inst_executed_ops_hadd_hmul_hfma_pred_on.sum")]
    pub sass_thread_inst_executed_ops_hadd_hmul_hfma_pred_on_sum: Option<Metric<Float>>,

    #[serde(rename = "sm__inst_executed_pipe_alu.sum")]
    pub inst_executed_pipe_alu_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__inst_executed_pipe_fma.sum")]
    pub inst_executed_pipe_fma_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__inst_executed_pipe_fp16.sum")]
    pub inst_executed_pipe_fp16_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__inst_executed_pipe_fp64.sum")]
    pub inst_executed_pipe_fp64_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__inst_executed_pipe_tensor.sum")]
    pub inst_executed_pipe_tensor_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__inst_executed_pipe_tex.sum")]
    pub inst_executed_pipe_tex_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__inst_executed_pipe_xu.sum")]
    pub inst_executed_pipe_xu_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__inst_executed_pipe_lsu.sum")]
    pub inst_executed_pipe_lsu_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__sass_thread_inst_executed_op_fp16_pred_on.sum")]
    pub sass_thread_inst_executed_op_fp16_pred_on_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__sass_thread_inst_executed_op_fp32_pred_on.sum")]
    pub sass_thread_inst_executed_op_fp32_pred_on_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__sass_thread_inst_executed_op_fp64_pred_on.sum")]
    pub sass_thread_inst_executed_op_fp64_pred_on_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__sass_thread_inst_executed_op_dmul_pred_on.sum")]
    pub sass_thread_inst_executed_op_dmul_pred_on_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__sass_thread_inst_executed_op_dfma_pred_on.sum")]
    pub sass_thread_inst_executed_op_dfma_pred_on_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__sass_thread_inst_executed.sum")]
    pub sass_thread_inst_executed_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__sass_inst_executed_op_shared_st.sum")]
    pub sass_inst_executed_op_shared_st_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__sass_inst_executed_op_shared_ld.sum")]
    pub sass_inst_executed_op_shared_ld_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__sass_inst_executed_op_memory_128b.sum")]
    pub sass_inst_executed_op_memory_128b_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__sass_inst_executed_op_memory_64b.sum")]
    pub sass_inst_executed_op_memory_64b_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__sass_inst_executed_op_memory_32b.sum")]
    pub sass_inst_executed_op_memory_32b_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__sass_inst_executed_op_memory_16b.sum")]
    pub sass_inst_executed_op_memory_16b_sum: Option<Metric<Float>>,
    #[serde(rename = "sm__sass_inst_executed_op_memory_8b.sum")]
    pub sass_inst_executed_op_memory_8b_sum: Option<Metric<Float>>,
}
