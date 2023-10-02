use super::metrics::Float;
use crate::Metric;

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SMScheduler {
    #[serde(rename = "smsp__active_warps_avg_per_active_cycle")]
    pub active_warps_avg_per_active_cycle: Option<Metric<Float>>,
    #[serde(rename = "smsp__eligible_warps_avg_per_active_cycle")]
    pub eligible_warps_avg_per_active_cycle: Option<Metric<Float>>,
    // begin copy
    #[serde(rename = "smsp__inst_executed_tex_ops")]
    pub smsp_inst_executed_tex_ops: Option<Metric<Float>>,
    #[serde(rename = "smsp__inst_issued0_active_per_active_cycle_pct")]
    pub smsp_inst_issued0_active_per_active_cycle_pct: Option<Metric<Float>>,
    #[serde(rename = "smsp__inst_issued_avg")]
    pub smsp_inst_issued_avg: Option<Metric<Float>>,
    #[serde(rename = "smsp__inst_issued_per_issue_active")]
    pub smsp_inst_issued_per_issue_active: Option<Metric<Float>>,
    #[serde(rename = "smsp__inst_issued_sum")]
    pub smsp_inst_issued_sum: Option<Metric<Float>>,
    #[serde(rename = "smsp__issue_active_avg_per_active_cycle")]
    pub smsp_issue_active_avg_per_active_cycle: Option<Metric<Float>>,
    #[serde(rename = "smsp__issue_active_per_active_cycle_pct")]
    pub smsp_issue_active_per_active_cycle_pct: Option<Metric<Float>>,
    #[serde(rename = "smsp__maximum_warps_avg_per_active_cycle")]
    pub maximum_warps_avg_per_active_cycle: Option<Metric<Float>>,
    #[serde(rename = "smsp__pcsamp_aggregated_passes")]
    pub smsp_pcsamp_aggregated_passes: Option<Metric<Float>>,
    #[serde(rename = "smsp__pcsamp_buffer_size_bytes")]
    pub smsp_pcsamp_buffer_size_bytes: Option<Metric<Float>>,
    #[serde(rename = "smsp__pcsamp_dropped_bytes")]
    pub smsp_pcsamp_dropped_bytes: Option<Metric<Float>>,
    #[serde(rename = "smsp__pcsamp_interval")]
    pub smsp_pcsamp_interval: Option<Metric<Float>>,
    #[serde(rename = "smsp__pcsamp_interval_cycles")]
    pub smsp_pcsamp_interval_cycles: Option<Metric<Float>>,
    #[serde(rename = "smsp__pcsamp_warp_stall_allocation_stall")]
    pub smsp_pcsamp_warp_stall_allocation_stall: Option<Metric<Float>>,
    #[serde(rename = "smsp__pcsamp_warp_stall_barrier")]
    pub smsp_pcsamp_warp_stall_barrier: Option<Metric<Float>>,
    #[serde(rename = "smsp__pcsamp_warp_stall_dispatch_stall")]
    pub smsp_pcsamp_warp_stall_dispatch_stall: Option<Metric<Float>>,
    #[serde(rename = "smsp__pcsamp_warp_stall_drain")]
    pub smsp_pcsamp_warp_stall_drain: Option<Metric<Float>>,
    #[serde(rename = "smsp__pcsamp_warp_stall_imc_miss")]
    pub smsp_pcsamp_warp_stall_imc_miss: Option<Metric<Float>>,
    #[serde(rename = "smsp__pcsamp_warp_stall_long_scoreboard")]
    pub smsp_pcsamp_warp_stall_long_scoreboard: Option<Metric<Float>>,
    #[serde(rename = "smsp__pcsamp_warp_stall_math_pipe_throttle")]
    pub smsp_pcsamp_warp_stall_math_pipe_throttle: Option<Metric<Float>>,
    #[serde(rename = "smsp__pcsamp_warp_stall_membar")]
    pub smsp_pcsamp_warp_stall_membar: Option<Metric<Float>>,
    #[serde(rename = "smsp__pcsamp_warp_stall_mio_throttle")]
    pub smsp_pcsamp_warp_stall_mio_throttle: Option<Metric<Float>>,
    #[serde(rename = "smsp__pcsamp_warp_stall_misc")]
    pub smsp_pcsamp_warp_stall_misc: Option<Metric<Float>>,
    #[serde(rename = "smsp__pcsamp_warp_stall_no_instructions")]
    pub smsp_pcsamp_warp_stall_no_instructions: Option<Metric<Float>>,
    #[serde(rename = "smsp__pcsamp_warp_stall_not_selected")]
    pub smsp_pcsamp_warp_stall_not_selected: Option<Metric<Float>>,
    #[serde(rename = "smsp__pcsamp_warp_stall_selected")]
    pub smsp_pcsamp_warp_stall_selected: Option<Metric<Float>>,
    #[serde(rename = "smsp__pcsamp_warp_stall_short_scoreboard")]
    pub smsp_pcsamp_warp_stall_short_scoreboard: Option<Metric<Float>>,
    #[serde(rename = "smsp__pcsamp_warp_stall_tex_throttle")]
    pub smsp_pcsamp_warp_stall_tex_throttle: Option<Metric<Float>>,
    #[serde(rename = "smsp__pcsamp_warp_stall_tile_allocation_stall")]
    pub smsp_pcsamp_warp_stall_tile_allocation_stall: Option<Metric<Float>>,
    #[serde(rename = "smsp__pcsamp_warp_stall_wait")]
    pub smsp_pcsamp_warp_stall_wait: Option<Metric<Float>>,
    #[serde(rename = "smsp__thread_inst_executed_not_pred_off_per_inst_executed")]
    pub smsp_thread_inst_executed_not_pred_off_per_inst_executed: Option<Metric<Float>>,
    #[serde(rename = "smsp__thread_inst_executed_per_inst_executed")]
    pub smsp_thread_inst_executed_per_inst_executed: Option<Metric<Float>>,
    #[serde(rename = "smsp__warp_cycles_per_inst_executed")]
    pub smsp_warp_cycles_per_inst_executed: Option<Metric<Float>>,
    #[serde(rename = "smsp__warp_cycles_per_inst_issued")]
    pub smsp_warp_cycles_per_inst_issued: Option<Metric<Float>>,
    #[serde(rename = "smsp__warp_cycles_per_issue_active")]
    pub smsp_warp_cycles_per_issue_active: Option<Metric<Float>>,
    #[serde(rename = "smsp__warp_cycles_per_issue_stall_allocation_stall")]
    pub smsp_warp_cycles_per_issue_stall_allocation_stall: Option<Metric<Float>>,
    #[serde(rename = "smsp__warp_cycles_per_issue_stall_barrier")]
    pub smsp_warp_cycles_per_issue_stall_barrier: Option<Metric<Float>>,
    #[serde(rename = "smsp__warp_cycles_per_issue_stall_dispatch_stall")]
    pub smsp_warp_cycles_per_issue_stall_dispatch_stall: Option<Metric<Float>>,
    #[serde(rename = "smsp__warp_cycles_per_issue_stall_drain")]
    pub smsp_warp_cycles_per_issue_stall_drain: Option<Metric<Float>>,
    #[serde(rename = "smsp__warp_cycles_per_issue_stall_imc_miss")]
    pub smsp_warp_cycles_per_issue_stall_imc_miss: Option<Metric<Float>>,
    #[serde(rename = "smsp__warp_cycles_per_issue_stall_long_scoreboard")]
    pub smsp_warp_cycles_per_issue_stall_long_scoreboard: Option<Metric<Float>>,
    #[serde(rename = "smsp__warp_cycles_per_issue_stall_math_pipe_throttle")]
    pub smsp_warp_cycles_per_issue_stall_math_pipe_throttle: Option<Metric<Float>>,
    #[serde(rename = "smsp__warp_cycles_per_issue_stall_membar")]
    pub smsp_warp_cycles_per_issue_stall_membar: Option<Metric<Float>>,
    #[serde(rename = "smsp__warp_cycles_per_issue_stall_mio_throttle")]
    pub smsp_warp_cycles_per_issue_stall_mio_throttle: Option<Metric<Float>>,
    #[serde(rename = "smsp__warp_cycles_per_issue_stall_misc")]
    pub smsp_warp_cycles_per_issue_stall_misc: Option<Metric<Float>>,
    #[serde(rename = "smsp__warp_cycles_per_issue_stall_no_instructions")]
    pub smsp_warp_cycles_per_issue_stall_no_instructions: Option<Metric<Float>>,
    #[serde(rename = "smsp__warp_cycles_per_issue_stall_not_selected")]
    pub smsp_warp_cycles_per_issue_stall_not_selected: Option<Metric<Float>>,
    #[serde(rename = "smsp__warp_cycles_per_issue_stall_selected")]
    pub smsp_warp_cycles_per_issue_stall_selected: Option<Metric<Float>>,
    #[serde(rename = "smsp__warp_cycles_per_issue_stall_short_scoreboard")]
    pub smsp_warp_cycles_per_issue_stall_short_scoreboard: Option<Metric<Float>>,
    #[serde(rename = "smsp__warp_cycles_per_issue_stall_tex_throttle")]
    pub smsp_warp_cycles_per_issue_stall_tex_throttle: Option<Metric<Float>>,
    #[serde(rename = "smsp__warp_cycles_per_issue_stall_tile_allocation_stall")]
    pub smsp_warp_cycles_per_issue_stall_tile_allocation_stall: Option<Metric<Float>>,
    #[serde(rename = "smsp__warp_cycles_per_issue_stall_wait")]
    pub smsp_warp_cycles_per_issue_stall_wait: Option<Metric<Float>>,
    #[serde(rename = "smsp__warps_per_cycle_max")]
    pub smsp_warps_per_cycle_max: Option<Metric<Float>>,
    #[serde(rename = "smsp__thread_inst_executed_per_inst_executed.ratio")]
    pub smsp_thread_inst_executed_per_inst_executed_ratio: Option<Metric<Float>>,
    #[serde(rename = "smsp__cycles_active.avg.pct_of_peak_sustained_elapsed")]
    pub smsp_cycles_active_avg_pct_of_peak_sustained_elapsed: Option<Metric<Float>>,
    /// The average number of instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_avg")]
    pub inst_executed_avg: Option<Metric<Float>>,
    /// The average number of instructions executed per active cycle by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_avg_per_active_cycle")]
    pub inst_executed_avg_per_active_cycle: Option<Metric<Float>>,
    /// The average number of instructions executed per elapsed cycle by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_avg_per_elapsed_cycle")]
    pub inst_executed_avg_per_elapsed_cycle: Option<Metric<Float>>,
    /// The average number of cs instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_cs_avg")]
    pub inst_executed_cs_avg: Option<Metric<Float>>,
    /// The maximum number of cs instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_cs_max")]
    pub inst_executed_cs_max: Option<Metric<Float>>,
    /// The minimum number of cs instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_cs_min")]
    pub inst_executed_cs_min: Option<Metric<Float>>,
    /// The percentage of instructions executed on a SM scheduler that were cs instructions.
    #[serde(rename = "smsp__inst_executed_cs_pct")]
    pub inst_executed_cs_pct: Option<Metric<Float>>,
    /// The total number of cs instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_cs_sum")]
    pub inst_executed_cs_sum: Option<Metric<Float>>,
    /// The average number of fs instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_fs_avg")]
    pub inst_executed_fs_avg: Option<Metric<Float>>,
    /// The maximum number of fs instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_fs_max")]
    pub inst_executed_fs_max: Option<Metric<Float>>,
    /// The minimum number of fs instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_fs_min")]
    pub inst_executed_fs_min: Option<Metric<Float>>,
    /// The percentage of instructions executed on a SM scheduler that were fs instructions.
    #[serde(rename = "smsp__inst_executed_fs_pct")]
    pub inst_executed_fs_pct: Option<Metric<Float>>,
    /// The total number of fs instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_fs_sum")]
    pub inst_executed_fs_sum: Option<Metric<Float>>,
    /// The average number of LD instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_generic_loads_avg")]
    pub inst_executed_generic_loads_avg: Option<Metric<Float>>,
    /// The maximum number of LD instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_generic_loads_max")]
    pub inst_executed_generic_loads_max: Option<Metric<Float>>,
    /// The minimum number of LD instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_generic_loads_min")]
    pub inst_executed_generic_loads_min: Option<Metric<Float>>,
    /// The total number of LD instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_generic_loads_sum")]
    pub inst_executed_generic_loads_sum: Option<Metric<Float>>,
    /// The average number of ST instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_generic_stores_avg")]
    pub inst_executed_generic_stores_avg: Option<Metric<Float>>,
    /// The maximum number of ST instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_generic_stores_max")]
    pub inst_executed_generic_stores_max: Option<Metric<Float>>,
    /// The minimum number of ST instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_generic_stores_min")]
    pub inst_executed_generic_stores_min: Option<Metric<Float>>,
    /// The total number of ST instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_generic_stores_sum")]
    pub inst_executed_generic_stores_sum: Option<Metric<Float>>,
    /// The average number of ATOM(ATOM.CAS) instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_global_atomics_avg")]
    pub inst_executed_global_atomics_avg: Option<Metric<Float>>,
    /// The maximum number of ATOM(ATOM.CAS) instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_global_atomics_max")]
    pub inst_executed_global_atomics_max: Option<Metric<Float>>,
    /// The minimum number of ATOM(ATOM.CAS) instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_global_atomics_min")]
    pub inst_executed_global_atomics_min: Option<Metric<Float>>,
    /// The total number of ATOM(ATOM.CAS) instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_global_atomics_sum")]
    pub inst_executed_global_atomics_sum: Option<Metric<Float>>,
    /// The average number of LDG instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_global_loads_avg")]
    pub inst_executed_global_loads_avg: Option<Metric<Float>>,
    /// The maximum number of LDG instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_global_loads_max")]
    pub inst_executed_global_loads_max: Option<Metric<Float>>,
    /// The minimum number of LDG instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_global_loads_min")]
    pub inst_executed_global_loads_min: Option<Metric<Float>>,
    /// The total number of LDG instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_global_loads_sum")]
    pub inst_executed_global_loads_sum: Option<Metric<Float>>,
    /// The average number of RED instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_global_reductions_avg ")]
    pub inst_executed_global_reductions_avg: Option<Metric<Float>>,
    /// The maximum number of RED instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_global_reductions_max")]
    pub inst_executed_global_reductions_max: Option<Metric<Float>>,
    /// The minimum number of RED instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_global_reductions_min")]
    pub inst_executed_global_reductions_min: Option<Metric<Float>>,
    /// The total number of RED instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_global_reductions_sum")]
    pub inst_executed_global_reductions_sum: Option<Metric<Float>>,
    /// The average number of STG instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_global_stores_avg")]
    pub inst_executed_global_stores_avg: Option<Metric<Float>>,
    /// The maximum number of STG instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_global_stores_max")]
    pub inst_executed_global_stores_max: Option<Metric<Float>>,
    /// The minimum number of STG instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_global_stores_min")]
    pub inst_executed_global_stores_min: Option<Metric<Float>>,
    /// The total number of STG instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_global_stores_sum")]
    pub inst_executed_global_stores_sum: Option<Metric<Float>>,
    /// The average number of gs instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_gs_avg")]
    pub inst_executed_gs_avg: Option<Metric<Float>>,
    /// The maximum number of gs instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_gs_max")]
    pub inst_executed_gs_max: Option<Metric<Float>>,
    /// The minimum number of gs instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_gs_min")]
    pub inst_executed_gs_min: Option<Metric<Float>>,
    /// The percentage of instructions executed on a SM scheduler that were gs instructions.
    #[serde(rename = "smsp__inst_executed_gs_pct")]
    pub inst_executed_gs_pct: Option<Metric<Float>>,
    /// The total number of gs instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_gs_sum")]
    pub inst_executed_gs_sum: Option<Metric<Float>>,
    /// The average number of LDL instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_local_loads_avg")]
    pub inst_executed_local_loads_avg: Option<Metric<Float>>,
    /// The maximum number of LDL instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_local_loads_max")]
    pub inst_executed_local_loads_max: Option<Metric<Float>>,
    /// The minimum number of LDL instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_local_loads_min")]
    pub inst_executed_local_loads_min: Option<Metric<Float>>,
    /// The total number of LDL instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_local_loads_sum")]
    pub inst_executed_local_loads_sum: Option<Metric<Float>>,
    /// The average number of STL instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_local_stores_avg")]
    pub inst_executed_local_stores_avg: Option<Metric<Float>>,
    /// The maximum number of STL instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_local_stores_max")]
    pub inst_executed_local_stores_max: Option<Metric<Float>>,
    /// The minimum number of STL instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_local_stores_min")]
    pub inst_executed_local_stores_min: Option<Metric<Float>>,
    /// The total number of STL instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_local_stores_sum")]
    pub inst_executed_local_stores_sum: Option<Metric<Float>>,
    /// The maximum number of instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_max")]
    pub inst_executed_max: Option<Metric<Float>>,
    /// The minimum number of instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_min")]
    pub inst_executed_min: Option<Metric<Float>>,
    /// The number of warp instructions executed per warp by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_per_warp")]
    pub inst_executed_per_warp: Option<Metric<Float>>,
    /// The average number of ATOMS(ATOMS.CAS) instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_shared_atomics_avg")]
    pub inst_executed_shared_atomics_avg: Option<Metric<Float>>,
    /// The maximum number of ATOMS(ATOMS.CAS) instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_shared_atomics_max")]
    pub inst_executed_shared_atomics_max: Option<Metric<Float>>,
    /// The minimum number of ATOMS(ATOMS.CAS) instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_shared_atomics_min")]
    pub inst_executed_shared_atomics_min: Option<Metric<Float>>,
    /// The total number of ATOMS(ATOMS.CAS) instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_shared_atomics_sum")]
    pub inst_executed_shared_atomics_sum: Option<Metric<Float>>,
    /// The average number of LDS instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_shared_loads_avg")]
    pub inst_executed_shared_loads_avg: Option<Metric<Float>>,
    /// The maximum number of LDS instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_shared_loads_max")]
    pub inst_executed_shared_loads_max: Option<Metric<Float>>,
    /// The minimum number of LDS instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_shared_loads_min")]
    pub inst_executed_shared_loads_min: Option<Metric<Float>>,
    /// The total number of LDS instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_shared_loads_sum")]
    pub inst_executed_shared_loads_sum: Option<Metric<Float>>,
    /// The average number of STS instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_shared_stores_avg")]
    pub inst_executed_shared_stores_avg: Option<Metric<Float>>,
    /// The maximum number of STS instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_shared_stores_max")]
    pub inst_executed_shared_stores_max: Option<Metric<Float>>,
    /// The minimum number of STS instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_shared_stores_min")]
    pub inst_executed_shared_stores_min: Option<Metric<Float>>,
    /// The total number of STS instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_shared_stores_sum")]
    pub inst_executed_shared_stores_sum: Option<Metric<Float>>,
    /// The total number of instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_sum")]
    pub inst_executed_sum: Option<Metric<Float>>,
    /// The total number of instructions executed per active cycle by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_sum_per_active_cycle")]
    pub inst_executed_sum_per_active_cycle: Option<Metric<Float>>,
    /// The total number of instructions executed per elapsed cycle by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_sum_per_elapsed_cycle")]
    pub inst_executed_sum_per_elapsed_cycle: Option<Metric<Float>>,
    /// The average number of SUATOM(SUATOM.CAS) instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_surface_atomics_avg")]
    pub inst_executed_surface_atomics_avg: Option<Metric<Float>>,
    /// The maximum number of SUATOM(SUATOM.CAS) instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_surface_atomics_max")]
    pub inst_executed_surface_atomics_max: Option<Metric<Float>>,
    /// The minimum number of SUATOM(SUATOM.CAS) instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_surface_atomics_min")]
    pub inst_executed_surface_atomics_min: Option<Metric<Float>>,
    /// The total number of SUATOM(SUATOM.CAS) instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_surface_atomics_sum")]
    pub inst_executed_surface_atomics_sum: Option<Metric<Float>>,
    /// The average number of SULD instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_surface_loads_avg")]
    pub inst_executed_surface_loads_avg: Option<Metric<Float>>,
    /// The maximum number of SULD instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_surface_loads_max")]
    pub inst_executed_surface_loads_max: Option<Metric<Float>>,
    /// The minimum number of SULD instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_surface_loads_min")]
    pub inst_executed_surface_loads_min: Option<Metric<Float>>,
    /// The total number of SULD instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_surface_loads_sum")]
    pub inst_executed_surface_loads_sum: Option<Metric<Float>>,
    /// The average number of SURED instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_surface_reductions_avg")]
    pub inst_executed_surface_reductions_avg: Option<Metric<Float>>,
    /// The maximum number of SURED instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_surface_reductions_max")]
    pub inst_executed_surface_reductions_max: Option<Metric<Float>>,
    /// The minimum number of SURED instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_surface_reductions_min")]
    pub inst_executed_surface_reductions_min: Option<Metric<Float>>,
    /// The total number of SURED instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_surface_reductions_sum")]
    pub inst_executed_surface_reductions_sum: Option<Metric<Float>>,
    /// The average number of SUST instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_surface_stores_avg")]
    pub inst_executed_surface_stores_avg: Option<Metric<Float>>,
    /// The maximum number of SUST instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_surface_stores_max")]
    pub inst_executed_surface_stores_max: Option<Metric<Float>>,
    /// The minimum number of SUST instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_surface_stores_min")]
    pub inst_executed_surface_stores_min: Option<Metric<Float>>,
    /// The total number of SUST instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_surface_stores_sum")]
    pub inst_executed_surface_stores_sum: Option<Metric<Float>>,
    /// The average number of tcs instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_tcs_avg")]
    pub inst_executed_tcs_avg: Option<Metric<Float>>,
    /// The maximum number of tcs instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_tcs_max")]
    pub inst_executed_tcs_max: Option<Metric<Float>>,
    /// The minimum number of tcs instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_tcs_min")]
    pub inst_executed_tcs_min: Option<Metric<Float>>,
    /// The percentage of instructions executed on a SM scheduler that were tcs instructions.
    #[serde(rename = "smsp__inst_executed_tcs_pct")]
    pub inst_executed_tcs_pct: Option<Metric<Float>>,
    /// The total number of tcs instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_tcs_sum")]
    pub inst_executed_tcs_sum: Option<Metric<Float>>,
    /// The average number of tes instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_tes_avg")]
    pub inst_executed_tes_avg: Option<Metric<Float>>,
    /// The maximum number of tes instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_tes_max")]
    pub inst_executed_tes_max: Option<Metric<Float>>,
    /// The minimum number of tes instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_tes_min")]
    pub inst_executed_tes_min: Option<Metric<Float>>,
    /// The percentage of instructions executed on a SM scheduler that were tes instructions.
    #[serde(rename = "smsp__inst_executed_tes_pct")]
    pub inst_executed_tes_pct: Option<Metric<Float>>,
    /// The total number of tes instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_tes_sum")]
    pub inst_executed_tes_sum: Option<Metric<Float>>,
    /// The total number of TEX, TEXS, TLD, TLDS, TLD4, TLD4S, TMML, TXA, TXD,
    /// and TXQ instructions executed over all SMs.
    #[serde(rename = "smsp__inst_executed_tex_ops")]
    pub inst_executed_tex_ops: Option<Metric<Float>>,
    /// The average number of vs instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_vs_avg")]
    pub inst_executed_vs_avg: Option<Metric<Float>>,
    /// The maximum number of vs instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_vs_max")]
    pub inst_executed_vs_max: Option<Metric<Float>>,
    /// The minimum number of vs instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_vs_min")]
    pub inst_executed_vs_min: Option<Metric<Float>>,
    /// The percentage of instructions executed on a SM scheduler that were vs instructions.
    #[serde(rename = "smsp__inst_executed_vs_pct")]
    pub inst_executed_vs_pct: Option<Metric<Float>>,
    /// The total number of vs instructions executed by a SM scheduler.
    #[serde(rename = "smsp__inst_executed_vs_sum")]
    pub inst_executed_vs_sum: Option<Metric<Float>>,
    // end copy
    /// The average number of instructions executed by active threads by a SM scheduler.
    #[serde(rename = "smsp__thread_inst_executed_avg")]
    pub thread_inst_executed_avg: Option<Metric<Float>>,
    /// The maximum number of instructions executed by active threads by a SM scheduler.
    #[serde(rename = "smsp__thread_inst_executed_max")]
    pub thread_inst_executed_max: Option<Metric<Float>>,
    /// The minimum number of instructions executed by active threads by a SM scheduler.
    #[serde(rename = "smsp__thread_inst_executed_min")]
    pub thread_inst_executed_min: Option<Metric<Float>>,
    /// The average number of thread instructions executed by active not predicated off threads by a SM     scheduler.
    #[serde(rename = "smsp__thread_inst_executed_not_pred_off_avg")]
    pub thread_inst_executed_not_pred_off_avg: Option<Metric<Float>>,
    /// The maximum number of thread instructions executed by active not predicated off threads by a SM scheduler.
    #[serde(rename = "smsp__thread_inst_executed_not_pred_off_max")]
    pub thread_inst_executed_not_pred_off_max: Option<Metric<Float>>,
    /// The minimum number of thread instructions executed by active not predicated off threads by a SM scheduler.
    #[serde(rename = "smsp__thread_inst_executed_not_pred_off_min")]
    pub thread_inst_executed_not_pred_off_min: Option<Metric<Float>>,
    /// The average number of active threads not predicated off per instruction executed.
    #[serde(rename = "smsp__thread_inst_executed_not_pred_off_per_inst_executed")]
    pub thread_inst_executed_not_pred_off_per_inst_executed: Option<Metric<Float>>,
    /// The percentage of active not predicated off threads per instruction executed.
    #[serde(rename = "smsp__thread_inst_executed_not_pred_off_per_inst_executed_pct")]
    pub thread_inst_executed_not_pred_off_per_inst_executed_pct: Option<Metric<Float>>,
    /// The total number of thread instructions executed by active not predicated off threads by a SM scheduler.
    #[serde(rename = "smsp__thread_inst_executed_not_pred_off_sum")]
    pub thread_inst_executed_not_pred_off_sum: Option<Metric<Float>>,
    /// The average number of active threads per instruction executed.
    #[serde(rename = "smsp__thread_inst_executed_per_inst_executed")]
    pub thread_inst_executed_per_inst_executed: Option<Metric<Float>>,
    /// The percentage of active threads per instruction executed.
    #[serde(rename = "smsp__thread_inst_executed_per_inst_executed_pct")]
    pub thread_inst_executed_per_inst_executed_pct: Option<Metric<Float>>,
    /// The total number of instructions executed by active threads by a SM scheduler.
    #[serde(rename = "smsp__thread_inst_executed_sum")]
    pub thread_inst_executed_sum: Option<Metric<Float>>,
}
