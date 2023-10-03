use crate::Metric;

#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[repr(transparent)]
#[serde(transparent)]
pub struct Float(pub f32);

impl From<f32> for Float {
    fn from(value: f32) -> Self {
        Self(value)
    }
}

impl From<Float> for f32 {
    fn from(value: Float) -> Self {
        value.0
    }
}

#[derive(thiserror::Error, Debug, PartialEq)]
pub enum ParseFloatError {
    #[error(transparent)]
    Parse(#[from] std::num::ParseFloatError),
    #[error("bad format: {value:?} ({reason})")]
    BadFormat { value: String, reason: String },
}

impl std::str::FromStr for Float {
    type Err = ParseFloatError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut s = s.to_ascii_lowercase();

        if s.chars().count() > 0 {
            let first_comma = s.chars().position(|c| c == ',');
            let first_dot = s.chars().position(|c| c == '.');
            match (first_comma, first_dot) {
                (Some(_), None) => {
                    let num_commas = s.chars().filter(|&c| c == ',').count();
                    if num_commas > 1 {
                        // remove commas
                        s = s
                            .chars()
                            .filter(|&c| c != ',' && c != ' ')
                            .collect::<String>();
                    } else {
                        return Err(ParseFloatError::BadFormat {
                            value: s,
                            reason: "comma without floating point".to_string(),
                        });
                    }
                }
                (Some(first_comma), Some(first_dot)) => {
                    // sanity check comma before dot
                    if first_comma >= first_dot {
                        return Err(ParseFloatError::BadFormat {
                            value: s,
                            reason: "decimal point followed by comma separator".to_string(),
                        });
                    }

                    // todo: sanity check only single dot
                    // remove commas
                    s = s
                        .chars()
                        .filter(|&c| c != ',' && c != ' ')
                        .collect::<String>();
                }
                _ => {}
            }
        }
        let value = f32::from_str(&s)?;
        Ok(Self(value))
    }
}

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Metrics {
    #[serde(flatten)]
    pub device: super::device::Device,
    #[serde(flatten)]
    pub dram: super::dram::DRAM,
    #[serde(flatten)]
    pub profiler: super::profiler::Profiler,
    /// LTS is a level 2 cache slice (sub-partition).
    #[serde(flatten)]
    pub lts: super::l2::L2CacheSlice,
    #[serde(flatten)]
    pub sm: super::sm::SM,
    #[serde(flatten)]
    pub sm_scheduler: super::scheduler::SMScheduler,
    #[serde(flatten)]
    pub tex: super::tex::Tex,
    #[serde(flatten)]
    pub l1_tex: super::l1_tex::L1Tex,

    // other
    #[serde(rename = "ID")]
    pub id: Option<Metric<usize>>,
    #[serde(rename = "Process ID")]
    pub process_id: Option<Metric<usize>>,
    #[serde(rename = "Process Name")]
    pub process_name: Option<Metric<String>>,
    #[serde(rename = "Host Name")]
    pub host_name: Option<Metric<String>>,
    #[serde(rename = "Kernel Name")]
    pub kernel_name: Option<Metric<String>>,
    #[serde(rename = "Kernel Time")]
    pub kernel_time: Option<Metric<String>>,
    #[serde(rename = "Context")]
    pub context: Option<Metric<usize>>,
    #[serde(rename = "Stream")]
    pub stream: Option<Metric<usize>>,
    #[serde(rename = "fbpa__sol_pct")]
    pub fbpa_sol_pct: Option<Metric<Float>>,
    #[serde(rename = "gpc__elapsed_cycles_max")]
    pub gpc_elapsed_cycles_max: Option<Metric<Float>>,
    #[serde(rename = "gpc__elapsed_cycles.avg")]
    pub gpc_elapsed_cycles_avg: Option<Metric<Float>>,
    #[serde(rename = "gpc__frequency")]
    pub gpc_frequency: Option<Metric<Float>>,
    #[serde(rename = "gpu__compute_memory_request_utilization_pct")]
    pub gpu_compute_memory_request_utilization_pct: Option<Metric<Float>>,
    #[serde(rename = "gpu__compute_memory_sol_pct")]
    pub gpu_compute_memory_sol_pct: Option<Metric<Float>>,
    #[serde(rename = "gpu__time_duration")]
    pub gpu_time_duration: Option<Metric<Float>>,
    #[serde(rename = "inst_executed")]
    pub inst_executed: Option<Metric<String>>,
    #[serde(rename = "launch__block_size")]
    pub launch_block_size: Option<Metric<Float>>,
    #[serde(rename = "launch__context_id")]
    pub launch_context_id: Option<Metric<Float>>,
    #[serde(rename = "launch__function_pcs")]
    pub launch_function_pcs: Option<Metric<Float>>,
    #[serde(rename = "launch__grid_size")]
    pub launch_grid_size: Option<Metric<Float>>,
    #[serde(rename = "launch__occupancy_limit_blocks")]
    pub launch_occupancy_limit_blocks: Option<Metric<Float>>,
    #[serde(rename = "launch__occupancy_limit_registers")]
    pub launch_occupancy_limit_registers: Option<Metric<Float>>,
    #[serde(rename = "launch__occupancy_limit_shared_mem")]
    pub launch_occupancy_limit_shared_mem: Option<Metric<Float>>,
    #[serde(rename = "launch__occupancy_limit_warps")]
    pub launch_occupancy_limit_warps: Option<Metric<Float>>,
    #[serde(rename = "launch__occupancy_per_block_size")]
    pub launch_occupancy_per_block_size: Option<Metric<Float>>,
    #[serde(rename = "launch__occupancy_per_register_count")]
    pub launch_occupancy_per_register_count: Option<Metric<Float>>,
    #[serde(rename = "launch__occupancy_per_shared_mem_size")]
    pub launch_occupancy_per_shared_mem_size: Option<Metric<Float>>,
    #[serde(rename = "launch__registers_per_thread")]
    pub launch_registers_per_thread: Option<Metric<Float>>,
    #[serde(rename = "launch__registers_per_thread_allocated")]
    pub launch_registers_per_thread_allocated: Option<Metric<Float>>,
    #[serde(rename = "launch__shared_mem_config_size")]
    pub launch_shared_mem_config_size: Option<Metric<Float>>,
    #[serde(rename = "launch__shared_mem_per_block_allocated")]
    pub launch_shared_mem_per_block_allocated: Option<Metric<Float>>,
    #[serde(rename = "launch__shared_mem_per_block_dynamic")]
    pub launch_shared_mem_per_block_dynamic: Option<Metric<Float>>,
    #[serde(rename = "launch__shared_mem_per_block_static")]
    pub launch_shared_mem_per_block_static: Option<Metric<Float>>,
    #[serde(rename = "launch__stream_id")]
    pub launch_stream_id: Option<Metric<Float>>,
    #[serde(rename = "launch__thread_count")]
    pub launch_thread_count: Option<Metric<Float>>,
    #[serde(rename = "launch__waves_per_multiprocessor")]
    pub launch_waves_per_multiprocessor: Option<Metric<Float>>,
    #[serde(rename = "ltc__sol_pct")]
    pub ltc_sol_pct: Option<Metric<Float>>,
    #[serde(rename = "memory_access_size_type")]
    pub memory_access_size_type: Option<Metric<String>>,
    #[serde(rename = "memory_access_type")]
    pub memory_access_type: Option<Metric<String>>,
    #[serde(rename = "memory_l2_transactions_global")]
    pub memory_l2_transactions_global: Option<Metric<String>>,
    #[serde(rename = "memory_l2_transactions_local")]
    pub memory_l2_transactions_local: Option<Metric<String>>,
    #[serde(rename = "memory_shared_transactions")]
    pub memory_shared_transactions: Option<Metric<Float>>,
    #[serde(rename = "memory_type")]
    pub memory_type: Option<Metric<String>>,
    #[serde(rename = "sass__block_histogram")]
    pub sass_block_histogram: Option<Metric<Float>>,
    #[serde(rename = "sass__inst_executed_global_loads")]
    pub sass_inst_executed_global_loads: Option<Metric<Float>>,
    #[serde(rename = "sass__inst_executed_global_stores")]
    pub sass_inst_executed_global_stores: Option<Metric<Float>>,
    #[serde(rename = "sass__inst_executed_local_loads")]
    pub sass_inst_executed_local_loads: Option<Metric<Float>>,
    #[serde(rename = "sass__inst_executed_local_stores")]
    pub sass_inst_executed_local_stores: Option<Metric<Float>>,
    #[serde(rename = "sass__inst_executed_per_opcode")]
    pub sass_inst_executed_per_opcode: Option<Metric<Float>>,
    #[serde(rename = "sass__inst_executed_shared_loads")]
    pub sass_inst_executed_shared_loads: Option<Metric<Float>>,
    #[serde(rename = "sass__inst_executed_shared_stores")]
    pub sass_inst_executed_shared_stores: Option<Metric<Float>>,
    #[serde(rename = "sass__warp_histogram")]
    pub sass_warp_histogram: Option<Metric<Float>>,
    #[serde(rename = "thread_inst_executed_true")]
    pub thread_inst_executed_true: Option<Metric<String>>,
    // #[serde(flatten)]
    // pub other: std::collections::HashMap<String, Metric<serde_json::Value>>,
}

#[cfg(test)]
mod tests {
    use super::Float;
    use std::str::FromStr;

    #[test]
    fn test_parse_float() {
        assert_eq!(Float::from_str("12"), Ok(12.0.into()));
        assert_eq!(
            Float::from_str("12,00").ok(),
            None,
            "cannot interpret single comma"
        );
        assert_eq!(
            Float::from_str("12,001,233"),
            Ok(12_001_233.0.into()),
            "multiple comma separators disambiguate"
        );
        assert_eq!(
            Float::from_str("12,001,233.5347"),
            Ok(12_001_233.5347.into())
        );
        assert_eq!(
            Float::from_str("12.001,233").ok(),
            None,
            "cannot have decimal point followed by comma separator"
        );
        assert_eq!(Float::from_str("-0.5"), Ok((-0.5).into()));
    }
}
