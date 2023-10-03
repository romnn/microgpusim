use super::metrics::Float;
use crate::Metric;

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Profiler {
    #[serde(rename = "profiler__replayer_bytes_mem_accessible.avg")]
    pub profiler_replayer_bytes_mem_accessible_avg: Option<Metric<Float>>,
    #[serde(rename = "profiler__replayer_bytes_mem_accessible.max")]
    pub profiler_replayer_bytes_mem_accessible_max: Option<Metric<Float>>,
    #[serde(rename = "profiler__replayer_bytes_mem_accessible.min")]
    pub profiler_replayer_bytes_mem_accessible_min: Option<Metric<Float>>,
    #[serde(rename = "profiler__replayer_bytes_mem_accessible.sum")]
    pub profiler_replayer_bytes_mem_accessible_sum: Option<Metric<Float>>,
    #[serde(rename = "profiler__replayer_bytes_mem_backed_up.avg")]
    pub profiler_replayer_bytes_mem_backed_up_avg: Option<Metric<Float>>,
    #[serde(rename = "profiler__replayer_bytes_mem_backed_up.max")]
    pub profiler_replayer_bytes_mem_backed_up_max: Option<Metric<Float>>,
    #[serde(rename = "profiler__replayer_bytes_mem_backed_up.min")]
    pub profiler_replayer_bytes_mem_backed_up_min: Option<Metric<Float>>,
    #[serde(rename = "profiler__replayer_bytes_mem_backed_up.sum")]
    pub profiler_replayer_bytes_mem_backed_up_sum: Option<Metric<Float>>,
    #[serde(rename = "profiler__replayer_passes")]
    pub profiler_replayer_passes: Option<Metric<Float>>,
    #[serde(rename = "profiler__replayer_passes_type_warmup")]
    pub profiler_replayer_passes_type_warmup: Option<Metric<Float>>,
}
