mod metrics;

pub use metrics::Metrics;

#[derive(PartialEq, Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Output {
    // pub raw_metrics_log: String,
    // pub raw_commands_log: String,
    // pub metrics: Metrics,
}
