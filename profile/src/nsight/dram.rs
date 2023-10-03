use super::metrics::Float;
use crate::Metric;

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DRAM {
    #[serde(rename = "dram__bytes_per_sec")]
    pub bytes_per_sec: Option<Metric<Float>>,
    #[serde(rename = "dram__frequency")]
    pub frequency: Option<Metric<Float>>,
    #[serde(rename = "dram__read_bytes")]
    pub read_bytes: Option<Metric<Float>>,
    #[serde(rename = "dram__read_bytes_per_sec")]
    pub read_bytes_per_sec: Option<Metric<Float>>,
    #[serde(rename = "dram__read_pct")]
    pub read_pct: Option<Metric<Float>>,
    #[serde(rename = "dram__read_sectors")]
    pub read_sectors: Option<Metric<Float>>,
    #[serde(rename = "dram__write_bytes")]
    pub write_bytes: Option<Metric<Float>>,
    #[serde(rename = "dram__write_bytes_per_sec")]
    pub write_bytes_per_sec: Option<Metric<Float>>,
    #[serde(rename = "dram__write_pct")]
    pub write_pct: Option<Metric<Float>>,
    #[serde(rename = "dram__write_sectors")]
    pub write_sectors: Option<Metric<Float>>,
    #[serde(rename = "dram__sectors_read.sum")]
    pub sectors_read_sum: Option<Metric<Float>>,
    #[serde(rename = "dram__sectors_write.sum")]
    pub sectors_write_sum: Option<Metric<Float>>,
    #[serde(rename = "dram__bytes_read.sum")]
    pub bytes_read_sum: Option<Metric<Float>>,
}
