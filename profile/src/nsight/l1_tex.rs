use super::metrics::Float;
use crate::Metric;

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct L1Tex {
    #[serde(rename = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")]
    pub l1tex_t_sectors_pipe_lsu_mem_global_op_ld_sum: Option<Metric<Float>>,
    #[serde(rename = "l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit.sum")]
    pub l1tex_t_sectors_pipe_lsu_mem_global_op_st_lookup_hit_sum: Option<Metric<Float>>,
    #[serde(rename = "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum")]
    pub l1tex_t_sectors_pipe_lsu_mem_global_op_st_sum: Option<Metric<Float>>,
    #[serde(rename = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum")]
    pub l1tex_t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit_sum: Option<Metric<Float>>,
    #[serde(rename = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum")]
    pub l1tex_t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss_sum: Option<Metric<Float>>,
    #[serde(rename = "l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_miss.sum")]
    pub l1tex_t_sectors_pipe_lsu_mem_global_op_st_lookup_miss_sum: Option<Metric<Float>>,
}
