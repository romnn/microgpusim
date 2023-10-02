use super::metrics::Float;
use crate::Metric;

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LTS {
    #[serde(rename = "lts__request_tex_atomic_sectors_global_atom_utilization_pct")]
    pub lts_request_tex_atomic_sectors_global_atom_utilization_pct: Option<Metric<Float>>,
    #[serde(rename = "lts__request_tex_atomic_sectors_surface_atom_utilization_pct")]
    pub lts_request_tex_atomic_sectors_surface_atom_utilization_pct: Option<Metric<Float>>,
    #[serde(rename = "lts__request_tex_read_sectors_global_ld_cached_utilization_pct")]
    pub lts_request_tex_read_sectors_global_ld_cached_utilization_pct: Option<Metric<Float>>,
    #[serde(rename = "lts__request_tex_read_sectors_global_ld_uncached_utilization_pct")]
    pub lts_request_tex_read_sectors_global_ld_uncached_utilization_pct: Option<Metric<Float>>,
    #[serde(rename = "lts__request_tex_read_sectors_local_ld_cached_utilization_pct")]
    pub lts_request_tex_read_sectors_local_ld_cached_utilization_pct: Option<Metric<Float>>,
    #[serde(rename = "lts__request_tex_read_sectors_local_ld_uncached_utilization_pct")]
    pub lts_request_tex_read_sectors_local_ld_uncached_utilization_pct: Option<Metric<Float>>,
    #[serde(rename = "lts__request_tex_read_sectors_surface_ld_utilization_pct")]
    pub lts_request_tex_read_sectors_surface_ld_utilization_pct: Option<Metric<Float>>,
    #[serde(rename = "lts__request_tex_write_sectors_global_nonatom_utilization_pct")]
    pub lts_request_tex_write_sectors_global_nonatom_utilization_pct: Option<Metric<Float>>,
    #[serde(rename = "lts__request_tex_write_sectors_global_red_utilization_pct")]
    pub lts_request_tex_write_sectors_global_red_utilization_pct: Option<Metric<Float>>,
    #[serde(rename = "lts__request_tex_write_sectors_local_st_utilization_pct")]
    pub lts_request_tex_write_sectors_local_st_utilization_pct: Option<Metric<Float>>,
    #[serde(rename = "lts__request_tex_write_sectors_surface_nonatom_utilization_pct")]
    pub lts_request_tex_write_sectors_surface_nonatom_utilization_pct: Option<Metric<Float>>,
    #[serde(rename = "lts__request_tex_write_sectors_surface_red_utilization_pct")]
    pub lts_request_tex_write_sectors_surface_red_utilization_pct: Option<Metric<Float>>,
    #[serde(rename = "lts__request_total_sectors_hitrate_pct")]
    pub lts_request_total_sectors_hitrate_pct: Option<Metric<Float>>,
    #[serde(rename = "lts__t_sector_op_read_hit_rate.pct")]
    pub lts_t_sector_op_read_hit_rate_pct: Option<Metric<Float>>,
    #[serde(rename = "lts__t_sector_op_write_hit_rate.pct")]
    pub lts_t_sector_op_write_hit_rate_pct: Option<Metric<Float>>,
    #[serde(rename = "lts__t_sectors_srcunit_tex_op_read.sum")]
    pub lts_t_sectors_srcunit_tex_op_read_sum: Option<Metric<Float>>,
    #[serde(rename = "lts__t_sectors_srcunit_tex_op_write.sum")]
    pub lts_t_sectors_srcunit_tex_op_write_sum: Option<Metric<Float>>,
    #[serde(rename = "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum")]
    pub lts_t_sectors_srcunit_tex_op_read_lookup_hit_sum: Option<Metric<Float>>,
    #[serde(rename = "lts__t_sectors_srcunit_tex_op_write_lookup_hit.sum")]
    pub lts_t_sectors_srcunit_tex_op_write_lookup_hit_sum: Option<Metric<Float>>,
    #[serde(rename = "lts__t_sectors_srcunit_tex_op_read.sum.per_second")]
    pub lts_t_sectors_srcunit_tex_op_read_sum_per_second: Option<Metric<Float>>,
    #[serde(rename = "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum")]
    pub lts_t_sectors_srcunit_tex_op_read_lookup_miss_sum: Option<Metric<Float>>,
    #[serde(rename = "lts__t_sectors_srcunit_tex_op_write_lookup_miss.sum")]
    pub lts_t_sectors_srcunit_tex_op_write_lookup_miss_sum: Option<Metric<Float>>,
    /// Number of cycles the lts is busy.
    #[serde(rename = "lts__busy_cycles_avg")]
    pub busy_cycles_avg: Option<Metric<Float>>,
    /// Number of cycles the busiest lts is busy.
    #[serde(rename = "lts__busy_cycles_max")]
    pub busy_cycles_max: Option<Metric<Float>>,
    /// Number of cycles the sum of lts is busy.
    #[serde(rename = "lts__busy_cycles_sum")]
    pub busy_cycles_sum: Option<Metric<Float>>,
    /// Percentage of elapsed cycles the lts is busy.
    #[serde(rename = "lts__busy_pct_avg")]
    pub busy_pct_avg: Option<Metric<Float>>,
    /// Percentage of elapsed cycles the busiest lts is busy.
    #[serde(rename = "lts__busy_pct_max")]
    pub busy_pct_max: Option<Metric<Float>>,
    /// Percentage of elapsed cycles the sum of lts is busy.
    #[serde(rename = "lts__busy_pct_sum")]
    pub busy_pct_sum: Option<Metric<Float>>,
    /// The average count of the number of cycles within a range for a lts unit instance.
    #[serde(rename = "lts__elapsed_cycles_avg")]
    pub elapsed_cycles_avg: Option<Metric<Float>>,
    /// The maximum count of the number of cycles within a range for a lts unit instance.
    #[serde(rename = "lts__elapsed_cycles_max")]
    pub elapsed_cycles_max: Option<Metric<Float>>,
    /// The minimum count of the number of cycles within a range for a lts unit instance.
    #[serde(rename = "lts__elapsed_cycles_min")]
    pub elapsed_cycles_min: Option<Metric<Float>>,
    /// The total count of the number of cycles within a range for a lts unit instance.
    #[serde(rename = "lts__elapsed_cycles_sum")]
    pub elapsed_cycles_sum: Option<Metric<Float>>,
    /// The average frequency of the lts unit(s) in Hz.
    ///
    /// This is calculated as lts__elapsed_cycles_avg divided by gpu__time_duration.
    /// The value will be lower than expected if the measurement range contains GPU context switches.
    #[serde(rename = "lts__frequency")]
    pub frequency: Option<Metric<Float>>,
    /// Percentage of lts requests by crop to the total possible lts requests over the measurement range.
    #[serde(rename = "lts__request_crop_read_utilization_pct")]
    pub request_crop_read_utilization_pct: Option<Metric<Float>>,
    /// Percentage of lts requests by crop to the total possible lts requests over the measurement range.
    #[serde(rename = "lts__request_crop_utilization_pct")]
    pub request_crop_utilization_pct: Option<Metric<Float>>,
    /// Percentage of lts requests by crop to the total possible lts requests over the measurement range.
    #[serde(rename = "lts__request_crop_write_utilization_pct")]
    pub request_crop_write_utilization_pct: Option<Metric<Float>>,
    /// Number of lts bytes read per second by ia.
    #[serde(rename = "lts__request_ia_read_utilization_pct")]
    pub request_ia_read_utilization_pct: Option<Metric<Float>>,
    /// Number of bytes read by TEX global atom requests.
    #[serde(rename = "lts__request_tex_atomic_bytes_global_atom")]
    pub request_tex_atomic_bytes_global_atom: Option<Metric<Float>>,
    /// The read throughput in bytes per second by TEX global atom requests.
    #[serde(rename = "lts__request_tex_atomic_bytes_per_sec_global_atom")]
    pub request_tex_atomic_bytes_per_sec_global_atom: Option<Metric<Float>>,
    /// The read throughput in bytes per second by TEX surface atom requests.
    #[serde(rename = "lts__request_tex_atomic_bytes_per_sec_surface_atom")]
    pub request_tex_atomic_bytes_per_sec_surface_atom: Option<Metric<Float>>,
    /// Number of bytes read by TEX surface atom requests.
    #[serde(rename = "lts__request_tex_atomic_bytes_surface_atom")]
    pub request_tex_atomic_bytes_surface_atom: Option<Metric<Float>>,
    /// Number of lts sectors accessed for atomic cas by tex.
    #[serde(rename = "lts__request_tex_atomic_cas_sectors")]
    pub request_tex_atomic_cas_sectors: Option<Metric<Float>>,
    /// Percentage of lts requests by tex to the total possible lts requests over the measurement range.
    #[serde(rename = "lts__request_tex_atomic_cas_utilization_pct")]
    pub request_tex_atomic_cas_utilization_pct: Option<Metric<Float>>,
    /// Number of lts sectors accessed for atomic by tex.
    #[serde(rename = "lts__request_tex_atomic_sctors")]
    pub request_tex_atomic_sctors: Option<Metric<Float>>,
    /// Number of sectors read by TEX global atom requests.
    #[serde(rename = "lts__request_tex_atomic_sectors_global_atom")]
    pub request_tex_atomic_sectors_global_atom: Option<Metric<Float>>,
    /// Percentage utilization of LTS reads by TEX atomic and reduction requests.
    #[serde(rename = "lts__request_tex_atomic_sectors_global_atom_red_pct")]
    pub request_tex_atomic_sectors_global_atom_red_pct: Option<Metric<Float>>,
    /// Percentage utilization of LTS reads by TEX global atom requests.
    #[serde(rename = "lts__request_tex_atomic_sectors_global_atom_utilization_pct")]
    pub request_tex_atomic_sectors_global_atom_utilization_pct: Option<Metric<Float>>,
    /// Number of sectors read by TEX surface atom requests
    #[serde(rename = "lts__request_tex_atomic_sectors_surface_atom")]
    pub request_tex_atomic_sectors_surface_atom: Option<Metric<Float>>,
    /// Percentage utilization of LTS reads by TEX surface atom requests.
    #[serde(rename = "lts__request_tex_atomic_sectors_surface_atom_utilization_pct")]
    pub request_tex_atomic_sectors_surface_atom_utilization_pct: Option<Metric<Float>>,
    /// Percentage of lts requests by tex to the total possible lts requests over the measurement range
    #[serde(rename = "lts__request_tex_atomic_utilization_pct")]
    pub request_tex_atomic_utilization_pct: Option<Metric<Float>>,
    /// Number of bytes read by TEX cached global ld requests.
    #[serde(rename = "lts__request_tex_read_bytes_global_ld_cached")]
    pub request_tex_read_bytes_global_ld_cached: Option<Metric<Float>>,
    /// Number of bytes read by TEX uncached global ld requests.
    #[serde(rename = "lts__request_tex_read_bytes_global_ld_uncached")]
    pub request_tex_read_bytes_global_ld_uncached: Option<Metric<Float>>,
    /// Number of bytes read by TEX cached local ld requests.
    #[serde(rename = "lts__request_tex_read_bytes_local_ld_cached")]
    pub request_tex_read_bytes_local_ld_cached: Option<Metric<Float>>,
    /// Number of bytes read by TEX uncached local ld requests.
    #[serde(rename = "lts__request_tex_read_bytes_local_ld_uncached")]
    pub request_tex_read_bytes_local_ld_uncached: Option<Metric<Float>>,
    /// Throughput of reads by TEX cached global ld requests in bytes per second.
    #[serde(rename = "lts__request_tex_read_bytes_per_sec_global_ld_cached")]
    pub request_tex_read_bytes_per_sec_global_ld_cached: Option<Metric<Float>>,
    /// Throughput of reads by TEX uncached global ld requests in bytes per second.
    #[serde(rename = "lts__request_tex_read_bytes_per_sec_global_ld_uncached")]
    pub request_tex_read_bytes_per_sec_global_ld_uncached: Option<Metric<Float>>,
    /// Throughput of reads by TEX cached local ld requests in bytes per second.
    #[serde(rename = "lts__request_tex_read_bytes_per_sec_local_ld_cached")]
    pub request_tex_read_bytes_per_sec_local_ld_cached: Option<Metric<Float>>,
    /// Throughput of reads by TEX uncached local ld requests in bytes per second.
    #[serde(rename = "lts__request_tex_read_bytes_per_sec_local_ld_uncached")]
    pub request_tex_read_bytes_per_sec_local_ld_uncached: Option<Metric<Float>>,
    /// Throughput of reads by TEX surface ld requests in bytes per second.
    #[serde(rename = "lts__request_tex_read_bytes_per_sec_surface_ld")]
    pub request_tex_read_bytes_per_sec_surface_ld: Option<Metric<Float>>,
    /// Number of bytes read by TEX surface ld requests.
    #[serde(rename = "lts__request_tex_read_bytes_surface_ld")]
    pub request_tex_read_bytes_surface_ld: Option<Metric<Float>>,
    /// Number of lts sectors read by tex.
    #[serde(rename = "lts__request_tex_read_sectors")]
    pub request_tex_read_sectors: Option<Metric<Float>>,
    /// Number of sectors read by TEX cached global ld requests.
    #[serde(rename = "lts__request_tex_read_sectors_global_ld_cached")]
    pub request_tex_read_sectors_global_ld_cached: Option<Metric<Float>>,
    /// Percentage utilization of LTS read by TEX cached global ld.
    #[serde(rename = "lts__request_tex_read_sectors_global_ld_cached_utilization_pct")]
    pub request_tex_read_sectors_global_ld_cached_utilization_pct: Option<Metric<Float>>,
    /// Number of sectors read by TEX uncached global ld requests.
    #[serde(rename = "lts__request_tex_read_sectors_global_ld_uncached")]
    pub request_tex_read_sectors_global_ld_uncached: Option<Metric<Float>>,
    /// Percentage utilization of LTS read by TEX uncached global ld.
    #[serde(rename = "lts__request_tex_read_sectors_global_ld_uncached_utilization_pct")]
    pub request_tex_read_sectors_global_ld_uncached_utilization_pct: Option<Metric<Float>>,
    /// Number of sectors read by TEX cached local ld requests.
    #[serde(rename = "lts__request_tex_read_sectors_local_ld_cached")]
    pub request_tex_read_sectors_local_ld_cached: Option<Metric<Float>>,
    /// Percentage utilization of LTS read by TEX cached local ld.
    #[serde(rename = "lts__request_tex_read_sectors_local_ld_cached_utilization_pct")]
    pub request_tex_read_sectors_local_ld_cached_utilization_pct: Option<Metric<Float>>,
    /// Number of sectors read by TEX uncached local ld requests.
    #[serde(rename = "lts__request_tex_read_sectors_local_ld_uncached")]
    pub request_tex_read_sectors_local_ld_uncached: Option<Metric<Float>>,
    /// Percentage utilization of LTS read by TEX uncached local ld.
    #[serde(rename = "lts__request_tex_read_sectors_local_ld_uncached_utilization_pct")]
    pub request_tex_read_sectors_local_ld_uncached_utilization_pct: Option<Metric<Float>>,
    /// Number of sectors read by TEX surface ld requests.
    #[serde(rename = "lts__request_tex_read_sectors_surface_ld")]
    pub request_tex_read_sectors_surface_ld: Option<Metric<Float>>,
    /// Percentage utilization of LTS read by TEX surface ld.
    #[serde(rename = "lts__request_tex_read_sectors_surface_ld_utilization_pct")]
    pub request_tex_read_sectors_surface_ld_utilization_pct: Option<Metric<Float>>,
    /// Percentage of lts requests by tex to the total possible lts requests over the measurement range.
    #[serde(rename = "lts__request_tex_read_utilization_pct")]
    pub request_tex_read_utilization_pct: Option<Metric<Float>>,
    /// Number of lts sectors read or written by tex.
    #[serde(rename = "lts__request_tex_sectors")]
    pub request_tex_sectors: Option<Metric<Float>>,
    /// Percentage of lts requests by tex to the total possible lts requests over the measurement range.
    #[serde(rename = "lts__request_tex_utilization_pct")]
    pub request_tex_utilization_pct: Option<Metric<Float>>,
    /// Number of bytes written by TEX global nonatom requests.
    #[serde(rename = "lts__request_tex_write_bytes_global_nonatom")]
    pub request_tex_write_bytes_global_nonatom: Option<Metric<Float>>,
    /// Number of bytes written by TEX global red requests.
    #[serde(rename = "lts__request_tex_write_bytes_global_red")]
    pub request_tex_write_bytes_global_red: Option<Metric<Float>>,
    /// Number of bytes written by TEX local st requests.
    #[serde(rename = "lts__request_tex_write_bytes_local_st")]
    pub request_tex_write_bytes_local_st: Option<Metric<Float>>,
    /// The write throughput in bytes per second by TEX global nonatom requests.
    #[serde(rename = "lts__request_tex_write_bytes_per_sec_global_nonatom")]
    pub request_tex_write_bytes_per_sec_global_nonatom: Option<Metric<Float>>,
    /// The write throughput in bytes per second by TEX global red requests.
    #[serde(rename = "lts__request_tex_write_bytes_per_sec_global_red")]
    pub request_tex_write_bytes_per_sec_global_red: Option<Metric<Float>>,
    /// The write throughput in bytes per second by TEX local st requests.
    #[serde(rename = "lts__request_tex_write_bytes_per_sec_local_st")]
    pub request_tex_write_bytes_per_sec_local_st: Option<Metric<Float>>,
    /// The write throughput in bytes per second by TEX surface nonatom requests.
    #[serde(rename = "lts__request_tex_write_bytes_per_sec_surface_nonatom")]
    pub request_tex_write_bytes_per_sec_surface_nonatom: Option<Metric<Float>>,
    /// The write throughput in bytes per second by TEX surface red requests.
    #[serde(rename = "lts__request_tex_write_bytes_per_sec_surface_red")]
    pub request_tex_write_bytes_per_sec_surface_red: Option<Metric<Float>>,
    /// Number of bytes written by TEX surface nonatom requests.
    #[serde(rename = "lts__request_tex_write_bytes_surface_nonatom")]
    pub request_tex_write_bytes_surface_nonatom: Option<Metric<Float>>,
    /// Number of bytes written by TEX surface red requests.
    #[serde(rename = "lts__request_tex_write_bytes_surface_red")]
    pub request_tex_write_bytes_surface_red: Option<Metric<Float>>,
    /// Number of lts sectors written by tex.
    #[serde(rename = "lts__request_tex_write_sectors")]
    pub request_tex_write_sectors: Option<Metric<Float>>,
    /// Number of sectors written by TEX global nonatom requests.
    #[serde(rename = "lts__request_tex_write_sectors_global_nonatom")]
    pub request_tex_write_sectors_global_nonatom: Option<Metric<Float>>,
    /// Percentage utilization of LTS writes by TEX global nonatom requests.
    #[serde(rename = "lts__request_tex_write_sectors_global_nonatom_utilization_pct")]
    pub request_tex_write_sectors_global_nonatom_utilization_pct: Option<Metric<Float>>,
    /// Number of sectors written by TEX global red requests.
    #[serde(rename = "lts__request_tex_write_sectors_global_red")]
    pub request_tex_write_sectors_global_red: Option<Metric<Float>>,
    /// Percentage utilization of LTS writes by TEX global red requests.
    #[serde(rename = "lts__request_tex_write_sectors_global_red_utilization_pct")]
    pub request_tex_write_sectors_global_red_utilization_pct: Option<Metric<Float>>,
    /// Number of sectors written by TEX local st requests.
    #[serde(rename = "lts__request_tex_write_sectors_local_st")]
    pub request_tex_write_sectors_local_st: Option<Metric<Float>>,
    /// Percentage utilization of LTS writes by TEX local st requests.
    #[serde(rename = "lts__request_tex_write_sectors_local_st_utilization_pct")]
    pub request_tex_write_sectors_local_st_utilization_pct: Option<Metric<Float>>,
    /// Number of sectors written by TEX surface nonatom requests.
    #[serde(rename = "lts__request_tex_write_sectors_surface_nonatom")]
    pub request_tex_write_sectors_surface_nonatom: Option<Metric<Float>>,
    /// Percentage utilization of LTS writes by TEX surface nonatom requests.
    #[serde(rename = "lts__request_tex_write_sectors_surface_nonatom_utilization_pct")]
    pub request_tex_write_sectors_surface_nonatom_utilization_pct: Option<Metric<Float>>,
    /// Number of sectors written by TEX surface red requests.
    #[serde(rename = "lts__request_tex_write_sectors_surface_red")]
    pub request_tex_write_sectors_surface_red: Option<Metric<Float>>,
    /// Percentage utilization of LTS writes by TEX surface red requests.
    #[serde(rename = "lts__request_tex_write_sectors_surface_red_utilization_pct")]
    pub request_tex_write_sectors_surface_red_utilization_pct: Option<Metric<Float>>,
    /// Percentage of lts requests by tex to the total possible lts requests over the measurement range.
    #[serde(rename = "lts__request_tex_write_utilization_pct")]
    pub request_tex_write_utilization_pct: Option<Metric<Float>>,
    /// Percentage of all lts requested sectors that hit.
    #[serde(rename = "lts__request_total_sectors_hitrate_pct")]
    pub request_total_sectors_hitrate_pct: Option<Metric<Float>>,
    /// Percentage of lts requests by zrop to the total possible lts requests over the measurement range.
    #[serde(rename = "lts__request_zrop_read_utilization_pct")]
    pub request_zrop_read_utilization_pct: Option<Metric<Float>>,
    /// Percentage of lts requests by zrop to the total possible lts requests over the measurement range.
    #[serde(rename = "lts__request_zrop_utilization_pct")]
    pub request_zrop_utilization_pct: Option<Metric<Float>>,
    /// Percentage of lts requests by zrop to the total possible lts requests over the measurement range.
    #[serde(rename = "lts__request_zrop_write_utilization_pct")]
    pub request_zrop_write_utilization_pct: Option<Metric<Float>>,
}
