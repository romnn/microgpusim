use super::metrics::Float;
use crate::Metric;

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Tex {
    #[serde(rename = "tex__global_ld_unique_sector_requests")]
    pub global_ld_unique_sector_requests: Option<Metric<Float>>,
    #[serde(rename = "tex__local_ld_unique_sector_requests")]
    pub local_ld_unique_sector_requests: Option<Metric<Float>>,
    #[serde(rename = "tex__m_rd_sectors_global_atom")]
    pub m_rd_sectors_global_atom: Option<Metric<Float>>,
    #[serde(rename = "tex__m_wr_bytes_global_atom")]
    pub m_wr_bytes_global_atom: Option<Metric<Float>>,
    #[serde(rename = "tex__m_wr_bytes_global_atom_per_sec")]
    pub m_wr_bytes_global_atom_per_sec: Option<Metric<Float>>,
    #[serde(rename = "tex__m_wr_bytes_global_nonatom")]
    pub m_wr_bytes_global_nonatom: Option<Metric<Float>>,
    #[serde(rename = "tex__m_wr_bytes_global_nonatom_per_sec")]
    pub m_wr_bytes_global_nonatom_per_sec: Option<Metric<Float>>,
    #[serde(rename = "tex__m_wr_bytes_global_red")]
    pub m_wr_bytes_global_red: Option<Metric<Float>>,
    #[serde(rename = "tex__m_wr_bytes_global_red_per_sec")]
    pub m_wr_bytes_global_red_per_sec: Option<Metric<Float>>,
    #[serde(rename = "tex__m_wr_bytes_local_st")]
    pub m_wr_bytes_local_st: Option<Metric<Float>>,
    #[serde(rename = "tex__m_wr_bytes_local_st_per_sec")]
    pub m_wr_bytes_local_st_per_sec: Option<Metric<Float>>,
    #[serde(rename = "tex__m_wr_bytes_surface_atom")]
    pub m_wr_bytes_surface_atom: Option<Metric<Float>>,
    #[serde(rename = "tex__m_wr_bytes_surface_atom_per_sec")]
    pub m_wr_bytes_surface_atom_per_sec: Option<Metric<Float>>,
    #[serde(rename = "tex__m_wr_bytes_surface_nonatom")]
    pub m_wr_bytes_surface_nonatom: Option<Metric<Float>>,
    #[serde(rename = "tex__m_wr_bytes_surface_nonatom_per_sec")]
    pub m_wr_bytes_surface_nonatom_per_sec: Option<Metric<Float>>,
    #[serde(rename = "tex__m_wr_bytes_surface_red")]
    pub m_wr_bytes_surface_red: Option<Metric<Float>>,
    #[serde(rename = "tex__m_wr_bytes_surface_red_per_sec")]
    pub m_wr_bytes_surface_red_per_sec: Option<Metric<Float>>,
    // #[serde(rename = "tex__m_wr_sectors_global_atom")]
    // pub m_wr_sectors_global_atom: Option<Metric<Float>>,
    // #[serde(rename = "tex__m_wr_sectors_global_atom_pct")]
    // pub m_wr_sectors_global_atom_pct: Option<Metric<Float>>,
    // #[serde(rename = "tex__m_wr_sectors_global_nonatom")]
    // pub m_wr_sectors_global_nonatom: Option<Metric<Float>>,
    // #[serde(rename = "tex__m_wr_sectors_global_nonatom_pct")]
    // pub m_wr_sectors_global_nonatom_pct: Option<Metric<Float>>,
    // #[serde(rename = "tex__m_wr_sectors_global_red")]
    // pub m_wr_sectors_global_red: Option<Metric<Float>>,
    #[serde(rename = "tex__m_wr_sectors_global_red_pct")]
    pub m_wr_sectors_global_red_pc: Option<Metric<Float>>,
    // #[serde(rename = "tex__m_wr_sectors_local_st")]
    // pub m_wr_sectors_local_st: Option<Metric<Float>>,
    #[serde(rename = "tex__m_wr_sectors_local_st_pct")]
    pub m_wr_sectors_local_st_pct: Option<Metric<Float>>,
    // #[serde(rename = "tex__m_wr_sectors_surface_atom")]
    // pub m_wr_sectors_surface_atom: Option<Metric<Float>>,
    // #[serde(rename = "tex__m_wr_sectors_surface_atom_pct")]
    // pub m_wr_sectors_surface_atom_pct: Option<Metric<Float>>,
    // #[serde(rename = "tex__m_wr_sectors_surface_nonatom")]
    // pub m_wr_sectors_surface_nonatom: Option<Metric<Float>>,
    // #[serde(rename = "tex__m_wr_sectors_surface_nonatom_pct")]
    // pub m_wr_sectors_surface_nonatom_pct: Option<Metric<Float>>,
    // #[serde(rename = "tex__m_wr_sectors_surface_red")]
    // pub m_wr_sectors_surface_red: Option<Metric<Float>>,
    #[serde(rename = "tex__m_wr_sectors_surface_red_pct")]
    pub m_wr_sectors_surface_red_pct: Option<Metric<Float>>,
    #[serde(rename = "tex__sol_pct")]
    pub sol_pct: Option<Metric<Float>>,
    #[serde(rename = "tex__t_bytes_miss_global_ld_cached")]
    pub t_bytes_miss_global_ld_cached: Option<Metric<Float>>,
    #[serde(rename = "tex__t_bytes_miss_global_ld_cached_per_sec")]
    pub t_bytes_miss_global_ld_cached_per_sec: Option<Metric<Float>>,
    #[serde(rename = "tex__t_bytes_miss_global_ld_uncached")]
    pub t_bytes_miss_global_ld_uncached: Option<Metric<Float>>,
    #[serde(rename = "tex__t_bytes_miss_global_ld_uncached_per_sec")]
    pub t_bytes_miss_global_ld_uncached_per_sec: Option<Metric<Float>>,
    #[serde(rename = "tex__t_bytes_miss_local_ld_cached")]
    pub t_bytes_miss_local_ld_cached: Option<Metric<Float>>,
    #[serde(rename = "tex__t_bytes_miss_local_ld_cached_per_sec")]
    pub t_bytes_miss_local_ld_cached_per_sec: Option<Metric<Float>>,
    #[serde(rename = "tex__t_bytes_miss_local_ld_uncached")]
    pub t_bytes_miss_local_ld_uncached: Option<Metric<Float>>,
    #[serde(rename = "tex__t_bytes_miss_local_ld_uncached_per_sec")]
    pub t_bytes_miss_local_ld_uncached_per_sec: Option<Metric<Float>>,
    #[serde(rename = "tex__t_bytes_miss_surface_ld")]
    pub t_bytes_miss_surface_ld: Option<Metric<Float>>,
    #[serde(rename = "tex__t_bytes_miss_surface_ld_per_sec")]
    pub t_bytes_miss_surface_ld_per_sec: Option<Metric<Float>>,
    // #[serde(rename = "tex__t_sectors_miss_global_ld_cached")]
    // pub t_sectors_miss_global_ld_cached: Option<Metric<Float>>,
    // #[serde(rename = "tex__t_sectors_miss_global_ld_uncached")]
    // pub t_sectors_miss_global_ld_uncached: Option<Metric<Float>>,
    // #[serde(rename = "tex__t_sectors_miss_local_ld_cached")]
    // pub t_sectors_miss_local_ld_cached: Option<Metric<Float>>,
    // #[serde(rename = "tex__t_sectors_miss_local_ld_uncached")]
    // pub t_sectors_miss_local_ld_uncached: Option<Metric<Float>>,
    // #[serde(rename = "tex__t_sectors_miss_surface_ld")]
    // pub t_sectors_miss_surface_ld: Option<Metric<Float>>,
    #[serde(rename = "tex__tex2sm_tex_nonatomic_active")]
    pub tex2sm_tex_nonatomic_active: Option<Metric<Float>>,
    #[serde(rename = "tex__tex2sm_tex_nonatomic_utilization")]
    pub tex2sm_tex_nonatomic_utilization: Option<Metric<Float>>,
    #[serde(rename = "tex__texin_requests_global_atom")]
    pub texin_requests_global_atom: Option<Metric<Float>>,
    #[serde(rename = "tex__texin_requests_global_atom_per_active_cycle_pct")]
    pub texin_requests_global_atom_per_active_cycle_pct: Option<Metric<Float>>,
    #[serde(rename = "tex__texin_requests_global_atomcas")]
    pub texin_requests_global_atomcas: Option<Metric<Float>>,
    #[serde(rename = "tex__texin_requests_global_atomcas_per_active_cycle_pct")]
    pub texin_requests_global_atomcas_per_active_cycle_pct: Option<Metric<Float>>,
    #[serde(rename = "tex__texin_requests_global_ld_cached")]
    pub texin_requests_global_ld_cached: Option<Metric<Float>>,
    #[serde(rename = "tex__texin_requests_global_ld_cached_per_active_cycle_pct")]
    pub texin_requests_global_ld_cached_per_active_cycle_pct: Option<Metric<Float>>,
    #[serde(rename = "tex__texin_requests_global_ld_uncached")]
    pub texin_requests_global_ld_uncached: Option<Metric<Float>>,
    #[serde(rename = "tex__texin_requests_global_ld_uncached_per_active_cycle_pct")]
    pub texin_requests_global_ld_uncached_per_active_cycle_pct: Option<Metric<Float>>,
    #[serde(rename = "tex__texin_requests_global_red")]
    pub texin_requests_global_red: Option<Metric<Float>>,
    #[serde(rename = "tex__texin_requests_global_red_per_active_cycle_pct")]
    pub texin_requests_global_red_per_active_cycle_pct: Option<Metric<Float>>,
    #[serde(rename = "tex__texin_requests_global_st")]
    pub texin_requests_global_st: Option<Metric<Float>>,
    #[serde(rename = "tex__texin_requests_global_st_per_active_cycle_pct")]
    pub texin_requests_global_st_per_active_cycle_pct: Option<Metric<Float>>,
    #[serde(rename = "tex__texin_requests_local_ld_cached")]
    pub texin_requests_local_ld_cached: Option<Metric<Float>>,
    #[serde(rename = "tex__texin_requests_local_ld_cached_per_active_cycle_pct")]
    pub texin_requests_local_ld_cached_per_active_cycle_pct: Option<Metric<Float>>,
    #[serde(rename = "tex__texin_requests_local_ld_uncached")]
    pub texin_requests_local_ld_uncached: Option<Metric<Float>>,
    #[serde(rename = "tex__texin_requests_local_ld_uncached_per_active_cycle_pct")]
    pub texin_requests_local_ld_uncached_per_active_cycle_pct: Option<Metric<Float>>,
    #[serde(rename = "tex__texin_requests_local_st")]
    pub texin_requests_local_st: Option<Metric<Float>>,
    #[serde(rename = "tex__texin_requests_local_st_per_active_cycle_pct")]
    pub texin_requests_local_st_per_active_cycle_pct: Option<Metric<Float>>,
    #[serde(rename = "tex__texin_requests_surface_atom")]
    pub texin_requests_surface_atom: Option<Metric<Float>>,
    #[serde(rename = "tex__texin_requests_surface_atom_per_active_cycle_pct")]
    pub texin_requests_surface_atom_per_active_cycle_pct: Option<Metric<Float>>,
    #[serde(rename = "tex__texin_requests_surface_atomcas")]
    pub texin_requests_surface_atomcas: Option<Metric<Float>>,
    #[serde(rename = "tex__texin_requests_surface_atomcas_per_active_cycle_pct")]
    pub texin_requests_surface_atomcas_per_active_cycle_pct: Option<Metric<Float>>,
    #[serde(rename = "tex__texin_requests_surface_ld")]
    pub texin_requests_surface_ld: Option<Metric<Float>>,
    #[serde(rename = "tex__texin_requests_surface_ld_per_active_cycle_pct")]
    pub texin_requests_surface_ld_per_active_cycle_pct: Option<Metric<Float>>,
    #[serde(rename = "tex__texin_requests_surface_red")]
    pub texin_requests_surface_red: Option<Metric<Float>>,
    #[serde(rename = "tex__texin_requests_surface_red_per_active_cycle_pct")]
    pub texin_requests_surface_red_per_active_cycle_pct: Option<Metric<Float>>,
    #[serde(rename = "tex__texin_requests_surface_st")]
    pub texin_requests_surface_st: Option<Metric<Float>>,
    #[serde(rename = "tex__texin_requests_surface_st_per_active_cycle_pct")]
    pub texin_requests_surface_st_per_active_cycle_pct: Option<Metric<Float>>,
    /// Number of cycles the tex unit is busy.
    #[serde(rename = "tex__busy_cycles_avg")]
    pub busy_cycles_avg: Option<Metric<Float>>,
    ///Number of cycles the busiest tex unit is busy.
    #[serde(rename = "tex__busy_cycles_max")]
    pub busy_cycles_max: Option<Metric<Float>>,
    /// Percentage of elapsed cycles the tex unit is busy.
    #[serde(rename = "tex__busy_pct_avg")]
    pub busy_pct_avg: Option<Metric<Float>>,
    /// Percentage of elapsed cycles the busiest tex unit is busy.
    #[serde(rename = "tex__busy_pct_max")]
    pub busy_pct_max: Option<Metric<Float>>,
    /// The average count of the number of cycles within a range for a tex unit instance.
    #[serde(rename = "tex__elapsed_cycles_avg")]
    pub elapsed_cycles_avg: Option<Metric<Float>>,
    ///The maximum count of the number of cycles within a range for a tex unit instance.
    #[serde(rename = "tex__elapsed_cycles_max")]
    pub elapsed_cycles_max: Option<Metric<Float>>,
    /// The minimum count of the number of cycles within a range for a tex unit instance.
    #[serde(rename = "tex__elapsed_cycles_min")]
    pub elapsed_cycles_min: Option<Metric<Float>>,
    /// The total count of the number of cycles within a range for a tex unit instance.
    #[serde(rename = "tex__elapsed_cycles_sum")]
    pub elapsed_cycles_sum: Option<Metric<Float>>,
    /// The average frequency of the tex unit(s) in Hz.
    ///
    /// This is calculated as tex__elapsed_cycles_avg divided by gpu__time_duration.
    /// The value will be lower than expected if the measurement range contains GPU context switches.
    #[serde(rename = "tex__frequency")]
    pub frequency: Option<Metric<Float>>,
    /// Percentage of tex requests that hit.
    #[serde(rename = "tex__hitrate_pct")]
    pub hitrate_pct: Option<Metric<Float>>,
    /// Total number of bytes rd by global atom.
    #[serde(rename = "tex__m_rd_bytes_global_atom")]
    pub m_rd_bytes_global_atom: Option<Metric<Float>>,
    /// Total number of bytes per second rd by global_atom.
    #[serde(rename = "tex__m_rd_bytes_global_atom_per_sec")]
    pub m_rd_bytes_global_atom_per_sec: Option<Metric<Float>>,
    /// Number of bytes TEX read from L2 for cached global ld requests.
    #[serde(rename = "tex__m_rd_bytes_miss_global_ld_cached")]
    pub m_rd_bytes_miss_global_ld_cached: Option<Metric<Float>>,
    /// Number of bytes TEX read from L2 for uncached global ld requests.
    #[serde(rename = "tex__m_rd_bytes_miss_global_ld_uncached")]
    pub m_rd_bytes_miss_global_ld_uncached: Option<Metric<Float>>,
    /// Number of bytes TEX read from L2 for cached local ld requests.
    #[serde(rename = "tex__m_rd_bytes_miss_local_ld_cached")]
    pub m_rd_bytes_miss_local_ld_cached: Option<Metric<Float>>,
    /// Number of bytes TEX read from L2 for uncached local ld requests.
    #[serde(rename = "tex__m_rd_bytes_miss_local_ld_uncached")]
    pub m_rd_bytes_miss_local_ld_uncached: Option<Metric<Float>>,
    /// Number of bytes TEX read from L2 for surface ld requests.
    #[serde(rename = "tex__m_rd_bytes_miss_surface_ld")]
    pub m_rd_bytes_miss_surface_ld: Option<Metric<Float>>,
    /// Total number of bytes rd by surface atom.
    #[serde(rename = "tex__m_rd_bytes_surface_atom")]
    pub m_rd_bytes_surface_atom: Option<Metric<Float>>,
    /// Total number of bytes per second rd by surface_atom.
    #[serde(rename = "tex__m_rd_bytes_surface_atom_per_sec")]
    pub m_rd_bytes_surface_atom_per_sec: Option<Metric<Float>>,
    /// Number of sectors TEX read LTS by atomic and reduction operations.
    #[serde(rename = "tex__m_rd_sectors_atom_red")]
    pub m_rd_sectors_atom_red: Option<Metric<Float>>,
    /// Percentage utilization of TEX read LTS by atomic and reduction operations.
    #[serde(rename = "tex__m_rd_sectors_atom_red_pct")]
    pub m_rd_sectors_atom_red_pct: Option<Metric<Float>>,
    /// Total number of sectors rd by global atom.
    #[serde(rename = "tex__m_rd_sectors_gobal_atom")]
    pub m_rd_sectors_gobal_atom: Option<Metric<Float>>,
    /// Percentage of sectors rd by global atom.
    #[serde(rename = "tex__m_rd_sectors_global_atom_pct")]
    pub m_rd_sectors_global_atom_pct: Option<Metric<Float>>,
    /// Number of sectors TEX read from L2 for cached global ld requests.
    #[serde(rename = "tex__m_rd_sectors_miss_global_ld_cached")]
    pub m_rd_sectors_miss_global_ld_cached: Option<Metric<Float>>,
    /// Percentage of TEX cached global ld sectors returned from LTS to the total possible TEX return sectors over the range.
    #[serde(rename = "tex__m_rd_sectors_miss_global_ld_cached_pct")]
    pub m_rd_sectors_miss_global_ld_cached_pct: Option<Metric<Float>>,
    /// Number of sectors TEX read from L2 for uncached global ld requests.
    #[serde(rename = "tex__m_rd_sectors_miss_global_ld_uncached")]
    pub m_rd_sectors_miss_global_ld_uncached: Option<Metric<Float>>,
    /// Percentage of TEX uncached global ld sectors returned from LTS to the total possible TEX return sectors over the range.
    #[serde(rename = "tex__m_rd_sectors_miss_global_ld_uncached_pct")]
    pub m_rd_sectors_miss_global_ld_uncached_pct: Option<Metric<Float>>,
    /// Number of sectors TEX read from L2 for cached local ld requests.
    #[serde(rename = "tex__m_rd_sectors_miss_local_ld_cached")]
    pub m_rd_sectors_miss_local_ld_cached: Option<Metric<Float>>,
    /// Percentage of TEX cached local ld sectors returned from LTS to the total possible TEX return sectors over the range.
    #[serde(rename = "tex__m_rd_sectors_miss_local_ld_cached_pct")]
    pub m_rd_sectors_miss_local_ld_cached_pct: Option<Metric<Float>>,
    /// Number of sectors TEX read from L2 for uncached local ld requests.
    #[serde(rename = "tex__m_rd_sectors_miss_local_ld_uncached")]
    pub m_rd_sectors_miss_local_ld_uncached: Option<Metric<Float>>,
    ///  Percentage of TEX uncached local ld sectors returned from LTS to the total possible TEX return sectors over the range.
    #[serde(rename = "tex__m_rd_sectors_miss_local_ld_uncached_pct")]
    pub m_rd_sectors_miss_local_ld_uncached_pct: Option<Metric<Float>>,
    /// Number of sectors TEX read from L2 for surface ld requests.
    #[serde(rename = "tex__m_rd_sectors_miss_surface_ld")]
    pub m_rd_sectors_miss_surface_ld: Option<Metric<Float>>,
    /// Percentage of TEX surface ld sectors returned from LTS to the total possible TEX return sectors over the range.
    #[serde(rename = "tex__m_rd_sectors_miss_surface_ld_pct")]
    pub m_rd_sectors_miss_surface_ld_pct: Option<Metric<Float>>,
    /// Total number of sectors rd by surface atom.
    #[serde(rename = "tex__m_rd_sectors_surface_atom")]
    pub m_rd_sectors_surface_atom: Option<Metric<Float>>,
    /// Percentage of sectors rd by surface atom.
    // #[serde(rename = "tex__m_rd_sectors_surface_atom_pct")]
    // pub m_rd_sectors_surface_atom_pct: Option<Metric<Float>>,
    // tex__m_wr_bytes_global_atom                                                 Total number of bytes wr by global atom.
    // tex__m_wr_bytes_global_atom_per_sec                                         Total number of bytes per second wr by global_atom.
    // tex__m_wr_bytes_global_nonatom                                              Total number of bytes wr by global nonatom.
    // tex__m_wr_bytes_global_nonatom_per_sec                                      Total number of bytes per second wr by global_nonatom.
    // tex__m_wr_bytes_global_red                                                  Total number of bytes wr by global red.
    // tex__m_wr_bytes_global_red_per_sec                                          Total number of bytes per second wr by global_red.
    // tex__m_wr_bytes_local_st                                                    Total number of bytes wr by local st.
    // tex__m_wr_bytes_local_st_per_sec                                            Total number of bytes per second wr by local_st.
    // tex__m_wr_bytes_surface_atom                                                Total number of bytes wr by surface atom.
    // tex__m_wr_bytes_surface_atom_per_sec                                        Total number of bytes per second wr by surface_atom.
    // tex__m_wr_bytes_surface_nonatom                                             Total number of bytes wr by surface nonatom.
    // tex__m_wr_bytes_surface_nonatom_per_sec                                     Total number of bytes per second wr by surface_nonatom.
    // tex__m_wr_bytes_surface_red                                                 Total number of bytes wr by surface red.
    // tex__m_wr_bytes_surface_red_per_sec                                         Total number of bytes per second wr by surface_red.
    /// Number of sectors TEX written LTS by atomic and reduction operations.
    #[serde(rename = "tex__m_wr_sectors_atom_red")]
    pub m_wr_sectors_atom_red: Option<Metric<Float>>,
    /// Percentage utilization of TEX write LTS by atomic and reduction operations.
    #[serde(rename = "tex__m_wr_sectors_atom_red_pct")]
    pub m_wr_sectors_atom_red_pct: Option<Metric<Float>>,
    /// Total number of sectors wr by global atom.
    #[serde(rename = "tex__m_wr_sectors_global_atom")]
    pub m_wr_sectors_global_atom: Option<Metric<Float>>,
    /// Percentage of sectors wr by global atom.
    #[serde(rename = "tex__m_wr_sectors_global_atom_pct")]
    pub m_wr_sectors_global_atom_pct: Option<Metric<Float>>,
    /// Total number of sectors wr by global nonatom.
    #[serde(rename = "tex__m_wr_sectors_global_nonatom")]
    pub m_wr_sectors_global_nonatom: Option<Metric<Float>>,
    /// Percentage of sectors wr by global nonatom.
    #[serde(rename = "tex__m_wr_sectors_global_nonatom_pct")]
    pub m_wr_sectors_global_nonatom_pct: Option<Metric<Float>>,
    /// Total number of sectors wr by global red.
    #[serde(rename = "tex__m_wr_sectors_global_red")]
    pub m_wr_sectors_global_red: Option<Metric<Float>>,
    /// Percentage of sectors wr by global red.
    #[serde(rename = "tex__m_wr_sectors_global_red_pct")]
    pub m_wr_sectors_global_red_pct: Option<Metric<Float>>,
    /// Total number of sectors wr by local st.
    #[serde(rename = "tex__m_wr_sectors_local_st")]
    pub m_wr_sectors_local_st: Option<Metric<Float>>,
    /// Percentage of sectors wr by local st.
    #[serde(rename = "tex__m_wr_sectors_local_st_pct")]
    pub m_rd_sectors_surface_atom_pct: Option<Metric<Float>>,
    /// Total number of sectors wr by surface atom.
    #[serde(rename = "tex__m_wr_sectors_surface_atom")]
    pub m_wr_sectors_surface_atom: Option<Metric<Float>>,
    /// Percentage of sectors wr by surface atom.
    #[serde(rename = "tex__m_wr_sectors_surface_atom_pct")]
    pub m_wr_sectors_surface_atom_pct: Option<Metric<Float>>,
    /// Total number of sectors wr by surface nonatom.
    #[serde(rename = "tex__m_wr_sectors_surface_nonatom")]
    pub m_wr_sectors_surface_nonatom: Option<Metric<Float>>,
    /// Percentage of sectors wr by surface nonatom.
    #[serde(rename = "tex__m_wr_sectors_surface_nonatom_pct")]
    pub m_wr_sectors_surface_nonatom_pct: Option<Metric<Float>>,
    /// Total number of sectors wr by surface red.
    #[serde(rename = "tex__m_wr_sectors_surface_red")]
    pub m_wr_sectors_surface_red: Option<Metric<Float>>,
    // tex__m_wr_sectors_surface_red_pct                                           Percentage of sectors wr by surface red.
    // tex__read_bytes                                                             Texture memory read in bytes.
    // tex__sm_utilization_pct                                                     Percentage utilization of the TEX to SM interface.
    // tex__sol_pct                                                                SOL percentage of texture unit.
    // tex__t_bytes_hit_global_ld_cached                                           Number of TEX cache sector hit in bytes from cached global ld requests.
    // tex__t_bytes_hit_global_ld_cached_per_sec                                   Throughput of TEX cache sector hit in bytes per second from cached global ld requests.
    // tex__t_bytes_hit_local_ld_cached                                            Number of TEX cache sector hit in bytes from cached local ld requests.
    // tex__t_bytes_hit_local_ld_cached_per_sec                                    Throughput of TEX cache sector hit in bytes per second from cached local ld requests.
    // tex__t_bytes_miss_global_ld_cached                                          Number of TEX cache sector miss in bytes from cached global ld requests.
    // tex__t_bytes_miss_global_ld_cached_per_sec                                  Throughput of TEX cache sector miss in bytes per second from cached global ld requests.
    // tex__t_bytes_miss_global_ld_uncached                                        Number of TEX cache sector miss in bytes from uncached global ld requests.
    // tex__t_bytes_miss_global_ld_uncached_per_sec                                Throughput of TEX cache sector miss in bytes per second from uncached global ld requests.
    // tex__t_bytes_miss_local_ld_cached                                           Number of TEX cache sector miss in bytes from cached local ld requests.
    // tex__t_bytes_miss_local_ld_cached_per_sec                                   Throughput of TEX cache sector miss in bytes per second from cached local ld requests.
    // tex__t_bytes_miss_local_ld_uncached                                         Number of TEX cache sector miss in bytes from uncached local ld requests.
    // tex__t_bytes_miss_local_ld_uncached_per_sec                                 Throughput of TEX cache sector miss in bytes per second from uncached local ld requests.
    // tex__t_bytes_miss_surface_ld                                                Number of TEX cache sector miss in bytes from surface ld requests.
    // tex__t_bytes_miss_surface_ld_per_sec                                        Throughput of TEX cache sector miss in bytes per second from surface ld requests.
    /// Number of TEX cache sector hit from cached global ld requests.
    #[serde(rename = "tex__t_sectors_hit_global_ld_cached")]
    pub t_sectors_hit_global_ld_cached: Option<Metric<Float>>,
    /// Number of TEX cache sector hit from cached local ld requests.
    #[serde(rename = "tex__t_sectors_hit_local_ld_cached")]
    pub t_sectors_hit_local_ld_cached: Option<Metric<Float>>,
    /// Number of TEX cache sector miss from cached global ld requests.
    #[serde(rename = "tex__t_sectors_miss_global_ld_cached")]
    pub t_sectors_miss_global_ld_cached: Option<Metric<Float>>,
    /// Number of TEX cache sector miss from uncached global ld requests.
    #[serde(rename = "tex__t_sectors_miss_global_ld_uncached")]
    pub t_sectors_miss_global_ld_uncached: Option<Metric<Float>>,
    /// Number of TEX cache sector miss from cached local ld requests.
    #[serde(rename = "tex__t_sectors_miss_local_ld_cached")]
    pub t_sectors_miss_local_ld_cached: Option<Metric<Float>>,
    /// Number of TEX cache sector miss from uncached local ld requests.
    #[serde(rename = "tex__t_sectors_miss_local_ld_uncached")]
    pub t_sectors_miss_local_ld_uncached: Option<Metric<Float>>,
    /// Number of TEX cache sector miss from surface ld requests.
    #[serde(rename = "tex__t_sectors_miss_surface_ld")]
    pub t_sectors_miss_surface_ld: Option<Metric<Float>>,
    // tex__tex2sm_tex_nonatomic_active                                            Number of cycles the TEX to SM interface is active for nonatomic operations.
    // tex__tex2sm_tex_nonatomic_utilization                                       Percentage of cycles the TEX to SM interface is active for nonatomic operations.
    // tex__texel_queries                                                          The total number of texels queried.
    // tex__texin_requests_global_atom                                             Number of global atom requests sent to TEX.
    // tex__texin_requests_global_atom_per_active_cycle_pct                        Percentage utilization of TEX request interface for global atom.
    // tex__texin_requests_global_atom_per_elapsed_cycle_pct                       Percentage utilization of TEX request interface for global atom.
    // tex__texin_requests_global_atomcas                                          Number of global atomcas requests sent to TEX.
    // tex__texin_requests_global_atomcas_per_active_cycle_pct                     Percentage utilization of TEX request interface for global atomcas.
    // tex__texin_requests_global_atomcas_per_elapsed_cycle_pct                    Percentage utilization of TEX request interface for global atomcas.
    // tex__texin_requests_global_ld_cached                                        Number of cached global ld requests sent to TEX.
    // tex__texin_requests_global_ld_cached_per_active_cycle_pct                   Percentage utilization of TEX request interface for cached global ld.
    // tex__texin_requests_global_ld_cached_per_elapsed_cycle_pct                  Percentage utilization of TEX request interface for cached global ld.
    // tex__texin_requests_global_ld_uncached                                      Number of uncached global ld requests sent to TEX.
    // tex__texin_requests_global_ld_uncached_per_active_cycle_pct                 Percentage utilization of TEX request interface for uncached global ld.
    // tex__texin_requests_global_ld_uncached_per_elapsed_cycle_pct                Percentage utilization of TEX request interface for uncached global ld.
    // tex__texin_requests_global_red                                              Number of global red requests sent to TEX.
    // tex__texin_requests_global_red_per_active_cycle_pct                         Percentage utilization of TEX request interface for global red.
    // tex__texin_requests_global_red_per_elapsed_cycle_pct                        Percentage utilization of TEX request interface for global red.
    // tex__texin_requests_global_st                                               Number of global st requests sent to TEX.
    // tex__texin_requests_global_st_per_active_cycle_pct                          Percentage utilization of TEX request interface for global st.
    // tex__texin_requests_global_st_per_elapsed_cycle_pct                         Percentage utilization of TEX request interface for global st.
    // tex__texin_requests_local_ld_cached                                         Number of cached local ld requests sent to TEX.
    // tex__texin_requests_local_ld_cached_per_active_cycle_pct                    Percentage utilization of TEX request interface for cached local ld.
    // tex__texin_requests_local_ld_cached_per_elapsed_cycle_pct                   Percentage utilization of TEX request interface for cached local ld.
    // tex__texin_requests_local_ld_uncached                                       Number of uncached local ld requests sent to TEX.
    // tex__texin_requests_local_ld_uncached_per_active_cycle_pct                  Percentage utilization of TEX request interface for uncached local ld.
    // tex__texin_requests_local_ld_uncached_per_elapsed_cycle_pct                 Percentage utilization of TEX request interface for uncached local ld.
    // tex__texin_requests_local_st                                                Number of local st requests sent to TEX.
    // tex__texin_requests_local_st_per_active_cycle_pct                           Percentage utilization of TEX request interface for local st.
    // tex__texin_requests_local_st_per_elapsed_cycle_pct                          Percentage utilization of TEX request interface for local st.
    // tex__texin_requests_surface_atom                                            Number of surface atom requests sent to TEX.
    // tex__texin_requests_surface_atom_per_active_cycle_pct                       Percentage utilization of TEX request interface for surface atom.
    // tex__texin_requests_surface_atom_per_elapsed_cycle_pct                      Percentage utilization of TEX request interface for surface atom.
    // tex__texin_requests_surface_atomcas                                         Number of surface atomcas requests sent to TEX.
    // tex__texin_requests_surface_atomcas_per_active_cycle_pct                    Percentage utilization of TEX request interface for surface atomcas.
    // tex__texin_requests_surface_atomcas_per_elapsed_cycle_pct                   Percentage utilization of TEX request interface for surface atomcas.
    // tex__texin_requests_surface_ld                                              Number of surface ld requests sent to TEX.
    // tex__texin_requests_surface_ld_per_active_cycle_pct                         Percentage utilization of TEX request interface for surface ld.
    // tex__texin_requests_surface_ld_per_elapsed_cycle_pct                        Percentage utilization of TEX request interface for surface ld.
    // tex__texin_requests_surface_red                                             Number of surface red requests sent to TEX.
    // tex__texin_requests_surface_red_per_active_cycle_pct                        Percentage utilization of TEX request interface for surface red.
    // tex__texin_requests_surface_red_per_elapsed_cycle_pct                       Percentage utilization of TEX request interface for surface red.
    // tex__texin_requests_surface_st                                              Number of surface st requests sent to TEX.
    // tex__texin_requests_surface_st_per_active_cycle_pct                         Percentage utilization of TEX request interface for surface st.
    // tex__texin_requests_surface_st_per_elapsed_cycle_pct                        Percentage utilization of TEX request interface for surface st.
    // tex__texin_tsl2_stall_cycles                                                Number of cycles the TEX TEXIN stage is stalled awaiting data from the texture header or sampler L2
    //                                                                             cache (TSL2).
    // tex__texin_tsl2_stall_cycles_per_elapsed_cycle                              Number of cycles the TEX TEXIN stage is stalled awaiting data from the texture header or sampler L2
    //                                                                             cache (TSL2) per TEX elapsed cycle.
    // tex__texin_tsl2_stall_cycles_per_elapsed_cycle_pct                          Number of cycles the TEX TEXIN stage is stalled awaiting data from the texture header or sampler L2
    //                                                                             cache (TSL2) per TEX elapsed cycle, as a percentage.
}
