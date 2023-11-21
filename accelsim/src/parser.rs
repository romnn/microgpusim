use super::read::BufReadLine;
use super::stats::Stats;
use color_eyre::eyre::{self, WrapErr};
use indexmap::IndexSet;
use itertools::Itertools;
use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Options {
    /// Aggregate stats for each named kernel
    pub per_kernel: bool,
    /// Aggregate stats for each individual named kernel instance
    pub kernel_instance: bool,
    /// Enable strict parsing
    pub strict: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            per_kernel: true,
            kernel_instance: true,
            strict: true,
        }
    }
}

macro_rules! stat {
    ($name:expr, $kind:expr, $regex:expr) => {
        ($name.to_string(), ($kind, Regex::new(&*$regex).unwrap()))
    };
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum StatKind {
    Aggregate,
    Abs,
    Rate,
}

static GPGPUSIM_BUILD_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r".*GPGPU-Sim.*\[build\s+(.*)\].*").unwrap());
static ACCELSIM_BUILD_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r".*Accel-Sim.*\[build\s+(.*)\].*").unwrap());

/// Does a quick 100-line pass to get the GPGPU-Sim Version number.
///
/// Assumes the reader is seeked to the beginning of the file.
fn get_version(mut f: impl BufReadLine + std::io::Seek) -> Option<String> {
    static MAX_LINES: usize = 100;
    let mut buffer = String::new();
    let mut lines = 0;
    while let Some(Ok(line)) = f.read_line(&mut buffer) {
        if lines >= MAX_LINES {
            break;
        }

        if let Some(build) = GPGPUSIM_BUILD_REGEX
            .captures(line)
            .and_then(|c| c.get(1))
            .map(|m| m.as_str().trim().to_string())
        {
            return Some(build);
        }

        if let Some(build) = ACCELSIM_BUILD_REGEX
            .captures(line)
            .and_then(|c| c.get(1))
            .map(|m| m.as_str().trim().to_string())
        {
            return Some(build);
        }

        lines += 1;
    }
    None
}

static EXIT_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"GPGPU-Sim: \*\*\* exit detected \*\*\*").unwrap());

/// Performs a quick 10000-line reverse pass,
/// to make sure the simualtion thread finished.
///
/// Assumes the reader is seeked to the end of the file and
/// reading lines in reverse.
fn check_finished(mut f: impl BufReadLine + std::io::Seek) -> bool {
    static MAX_LINES: usize = 10_000;
    let mut buffer = String::new();
    let mut lines = 0;

    while let Some(Ok(line)) = f.read_line(&mut buffer) {
        if lines >= MAX_LINES {
            break;
        }
        if EXIT_REGEX.captures(line).is_some() {
            return true;
        }

        lines += 1;
    }
    false
}

static LAST_KERNEL_BREAK_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"GPGPU-Sim: \*\* break due to reaching the maximum cycles \(or instructions\) \*\*")
        .unwrap()
});

static KERNEL_NAME_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"kernel_name\s+=\s+(.*)").unwrap());

/// Parses accelsim log and extracts statistics.
#[allow(clippy::too_many_lines)]
pub fn parse_stats(
    reader: impl std::io::Read + std::io::Seek,
    options: &Options,
) -> eyre::Result<Stats> {
    let mut reverse_reader = rev_buf_reader::RevBufReader::new(reader);
    let finished = check_finished(&mut reverse_reader);
    let mut reader = reverse_reader.into_inner();
    reader.rewind()?;

    let mut reader = std::io::BufReader::new(reader);
    let _version = get_version(&mut reader);

    // println!("GPGPU-sim version: {:?}", &version);

    let general_stats = vec![
        stat!(
            "gpu_total_instructions",
            StatKind::Aggregate,
            r"gpu_tot_sim_insn\s*=\s*(.*)"
        ),
        stat!(
            "gpgpu_simulation_time_sec",
            StatKind::Aggregate,
            r"gpgpu_simulation_time\s*=.*\(([0-9]+) sec\).*"
        ),
        stat!(
            "gpu_tot_sim_cycle",
            StatKind::Aggregate,
            r"gpu_tot_sim_cycle\s*=\s*(.*)"
        ),
        stat!(
            "warp_instruction_count",
            StatKind::Aggregate,
            r"gpgpu_n_tot_w_icount\s*=\s*(.*)"
        ),
        stat!(
            "kernel_launch_uid",
            StatKind::Aggregate,
            r"kernel_launch_uid\s*=\s*(.*)"
        ),
        stat!(
            "num_issued_blocks",
            StatKind::Aggregate,
            r"gpu_tot_issued_cta\s*=\s*(.*)"
        ),
        stat!(
            "occupancy",
            StatKind::Aggregate,
            r"gpu_occupancy\s*=\s*(.*)%"
        ),
        stat!(
            "total_occupancy",
            StatKind::Aggregate,
            r"gpu_tot_occupancy\s*=\s*(.*)%"
        ),
        // stat!(
        //     "max_total_param_size",
        //     StatKind::Aggregate,
        //     r"max_total_param_size\s*=\s*(.*)"
        // ),
    ];

    let mem_space = vec![
        ("GLOBAL_ACC_R", "global_read"),
        ("LOCAL_ACC_R", "local_read"),
        ("CONST_ACC_R", "constant_read"),
        ("TEXTURE_ACC_R", "texture_read"),
        ("GLOBAL_ACC_W", "global_write"),
        ("LOCAL_ACC_W", "local_write"),
        ("L1_WRBK_ACC", "l1_writeback"),
        ("L2_WRBK_ACC", "l2_writeback"),
        ("INST_ACC_R", "inst_read"),
        ("L1_WR_ALLOC_R", "l1_write_alloc_read"),
        ("L2_WR_ALLOC_R", "l2_write_alloc_read"),
    ];
    let outcome = [
        ("HIT", "hit"),
        ("HIT_RESERVED", "hit_reserved"),
        ("MISS", "miss"),
        ("RESERVATION_FAIL", "reservation_fail"),
        ("SECTOR_MISS", "sector_miss"),
        ("MSHR_HIT", "mshr_hit"),
    ];

    let mut l2_cache_stats: Vec<_> = mem_space
        .iter()
        .cartesian_product(outcome.iter())
        .map(|((space, _space_name), (outcome, _outcome_name))| {
            stat!(
                format!("l2_cache_{space}_{outcome}"),
                StatKind::Aggregate,
                [
                    r"\s+L2_cache_stats_breakdown\[",
                    space,
                    r"\]\[",
                    outcome,
                    r"\]\s*=\s*(.*)"
                ]
                .join("")
            )
        })
        .collect();

    l2_cache_stats.extend(mem_space.iter().map(|(space, space_name)| {
        stat!(
            format!("l2_cache_{space_name}_total"),
            StatKind::Aggregate,
            format!(r"\s+L2_cache_stats_breakdown\[{space}\]\[TOTAL_ACCESS\]\s*=\s*(.*)")
        )
    }));

    l2_cache_stats.extend([
        // stat!(
        //     "l2_cache_global_read_total",
        //     StatKind::Aggregate,
        //     r"\s+L2_cache_stats_breakdown\[GLOBAL_ACC_R\]\[TOTAL_ACCESS\]\s*=\s*(.*)"
        // ),
        // stat!(
        //     "l2_cache_global_write_total",
        //     StatKind::Aggregate,
        //     r"\s+L2_cache_stats_breakdown\[GLOBAL_ACC_W\]\[TOTAL_ACCESS\]\s*=\s*(.*)"
        // ),
        // stat!(
        //     "l2_cache_inst_read_total",
        //     StatKind::Aggregate,
        //     r"\s+L2_cache_stats_breakdown\[INST_ACC_W\]\[TOTAL_ACCESS\]\s*=\s*(.*)"
        // ),
        stat!(
            "l2_cache_total_accesses",
            StatKind::Aggregate,
            r"\s+L2_total_cache_accesses\s*=\s*(.*)"
        ),
        stat!(
            "l2_cache_total_misses",
            StatKind::Aggregate,
            r"\s+L2_total_cache_misses\s*=\s*(.*)"
        ),
        stat!(
            "l2_cache_total_reservation_fails",
            StatKind::Aggregate,
            r"\s+L2_total_cache_reservation_fails\s*=\s*(.*)"
        ),
    ]);

    let mut l1_data_cache_stats: Vec<_> = mem_space
        .iter()
        .cartesian_product(outcome.iter())
        .map(|((space, _space_name), (outcome, _outcome_name))| {
            stat!(
                format!("l1_data_cache_{space}_{outcome}"),
                StatKind::Aggregate,
                [
                    r"\s+Total_core_cache_stats_breakdown\[",
                    space,
                    r"\]\[",
                    outcome,
                    r"\]\s*=\s*(.*)"
                ]
                .join("")
            )
        })
        .collect();

    l1_data_cache_stats.extend(mem_space.iter().map(|(space, space_name)| {
        stat!(
            format!("l1_data_cache_{space_name}_total"),
            StatKind::Aggregate,
            format!(r"\s+Total_core_cache_stats_breakdown\[{space}\]\[TOTAL_ACCESS\]\s*=\s*(.*)")
        )
    }));

    l1_data_cache_stats.extend([
        stat!(
            "l1_data_cache_total_accesses",
            StatKind::Aggregate,
            r"L1D_total_cache_accesses\s*=\s*(.*)"
        ),
        stat!(
            "l1_data_cache_total_misses",
            StatKind::Aggregate,
            r"L1D_total_cache_misses\s*=\s*(.*)"
        ),
        stat!(
            "l1_data_cache_total_pending_hits",
            StatKind::Aggregate,
            r"L1D_total_cache_pending_hits\s*=\s*(.*)"
        ),
        stat!(
            "l1_data_cache_total_reservation_fails",
            StatKind::Aggregate,
            r"L1D_total_cache_reservation_fails\s*=\s*(.*)"
        ),
        stat!(
            "l1_data_cache_data_port_utilization",
            StatKind::Aggregate,
            r"L1D_cache_data_port_util\s*=\s*(.*)"
        ),
        stat!(
            "l1_data_cache_fill_port_utilization",
            StatKind::Aggregate,
            r"L1D_cache_fill_port_util\s*=\s*(.*)"
        ),
    ]);

    let dram_stats = vec![
        stat!(
            "total_dram_reads",
            StatKind::Aggregate,
            r"total dram reads\s*=\s*(.*)"
        ),
        stat!(
            "total_dram_writes",
            StatKind::Aggregate,
            r"total dram writes\s*=\s*(.*)"
        ),
    ];

    let inst_stats = vec![
        stat!(
            "num_load_inst",
            StatKind::Aggregate,
            r"gpgpu_n_load_insn\s*=\s*(.*)"
        ),
        stat!(
            "num_store_inst",
            StatKind::Aggregate,
            r"gpgpu_n_store_insn\s*=\s*(.*)"
        ),
        stat!(
            "num_shared_mem_inst",
            StatKind::Aggregate,
            r"gpgpu_n_shmem_insn\s*=\s*(.*)"
        ),
        stat!(
            "num_star_inst",
            StatKind::Aggregate,
            r"gpgpu_n_sstarr_insn\s*=\s*(.*)"
        ),
        stat!(
            "num_tex_inst",
            StatKind::Aggregate,
            r"gpgpu_n_tex_insn\s*=\s*(.*)"
        ),
        stat!(
            "num_const_mem_inst",
            StatKind::Aggregate,
            r"gpgpu_n_const_mem_insn\s*=\s*(.*)"
        ),
        stat!(
            "num_param_mem_inst",
            StatKind::Aggregate,
            r"gpgpu_n_param_mem_insn\s*=\s*(.*)"
        ),
    ];

    let stall_stats = vec![
        stat!(
            "num_shared_mem_stalls",
            StatKind::Aggregate,
            r"gpgpu_n_stall_shd_mem\s*=\s*(.*)"
        ),
        stat!(
            "num_shared_mem_const_resource_stalls",
            StatKind::Aggregate,
            r"gpgpu_stall_shd_mem[c_mem][resource_stall]\s*=\s*(.*)"
        ),
        stat!(
            "num_shared_mem_bank_conflict_stalls",
            StatKind::Aggregate,
            r"gpgpu_stall_shd_mem[s_mem][bk_conf]\s*=\s*(.*)"
        ),
        stat!(
            "num_shared_mem_global_resource_stalls",
            StatKind::Aggregate,
            r"gpgpu_stall_shd_mem[gl_mem][resource_stall]\s*=\s*(.*)"
        ),
        stat!(
            "num_shared_mem_global_coal_stalls",
            StatKind::Aggregate,
            r"gpgpu_stall_shd_mem[gl_mem][coal_stall]\s*=\s*(.*)"
        ),
        stat!(
            "num_shared_mem_global_data_port_stalls",
            StatKind::Aggregate,
            r"gpgpu_stall_shd_mem[gl_mem][data_port_stall]\s*=\s*(.*)"
        ),
        stat!(
            "num_register_set_bank_conflict_stalls",
            StatKind::Aggregate,
            r"gpu_reg_bank_conflict_stalls\s*=\s*(.*)"
        ),
        stat!(
            "num_dram_full_stalls",
            StatKind::Aggregate,
            r"gpu_stall_dramfull\s*=\s*(.*)"
        ),
        stat!(
            "num_interconn_to_shared_mem_stalls",
            StatKind::Aggregate,
            r"gpu_stall_icnt2sh\s*=\s*(.*)"
        ),
        stat!(
            "num_shared_mem_bank_conflicts",
            StatKind::Aggregate,
            r"gpgpu_n_shmem_bkconflict\s*=\s*(.*)"
        ),
        stat!(
            "num_cache_bank_conflicts",
            StatKind::Aggregate,
            r"gpgpu_n_cache_bkconflict\s*=\s*(.*)"
        ),
        stat!(
            "num_intra_warp_mshr_merge",
            StatKind::Aggregate,
            r"gpgpu_n_intrawarp_mshr_merge\s*=\s*(.*)"
        ),
        stat!(
            "num_const_mem_port_conflict",
            StatKind::Aggregate,
            r"gpgpu_n_cmem_portconflict\s*=\s*(.*)"
        ),
    ];

    let mem_stats = vec![
        stat!(
            "num_local_mem_read",
            StatKind::Aggregate,
            r"gpgpu_n_mem_read_local\s*=\s*(.*)"
        ),
        stat!(
            "num_local_mem_write",
            StatKind::Aggregate,
            r"gpgpu_n_mem_write_local\s*=\s*(.*)"
        ),
        stat!(
            "num_global_mem_read",
            StatKind::Aggregate,
            r"gpgpu_n_mem_read_global\s*=\s*(.*)"
        ),
        stat!(
            "num_global_mem_write",
            StatKind::Aggregate,
            r"gpgpu_n_mem_write_global\s*=\s*(.*)"
        ),
        stat!(
            "num_global_mem_write",
            StatKind::Aggregate,
            r"gpgpu_n_mem_write_global\s*=\s*(.*)"
        ),
        stat!(
            "num_tex_mem_total_accesses",
            StatKind::Aggregate,
            r"gpgpu_n_mem_texture\s*=\s*(.*)"
        ),
        stat!(
            "num_const_mem_total_accesses",
            StatKind::Aggregate,
            r"gpgpu_n_mem_const\s*=\s*(.*)"
        ),
    ];

    let l1_inst_cache_stats = vec![
        stat!(
            "l1_inst_cache_total_accesses",
            StatKind::Aggregate,
            r"L1I_total_cache_accesses\s*=\s*(.*)"
        ),
        stat!(
            "l1_inst_cache_total_misses",
            StatKind::Aggregate,
            r"L1I_total_cache_misses\s*=\s*(.*)"
        ),
        stat!(
            "l1_inst_cache_total_miss_rate",
            StatKind::Aggregate,
            r"L1I_total_cache_miss_rate\s*=\s*(.*)"
        ),
        stat!(
            "l1_inst_cache_total_pending_hits",
            StatKind::Aggregate,
            r"L1I_total_cache_pending_hits\s*=\s*(.*)"
        ),
        stat!(
            "l1_inst_cache_total_reservation_fails",
            StatKind::Aggregate,
            r"L1I_total_cache_reservation_fails\s*=\s*(.*)"
        ),
    ];

    let l1_const_cache_stats = vec![
        stat!(
            "l1_const_cache_total_accesses",
            StatKind::Aggregate,
            r"L1C_total_cache_accesses\s*=\s*(.*)"
        ),
        stat!(
            "l1_const_cache_total_misses",
            StatKind::Aggregate,
            r"L1C_total_cache_misses\s*=\s*(.*)"
        ),
        stat!(
            "l1_const_cache_total_pending_hits",
            StatKind::Aggregate,
            r"L1C_total_cache_pending_hits\s*=\s*(.*)"
        ),
        stat!(
            "l1_const_cache_total_reservation_fails",
            StatKind::Aggregate,
            r"L1C_total_cache_reservation_fails\s*=\s*(.*)"
        ),
    ];

    let l1_tex_cache_stats = vec![
        stat!(
            "l1_tex_cache_total_accesses",
            StatKind::Aggregate,
            r"L1T_total_cache_accesses\s*=\s*(.*)"
        ),
        stat!(
            "l1_tex_cache_total_misses",
            StatKind::Aggregate,
            r"L1T_total_cache_misses\s*=\s*(.*)"
        ),
        stat!(
            "l1_tex_cache_total_pending_hits",
            StatKind::Aggregate,
            r"L1T_total_cache_pending_hits\s*=\s*(.*)"
        ),
        stat!(
            "l1_tex_cache_total_reservation_fails",
            StatKind::Aggregate,
            r"L1T_total_cache_reservation_fails\s*=\s*(.*)"
        ),
    ];

    let aggregate_stats = vec![
        general_stats,
        mem_stats,
        l2_cache_stats,
        l1_data_cache_stats,
        l1_inst_cache_stats,
        l1_const_cache_stats,
        l1_tex_cache_stats,
        dram_stats,
        inst_stats,
        stall_stats,
    ]
    .concat();

    // These stats are reset each kernel and should not be diff'd
    // They cannot be used is only collecting the final_kernel stats
    let abs_stats = vec![
        stat!("gpu_ipc", StatKind::Abs, r"gpu_ipc\s*=\s*(.*)"),
        stat!("gpu_occupancy", StatKind::Abs, r"gpu_occupancy\s*=\s*(.*)%"),
        stat!(
            "l2_bandwidth_gbps",
            StatKind::Abs,
            r"L2_BW\s*=\s*(.*)+GB/Sec"
        ),
    ];

    // These stats are rates that aggregate - but cannot be diff'd
    // Only valid as a snapshot and most useful for the final kernel launch
    let rate_stats = vec![
        stat!(
            "gpgpu_simulation_rate",
            StatKind::Rate,
            r"gpgpu_simulation_rate\s+=\s+(.*)\s+\(inst/sec\)"
        ),
        stat!(
            "gpgpu_simulation_rate",
            StatKind::Rate,
            r"gpgpu_simulation_rate\s+=\s+(.*)\s+\(cycle/sec\)"
        ),
        stat!(
            "gpgpu_silicon_slowdown",
            StatKind::Rate,
            r"gpgpu_silicon_slowdown\s*=\s*(.*)x"
        ),
        stat!("gpu_tot_ipc", StatKind::Rate, r"gpu_tot_ipc\s*=\s*(.*)"),
    ];

    let stats: HashMap<String, (StatKind, Regex)> = aggregate_stats
        .into_iter()
        .chain(abs_stats)
        .chain(rate_stats)
        .collect();

    if !finished {
        if options.strict {
            eyre::bail!("invalid: termination message from GPGPU-Sim not found");
        }
        eprintln!("invalid: termination message from GPGPU-Sim not found");
    }

    let mut all_named_kernels: IndexSet<(String, usize)> = IndexSet::new();

    let mut stat_found: IndexSet<String> = IndexSet::new();

    let mut stat_map = Stats::default();

    if options.per_kernel {
        let mut current_kernel = (String::new(), 0);
        let mut last_kernel = (String::new(), 0);
        let mut raw_last: HashMap<String, f64> = HashMap::new();
        let mut running_kcount: HashMap<String, usize> = HashMap::new();

        let mut reader = std::io::BufReader::new(reader);
        let mut buffer = String::new();
        let mut kernel_launch_id = 0;

        while let Some(Ok(line)) = reader.read_line(&mut buffer) {
            // was simulation aborted due to too many instructions?
            // then ignore the last kernel launch, as it is no complete
            // (only appies if we are doing kernel-by-kernel stats)

            if LAST_KERNEL_BREAK_REGEX.captures(line).is_some() {
                log::warn!("found max instructions - ignoring last kernel");
                // remove incomplete kernel stats
                for stat_name in stats.keys() {
                    stat_map.remove(&(
                        current_kernel.0.to_string(),
                        current_kernel.1,
                        stat_name.to_string(),
                    ));
                    assert_eq!(
                        current_kernel.1,
                        running_kcount.get(&current_kernel.0).copied().unwrap_or(0),
                    );
                }
            }

            if let Some(kernel_name) = KERNEL_NAME_REGEX.captures(line).and_then(|c| c.get(1)) {
                let kernel_name = kernel_name.as_str().trim().to_string();
                last_kernel = current_kernel;

                current_kernel = (
                    kernel_name.clone(),
                    kernel_launch_id,
                    // running_kcount.get(&kernel_name).copied().unwrap_or(0),
                );

                if options.kernel_instance {
                    running_kcount
                        .entry(kernel_name.clone())
                        .and_modify(|count| *count += 1)
                        .or_insert(0);
                    kernel_launch_id += 1;
                }

                all_named_kernels.insert((current_kernel.0.clone(), current_kernel.1));

                // makes no sense when doing per kernel instance
                if !options.kernel_instance {
                    let k_count = stat_map
                        .entry((
                            current_kernel.0.clone(),
                            current_kernel.1,
                            "k-count".to_string(),
                        ))
                        .or_insert(0.0);
                    *k_count += 1.0;
                }
                continue;
            }

            for (stat_name, (stat_kind, stat_regex)) in &stats {
                if let Some(value) = stat_regex.captures(line).and_then(|c| c.get(1)) {
                    let value = value.as_str().trim();
                    let value = value.parse::<f64>().wrap_err_with(|| {
                        eyre::eyre!("invalid value {:?} for {:?}", value, stat_name)
                    })?;
                    stat_found.insert(stat_name.clone());
                    let key = (
                        current_kernel.0.to_string(),
                        current_kernel.1,
                        stat_name.to_string(),
                    );
                    // assert_eq!(
                    //     current_kernel.1,
                    //     running_kcount.get(&current_kernel.0).copied().unwrap_or(0),
                    // );

                    if *stat_kind != StatKind::Aggregate {
                        stat_map.insert(key.clone(), value);
                    } else if stat_map.contains_key(&key) {
                        // aggregate: compute diff with last kernel
                        let last_kernel_stat = raw_last.get(stat_name).copied().unwrap_or(0.0);
                        raw_last.insert(stat_name.clone(), value);
                        stat_map[&key] += value - last_kernel_stat;
                    } else {
                        let last_kernel_key =
                            (last_kernel.0.clone(), last_kernel.1, stat_name.to_string());
                        let stat_last_kernel = if stat_map.contains_key(&last_kernel_key) {
                            raw_last[stat_name]
                        } else {
                            0.0
                        };
                        raw_last.insert(stat_name.clone(), value);
                        stat_map.insert(key, value - stat_last_kernel);
                    }
                }
            }
        }
    } else {
        // not per kernel
        all_named_kernels.insert(("final_kernel".to_string(), 0));

        let mut reverse_reader = rev_buf_reader::RevBufReader::new(reader);

        let mut buffer = String::new();
        while let Some(Ok(line)) = reverse_reader.read_line(&mut buffer) {
            for (stat_name, (_stat_kind, stat_regex)) in &stats {
                if stat_found.contains(stat_name) {
                    continue;
                }
                if let Some(value) = stat_regex
                    .captures(line)
                    .and_then(|c| c.get(1))
                    .and_then(|m| m.as_str().trim().parse::<f64>().ok())
                {
                    stat_found.insert(stat_name.clone());
                    stat_map.insert(
                        ("final_kernel".to_string(), 0, stat_name.to_string()),
                        value,
                    );
                }

                if stat_found.len() == stats.len() {
                    break;
                }
            }
        }
    }

    log::debug!(
        "kernels: {:?}",
        stat_map
            .keys()
            .map(|(name, id, _)| (name, id))
            .collect::<IndexSet<_>>()
    );

    // fill in default values
    for (kernel_name, kernel_launch_id) in all_named_kernels.into_iter() {
        for (stat_name, _) in &stats {
            stat_map
                .entry((kernel_name.clone(), kernel_launch_id, stat_name.to_string()))
                .or_insert(0.0);
        }
    }

    stat_map.sort_keys();
    Ok(stat_map)
}
