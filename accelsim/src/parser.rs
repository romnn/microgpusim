use super::read::BufReadLine;
use clap::Parser;
use color_eyre::eyre;
use itertools::Itertools;

use once_cell::sync::Lazy;
use regex::Regex;
use std::{
    collections::{HashMap, HashSet},
    fs, io,
    path::{Path, PathBuf},
};

// -R -K -k -B rodinia_2.0-ft -C QV100-PTX
#[derive(Parser, Debug)]
pub struct Options {
    #[arg(short = 'i', long = "input")]
    pub input: PathBuf,

    #[arg(short = 'o', long = "output")]
    pub output: Option<PathBuf>,

    #[arg(short = 'k', long = "per-kernel")]
    pub per_kernel: bool,

    #[arg(short = 'K', long = "kernel-instance")]
    pub kernel_instance: bool,

    #[arg(long = "strict")]
    pub strict: bool,
}

impl Options {
    pub fn new(input: impl Into<PathBuf>) -> Self {
        Self {
            input: input.into(),
            output: None,
            per_kernel: false,
            kernel_instance: false,
            strict: false,
        }
    }

    pub fn save_to(&mut self, output: impl Into<PathBuf>) -> &mut Self {
        self.output = Some(output.into());
        self
    }
}

/// Stats map
pub type Stats = HashMap<(String, u16, String), f64>;

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
fn get_version(mut f: impl BufReadLine + io::Seek) -> Option<String> {
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
fn check_finished(mut f: impl BufReadLine + io::Seek) -> bool {
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
pub fn parse(options: &Options) -> eyre::Result<Stats> {
    let finished = {
        let file = fs::OpenOptions::new().read(true).open(&options.input)?;
        let mut reader = rev_buf_reader::RevBufReader::new(file);
        check_finished(&mut reader)
    };
    let version = {
        let file = fs::OpenOptions::new().read(true).open(&options.input)?;
        let mut reader = io::BufReader::new(file);
        get_version(&mut reader)
    };

    println!("GPGPU-sim version: {:?}", &version);

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
            r"gpu_occupancy\s*=\s*(.*)"
        ),
        stat!(
            "total_occupancy",
            StatKind::Aggregate,
            r"gpu_tot_occupancy\s*=\s*(.*)"
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
    let outcome = vec![
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
        .map(|((space, space_name), (outcome, outcome_name))| {
            stat!(
                format!("l2_cache_{space_name}_{outcome_name}"),
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
    l2_cache_stats.extend([
        stat!(
            "l2_cache_global_read_total",
            StatKind::Aggregate,
            r"\s+L2_cache_stats_breakdown\[GLOBAL_ACC_R\]\[TOTAL_ACCESS\]\s*=\s*(.*)"
        ),
        stat!(
            "l2_cache_global_write_total",
            StatKind::Aggregate,
            r"\s+L2_cache_stats_breakdown\[GLOBAL_ACC_W\]\[TOTAL_ACCESS\]\s*=\s*(.*)"
        ),
        stat!(
            "l2_cache_inst_read_total",
            StatKind::Aggregate,
            r"\s+L2_cache_stats_breakdown\[INST_ACC_W\]\[TOTAL_ACCESS\]\s*=\s*(.*)"
        ),
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

    let mut total_core_cache_stats: Vec<_> = mem_space
        .iter()
        .cartesian_product(outcome.iter())
        .map(|((space, space_name), (outcome, outcome_name))| {
            stat!(
                format!("total_core_cache_{space_name}_{outcome_name}"),
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
    total_core_cache_stats.extend([stat!(
        "total_core_cache_inst_read_total",
        StatKind::Aggregate,
        r"\s+Total_core_cache_stats_breakdown\[INST_ACC_R\]\[TOTAL_ACCESS\]\s*=\s*(.*)"
    )]);

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

    let l1_data_cache_stats = vec![
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
        l1_inst_cache_stats,
        l1_data_cache_stats,
        l1_const_cache_stats,
        l1_tex_cache_stats,
        total_core_cache_stats,
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
            eyre::bail!(
                "{} is invalid: termination message from GPGPU-Sim not found",
                options.input.display()
            );
        }
        eprintln!(
            "{} is invalid: termination message from GPGPU-Sim not found",
            options.input.display()
        );
    }

    let mut all_named_kernels: HashSet<(String, u16)> = HashSet::new();

    let mut stat_found: HashSet<String> = HashSet::new();

    let file = fs::OpenOptions::new().read(true).open(&options.input)?;

    let mut stat_map: Stats = HashMap::new();

    if options.per_kernel {
        let mut current_kernel = String::new();
        let mut last_kernel = (String::new(), 0);
        let mut raw_last: HashMap<String, f64> = HashMap::new();
        let mut running_kcount = HashMap::new();

        let mut reader = io::BufReader::new(file);
        let mut buffer = String::new();

        while let Some(Ok(line)) = reader.read_line(&mut buffer) {
            // was simulation aborted due to too many instructions?
            // then ignore the last kernel launch, as it is no complete
            // (only appies if we are doing kernel-by-kernel stats)

            if LAST_KERNEL_BREAK_REGEX.captures(line).is_some() {
                eprintln!(
                    "{}: found max instructions - ignoring last kernel",
                    options.input.display()
                );
                // remove
                for stat_name in stats.keys() {
                    stat_map.remove(&(
                        current_kernel.to_string(),
                        running_kcount[&current_kernel],
                        stat_name.to_string(),
                    ));
                }
            }

            if let Some(kernel_name) = KERNEL_NAME_REGEX
                .captures(line)
                .and_then(|c| c.get(1))
                .map(|m| m.as_str().trim().to_string())
            {
                let last_kernel_kcount = running_kcount.get(&current_kernel).copied().unwrap_or(0);
                last_kernel = (current_kernel, last_kernel_kcount);
                current_kernel = kernel_name;

                if options.kernel_instance {
                    if !running_kcount.contains_key(&current_kernel) {
                        running_kcount.insert(current_kernel.clone(), 0);
                    } else if let Some(c) = running_kcount.get_mut(&current_kernel) {
                        *c += 1;
                    }
                }

                all_named_kernels.insert((
                    current_kernel.clone(),
                    running_kcount.get(&current_kernel).copied().unwrap_or(0),
                ));

                let k_count = stat_map
                    .entry((
                        current_kernel.to_string(),
                        running_kcount.get(&current_kernel).copied().unwrap_or(0),
                        "k-count".to_string(),
                    ))
                    .or_insert(0.0);
                *k_count += 1.0;
                continue;
            }

            for (stat_name, (stat_kind, stat_regex)) in &stats {
                if let Some(value) = stat_regex
                    .captures(line)
                    .and_then(|c| c.get(1))
                    .and_then(|m| m.as_str().trim().parse::<f64>().ok())
                {
                    stat_found.insert(stat_name.clone());
                    let key = (
                        current_kernel.to_string(),
                        running_kcount.get(&current_kernel).copied().unwrap_or(0),
                        stat_name.to_string(),
                    );
                    if stat_kind != &StatKind::Aggregate {
                        stat_map.insert(key.clone(), value);
                    } else if stat_map.contains_key(&key) {
                        let stat_last_kernel = raw_last.get(stat_name).copied().unwrap_or(0.0);
                        raw_last.insert(stat_name.clone(), value);
                        if let Some(v) = stat_map.get_mut(&key) {
                            *v += value - stat_last_kernel;
                        }
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

        let mut reader = rev_buf_reader::RevBufReader::new(file);

        let mut buffer = String::new();
        while let Some(Ok(line)) = reader.read_line(&mut buffer) {
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

    if let Some(out_file) = options.output.as_ref() {
        save_stats_to_file(&stat_map, out_file)?;
    }

    Ok(stat_map)
}

fn save_stats_to_file(stats: &Stats, out_file: &Path) -> eyre::Result<()> {
    if let Some(parent) = &out_file.parent() {
        fs::create_dir_all(parent).ok();
    }
    let output_file = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(out_file)?;

    let mut csv_writer = csv::WriterBuilder::new()
        .flexible(false)
        .from_writer(output_file);

    csv_writer.write_record(["kernel", "kernel_id", "stat", "value"])?;

    // sort stats before writing to csv
    let mut sorted_stats: Vec<_> = stats.iter().collect();
    sorted_stats.sort_by(|a, b| a.0.cmp(b.0));

    for ((kernel, kcount, stat), value) in &sorted_stats {
        csv_writer.write_record([kernel, &kcount.to_string(), stat, &value.to_string()])?;
    }
    Ok(())
}
