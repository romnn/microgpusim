#![allow(warnings)]

pub mod cache;

use anyhow::Result;
use std::path::PathBuf;

pub fn locate() -> Result<PathBuf> {
    let use_remote = std::option_env!("USE_REMOTE_ACCELSIM")
        .map(|use_remote| use_remote.to_lowercase() == "yes")
        .unwrap_or(false);
    let accelsim_path = if use_remote {
        PathBuf::from(std::env!("OUT_DIR"))
            .canonicalize()?
            .join("accelsim")
    } else {
        PathBuf::from(std::env!("CARGO_MANIFEST_DIR"))
            .canonicalize()?
            .join("accel-sim-framework-dev")
    };
    Ok(accelsim_path)
}

pub fn locate_nvbit_tracer() -> Result<PathBuf> {
    let accelsim_path = locate()?;
    let default_tracer_root = accelsim_path.join("util/tracer_nvbit/");
    let tracer_root = if let Ok(path) = std::env::var("NVBIT_TRACER_ROOT") {
        PathBuf::from(path)
    } else {
        println!(
            "NVBIT_TRACER_ROOT environment variable is not set, trying {}",
            default_tracer_root.display()
        );
        default_tracer_root
    };
    Ok(tracer_root)
}


// use lazy_static::lazy_static;
// use regex::Regex;
// use std::io::{BufRead, Read, Seek};
// use std::path::Path;
// use std::process::{Command, Output};

// #[derive(thiserror::Error, Debug)]
// pub enum Error {
//     #[error(transparent)]
//     Io(#[from] std::io::Error),

//     #[error(transparent)]
//     Csv(#[from] csv::Error),

//     #[error("missing units")]
//     MissingUnits,

//     #[error("missing metrics")]
//     MissingMetrics,

//     #[error("command failed with bad exit code")]
//     Command(Output),
// }

// #[derive(PartialEq, Clone, Debug, serde::Serialize)]
// pub struct ProfilingResult {
//     pub raw: String,
//     pub metrics: NvprofMetrics,
// }

// #[derive(PartialEq, Clone, Debug, serde::Serialize)]
// pub struct NvprofMetrics {
//     pub test: Metric<f32>,
// }

// #[derive(Hash, PartialEq, Clone, Debug, serde::Serialize)]
// pub struct Metric<T> {
//     value: T,
//     unit: Option<String>,
// }

// #[derive(Clone, Debug, serde::Deserialize)]
// pub struct NvprofAllColumns {
//     Device: Option<String>,
//     Context: Option<String>,
//     Stream: Option<String>,
//     Kernel: Option<String>,
//     Correlation_ID: Option<String>,
//     elapsed_cycles_sm: Option<String>,
//     inst_per_warp: Option<String>,
//     branch_efficiency: Option<String>,
//     warp_execution_efficiency: Option<String>,
//     warp_nonpred_execution_efficiency: Option<String>,
//     inst_replay_overhead: Option<String>,
//     shared_load_transactions_per_request: Option<String>,
//     shared_store_transactions_per_request: Option<String>,
//     local_load_transactions_per_request: Option<String>,
//     local_store_transactions_per_request: Option<String>,
//     gld_transactions_per_request: Option<String>,
//     gst_transactions_per_request: Option<String>,
//     shared_store_transactions: Option<String>,
//     shared_load_transactions: Option<String>,
//     local_load_transactions: Option<String>,
//     local_store_transactions: Option<String>,
//     gld_transactions: Option<String>,
//     gst_transactions: Option<String>,
//     sysmem_read_transactions: Option<String>,
//     sysmem_write_transactions: Option<String>,
//     l2_read_transactions: Option<String>,
//     l2_write_transactions: Option<String>,
//     global_hit_rate: Option<String>,
//     local_hit_rate: Option<String>,
//     gld_requested_throughput: Option<String>,
//     gst_requested_throughput: Option<String>,
//     gld_throughput: Option<String>,
//     gst_throughput: Option<String>,
//     local_memory_overhead: Option<String>,
//     tex_cache_hit_rate: Option<String>,
//     l2_tex_read_hit_rate: Option<String>,
//     l2_tex_write_hit_rate: Option<String>,
//     tex_cache_throughput: Option<String>,
//     l2_tex_read_throughput: Option<String>,
//     l2_tex_write_throughput: Option<String>,
//     l2_read_throughput: Option<String>,
//     l2_write_throughput: Option<String>,
//     sysmem_read_throughput: Option<String>,
//     sysmem_write_throughput: Option<String>,
//     local_load_throughput: Option<String>,
//     local_store_throughput: Option<String>,
//     shared_load_throughput: Option<String>,
//     shared_store_throughput: Option<String>,
//     gld_efficiency: Option<String>,
//     gst_efficiency: Option<String>,
//     tex_cache_transactions: Option<String>,
//     flop_count_dp: Option<String>,
//     flop_count_dp_add: Option<String>,
//     flop_count_dp_fma: Option<String>,
//     flop_count_dp_mul: Option<String>,
//     flop_count_sp: Option<String>,
//     flop_count_sp_add: Option<String>,
//     flop_count_sp_fma: Option<String>,
//     flop_count_sp_mul: Option<String>,
//     flop_count_sp_special: Option<String>,
//     inst_executed: Option<String>,
//     inst_issued: Option<String>,
//     sysmem_utilization: Option<String>,
//     stall_inst_fetch: Option<String>,
//     stall_exec_dependency: Option<String>,
//     stall_memory_dependency: Option<String>,
//     stall_texture: Option<String>,
//     stall_sync: Option<String>,
//     stall_other: Option<String>,
//     stall_constant_memory_dependency: Option<String>,
//     stall_pipe_busy: Option<String>,
//     shared_efficiency: Option<String>,
//     inst_fp_32: Option<String>,
//     inst_fp_64: Option<String>,
//     inst_integer: Option<String>,
//     inst_bit_convert: Option<String>,
//     inst_control: Option<String>,
//     inst_compute_ld_st: Option<String>,
//     inst_misc: Option<String>,
//     inst_inter_thread_communication: Option<String>,
//     issue_slots: Option<String>,
//     cf_issued: Option<String>,
//     cf_executed: Option<String>,
//     ldst_issued: Option<String>,
//     ldst_executed: Option<String>,
//     atomic_transactions: Option<String>,
//     atomic_transactions_per_request: Option<String>,
//     l2_atomic_throughput: Option<String>,
//     l2_atomic_transactions: Option<String>,
//     l2_tex_read_transactions: Option<String>,
//     stall_memory_throttle: Option<String>,
//     stall_not_selected: Option<String>,
//     l2_tex_write_transactions: Option<String>,
//     flop_count_hp: Option<String>,
//     flop_count_hp_add: Option<String>,
//     flop_count_hp_mul: Option<String>,
//     flop_count_hp_fma: Option<String>,
//     inst_fp_16: Option<String>,
//     sysmem_read_utilization: Option<String>,
//     sysmem_write_utilization: Option<String>,
//     pcie_total_data_transmitted: Option<String>,
//     pcie_total_data_received: Option<String>,
//     inst_executed_global_loads: Option<String>,
//     inst_executed_local_loads: Option<String>,
//     inst_executed_shared_loads: Option<String>,
//     inst_executed_surface_loads: Option<String>,
//     inst_executed_global_stores: Option<String>,
//     inst_executed_local_stores: Option<String>,
//     inst_executed_shared_stores: Option<String>,
//     inst_executed_surface_stores: Option<String>,
//     inst_executed_global_atomics: Option<String>,
//     inst_executed_global_reductions: Option<String>,
//     inst_executed_surface_atomics: Option<String>,
//     inst_executed_surface_reductions: Option<String>,
//     inst_executed_shared_atomics: Option<String>,
//     inst_executed_tex_ops: Option<String>,
//     l2_global_load_bytes: Option<String>,
//     l2_local_load_bytes: Option<String>,
//     l2_surface_load_bytes: Option<String>,
//     l2_local_global_store_bytes: Option<String>,
//     l2_global_reduction_bytes: Option<String>,
//     l2_global_atomic_store_bytes: Option<String>,
//     l2_surface_store_bytes: Option<String>,
//     l2_surface_reduction_bytes: Option<String>,
//     l2_surface_atomic_store_bytes: Option<String>,
//     global_load_requests: Option<String>,
//     local_load_requests: Option<String>,
//     surface_load_requests: Option<String>,
//     global_store_requests: Option<String>,
//     local_store_requests: Option<String>,
//     surface_store_requests: Option<String>,
//     global_atomic_requests: Option<String>,
//     global_reduction_requests: Option<String>,
//     surface_atomic_requests: Option<String>,
//     surface_reduction_requests: Option<String>,
//     sysmem_read_bytes: Option<String>,
//     sysmem_write_bytes: Option<String>,
//     l2_tex_hit_rate: Option<String>,
//     texture_load_requests: Option<String>,
//     unique_warps_launched: Option<String>,
//     sm_efficiency: Option<String>,
//     achieved_occupancy: Option<String>,
//     ipc: Option<String>,
//     issued_ipc: Option<String>,
//     issue_slot_utilization: Option<String>,
//     eligible_warps_per_cycle: Option<String>,
//     tex_utilization: Option<String>,
//     l2_utilization: Option<String>,
//     shared_utilization: Option<String>,
//     ldst_fu_utilization: Option<String>,
//     cf_fu_utilization: Option<String>,
//     special_fu_utilization: Option<String>,
//     tex_fu_utilization: Option<String>,
//     single_precision_fu_utilization: Option<String>,
//     double_precision_fu_utilization: Option<String>,
//     flop_hp_efficiency: Option<String>,
//     flop_sp_efficiency: Option<String>,
//     flop_dp_efficiency: Option<String>,
//     dram_read_transactions: Option<String>,
//     dram_write_transactions: Option<String>,
//     dram_read_throughput: Option<String>,
//     dram_write_throughput: Option<String>,
//     dram_utilization: Option<String>,
//     half_precision_fu_utilization: Option<String>,
//     dram_write_bytes: Option<String>,
//     ecc_transactions: Option<String>,
//     ecc_throughput: Option<String>,
//     dram_read_bytes: Option<String>,
// }

// pub fn nvprof<P, A>(executable: P, args: A) -> Result<NvprofMetrics, Error>
// where
//     P: AsRef<Path>,
//     A: IntoIterator,
//     <A as IntoIterator>::Item: AsRef<std::ffi::OsStr>,
// {
//     let tmp_dir = tempfile::tempdir()?;
//     let log_file_path = tmp_dir.path().join("log_file.csv");
//     let mut cmd = Command::new("nvprof");
//     cmd.args([
//         "--unified-memory-profiling",
//         "off",
//         "--concurrent-kernels",
//         "off",
//         "--print-gpu-trace",
//         "--events",
//         "elapsed_cycles_sm",
//         "-u",
//         "us",
//         "--metrics",
//         "all",
//         "--demangling",
//         "off",
//         "--csv",
//         "--log-file",
//     ])
//     .arg(&log_file_path)
//     .arg(executable.as_ref())
//     .args(args.into_iter());
//     // dbg!(&cmd);

//     let result = cmd.output()?;
//     if !result.status.success() {
//         return Err(Error::Command(result));
//     }

//     let log_file = std::fs::OpenOptions::new()
//         .read(true)
//         .open(&log_file_path)?;

//     let mut original_log = String::new();
//     let mut log_reader = std::io::BufReader::new(log_file);
//     log_reader.read_to_string(&mut original_log)?;
//     log_reader.rewind()?;

//     // seek to valid start of csv data
//     let mut lines = log_reader.by_ref().lines();
//     for line in &mut lines {
//         let Ok(line) = line else {
//             continue
//         };
//         lazy_static! {
//             pub static ref PROFILE_RESULT_REGEX: Regex =
//                 Regex::new(r"^==\d*==\s*Profiling result:\s*$").unwrap();
//         }
//         lazy_static! {
//             pub static ref PROFILER_DISCONNECTED_REGEX: Regex =
//                 Regex::new(r"^==PROF== Disconnected\s*$").unwrap();
//         }
//         // println!("line: {:#?}", &line);
//         if PROFILE_RESULT_REGEX.is_match(&line) {
//             break;
//         }
//     }

//     // upgrade reader to a csv reader, keeping the current position
//     let mut csv_reader = csv::ReaderBuilder::new()
//         .flexible(false)
//         .from_reader(log_reader);

//     let mut records = csv_reader.deserialize();
//     let units: NvprofAllColumns = records.next().ok_or(Error::MissingUnits)??;
//     let metrics: NvprofAllColumns = records.next().ok_or(Error::MissingMetrics)??;

//     // dbg!(&units);
//     // dbg!(&metrics);
//     let result = NvprofMetrics {
//         test: Metric {
//             value: 0.0,
//             unit: None,
//         },
//     };
//     Ok(result)
// }

// log_file = results_dir / "{}.result.nvprof.txt".format(r)

// executable = path / inp.executable
// assert executable.is_file()
// utils.chmod_x(executable)

// cmd = [
//     "nvprof",
//     "--unified-memory-profiling",
//     "off",
//     "--concurrent-kernels",
//     "off",
//     "--print-gpu-trace",
//     "-u",
//     "us",
//     "--demangling",
//     "off",
//     "--csv",
//     "--log-file",
//     str(log_file.absolute()),
//     str(executable.absolute()),
//     inp.args,
// ]
// cmd = " ".join(cmd)
// try:
//     _, stdout, stderr, _ = utils.run_cmd(
//         cmd,
//         cwd=path,
//         timeout_sec=timeout_mins * 60,
//         save_to=results_dir / "nvprof-kernels",
//     )
//     print("stdout:")
//     print(stdout)
//     print("stderr:")
//     print(stderr)

//     with open(str(log_file.absolute()), "r") as f:
//         print("log file:")
//         print(f.read())

// except utils.ExecError as e:
//     with open(str(log_file.absolute()), "r") as f:
//         print("log file:")
//         print(f.read())
//     raise e

// cycles_log_file = results_dir / "{}.result.nvprof.cycles.txt".format(r)
