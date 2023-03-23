#![allow(warnings)]

use lazy_static::lazy_static;
use regex::Regex;
use std::collections::HashMap;
use std::io::{BufRead, Read, Seek};
use std::path::Path;
use std::process::{Command, Output};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Csv(#[from] csv::Error),

    #[error("missing units")]
    MissingUnits,

    #[error("missing metrics")]
    MissingMetrics,

    #[error("command failed with bad exit code")]
    Command(Output),
}

#[derive(Hash, PartialEq, Clone, Default, Debug, serde::Serialize, serde::Deserialize)]
pub struct Metric<T> {
    value: Option<T>,
    unit: Option<String>,
}

#[derive(PartialEq, Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ProfilingResult {
    pub raw: String,
    pub metrics: NvprofMetrics,
}

#[derive(PartialEq, Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct NvprofMetrics {
    pub Device: Metric<String>,
    pub Context: Metric<usize>,
    pub Stream: Metric<usize>,
    pub Kernel: Metric<String>,
    pub Correlation_ID: Metric<usize>,
    // pub elapsed_cycles_sm: Metric<String>,
    // pub inst_per_warp: Metric<String>,
    // pub branch_efficiency: Metric<String>,
    // pub warp_execution_efficiency: Metric<String>,
    // pub warp_nonpred_execution_efficiency: Metric<String>,
    // pub inst_replay_overhead: Metric<String>,
    pub shared_load_transactions_per_request: Metric<f32>,
    pub shared_store_transactions_per_request: Metric<f32>,
    pub local_load_transactions_per_request: Metric<f32>,
    pub local_store_transactions_per_request: Metric<f32>,
    pub gld_transactions_per_request: Metric<f32>,
    pub gst_transactions_per_request: Metric<f32>,
    pub shared_store_transactions: Metric<usize>,
    pub shared_load_transactions: Metric<usize>,
    pub local_load_transactions: Metric<usize>,
    pub local_store_transactions: Metric<usize>,
    pub gld_transactions: Metric<usize>,
    pub gst_transactions: Metric<usize>,
    pub sysmem_read_transactions: Metric<usize>,
    pub sysmem_write_transactions: Metric<usize>,
    pub l2_read_transactions: Metric<usize>,
    pub l2_write_transactions: Metric<usize>,
    // pub global_hit_rate: Metric<String>,
    // pub local_hit_rate: Metric<String>,
    // pub gld_requested_throughput: Metric<String>,
    // pub gst_requested_throughput: Metric<String>,
    // pub gld_throughput: Metric<String>,
    // pub gst_throughput: Metric<String>,
    // pub local_memory_overhead: Metric<String>,
    // pub tex_cache_hit_rate: Metric<String>,
    // pub l2_tex_read_hit_rate: Metric<String>,
    // pub l2_tex_write_hit_rate: Metric<String>,
    // pub tex_cache_throughput: Metric<String>,
    // pub l2_tex_read_throughput: Metric<String>,
    // pub l2_tex_write_throughput: Metric<String>,
    // pub l2_read_throughput: Metric<String>,
    // pub l2_write_throughput: Metric<String>,
    // pub sysmem_read_throughput: Metric<String>,
    // pub sysmem_write_throughput: Metric<String>,
    // pub local_load_throughput: Metric<String>,
    // pub local_store_throughput: Metric<String>,
    // pub shared_load_throughput: Metric<String>,
    // pub shared_store_throughput: Metric<String>,
    // pub gld_efficiency: Metric<String>,
    // pub gst_efficiency: Metric<String>,
    // pub tex_cache_transactions: Metric<String>,
    // pub flop_count_dp: Metric<String>,
    // pub flop_count_dp_add: Metric<String>,
    // pub flop_count_dp_fma: Metric<String>,
    // pub flop_count_dp_mul: Metric<String>,
    // pub flop_count_sp: Metric<String>,
    // pub flop_count_sp_add: Metric<String>,
    // pub flop_count_sp_fma: Metric<String>,
    // pub flop_count_sp_mul: Metric<String>,
    // pub flop_count_sp_special: Metric<String>,
    // pub inst_executed: Metric<String>,
    // pub inst_issued: Metric<String>,
    // pub sysmem_utilization: Metric<String>,
    // pub stall_inst_fetch: Metric<String>,
    // pub stall_exec_dependency: Metric<String>,
    // pub stall_memory_dependency: Metric<String>,
    // pub stall_texture: Metric<String>,
    // pub stall_sync: Metric<String>,
    // pub stall_other: Metric<String>,
    // pub stall_constant_memory_dependency: Metric<String>,
    // pub stall_pipe_busy: Metric<String>,
    // pub shared_efficiency: Metric<String>,
    // pub inst_fp_32: Metric<String>,
    // pub inst_fp_64: Metric<String>,
    // pub inst_integer: Metric<String>,
    // pub inst_bit_convert: Metric<String>,
    // pub inst_control: Metric<String>,
    // pub inst_compute_ld_st: Metric<String>,
    // pub inst_misc: Metric<String>,
    // pub inst_inter_thread_communication: Metric<String>,
    // pub issue_slots: Metric<String>,
    // pub cf_issued: Metric<String>,
    // pub cf_executed: Metric<String>,
    // pub ldst_issued: Metric<String>,
    // pub ldst_executed: Metric<String>,
    pub atomic_transactions: Metric<usize>,
    pub atomic_transactions_per_request: Metric<f32>,
    // pub l2_atomic_throughput: Metric<String>,
    // pub l2_atomic_transactions: Metric<String>,
    // pub l2_tex_read_transactions: Metric<String>,
    // pub stall_memory_throttle: Metric<String>,
    // pub stall_not_selected: Metric<String>,
    // pub l2_tex_write_transactions: Metric<String>,
    // pub flop_count_hp: Metric<String>,
    // pub flop_count_hp_add: Metric<String>,
    // pub flop_count_hp_mul: Metric<String>,
    // pub flop_count_hp_fma: Metric<String>,
    // pub inst_fp_16: Metric<String>,
    // pub sysmem_read_utilization: Option<String>,
    // pub sysmem_write_utilization: Option<String>,
    // pub pcie_total_data_transmitted: Option<String>,
    // pub pcie_total_data_received: Option<String>,
    // pub inst_executed_global_loads: Option<String>,
    // pub inst_executed_local_loads: Option<String>,
    // pub inst_executed_shared_loads: Option<String>,
    // pub inst_executed_surface_loads: Option<String>,
    // pub inst_executed_global_stores: Option<String>,
    // pub inst_executed_local_stores: Option<String>,
    // pub inst_executed_shared_stores: Option<String>,
    // pub inst_executed_surface_stores: Option<String>,
    // pub inst_executed_global_atomics: Option<String>,
    // pub inst_executed_global_reductions: Option<String>,
    // pub inst_executed_surface_atomics: Option<String>,
    // pub inst_executed_surface_reductions: Option<String>,
    // pub inst_executed_shared_atomics: Option<String>,
    // pub inst_executed_tex_ops: Option<String>,
    pub l2_global_load_bytes: Metric<usize>,
    pub l2_local_load_bytes: Metric<usize>,
    pub l2_surface_load_bytes: Metric<usize>,
    pub l2_local_global_store_bytes: Metric<usize>,
    pub l2_global_reduction_bytes: Metric<usize>,
    pub l2_global_atomic_store_bytes: Metric<usize>,
    pub l2_surface_store_bytes: Metric<usize>,
    pub l2_surface_reduction_bytes: Metric<usize>,
    pub l2_surface_atomic_store_bytes: Metric<usize>,
    pub global_load_requests: Metric<usize>,
    pub local_load_requests: Metric<usize>,
    pub surface_load_requests: Metric<usize>,
    pub global_store_requests: Metric<usize>,
    pub local_store_requests: Metric<usize>,
    pub surface_store_requests: Metric<usize>,
    pub global_atomic_requests: Metric<usize>,
    pub global_reduction_requests: Metric<usize>,
    pub surface_atomic_requests: Metric<usize>,
    pub surface_reduction_requests: Metric<usize>,
    // pub sysmem_read_bytes: Option<String>,
    // pub sysmem_write_bytes: Option<String>,
    // pub l2_tex_hit_rate: Option<String>,
    // pub texture_load_requests: Option<String>,
    // pub unique_warps_launched: Option<String>,
    // pub sm_efficiency: Option<String>,
    // pub achieved_occupancy: Option<String>,
    // pub ipc: Option<String>,
    // pub issued_ipc: Option<String>,
    // pub issue_slot_utilization: Option<String>,
    // pub eligible_warps_per_cycle: Option<String>,
    // pub tex_utilization: Option<String>,
    // pub l2_utilization: Option<String>,
    // pub shared_utilization: Option<String>,
    // pub ldst_fu_utilization: Option<String>,
    // pub cf_fu_utilization: Option<String>,
    // pub special_fu_utilization: Option<String>,
    // pub tex_fu_utilization: Option<String>,
    // pub single_precision_fu_utilization: Option<String>,
    // pub double_precision_fu_utilization: Option<String>,
    // pub flop_hp_efficiency: Option<String>,
    // pub flop_sp_efficiency: Option<String>,
    // pub flop_dp_efficiency: Option<String>,
    pub dram_read_transactions: Metric<usize>,
    pub dram_write_transactions: Metric<usize>,
    pub dram_read_throughput: Metric<f32>,
    pub dram_write_throughput: Metric<f32>,
    // pub dram_utilization: Option<String>,
    // pub half_precision_fu_utilization: Option<String>,
    pub dram_write_bytes: Metric<usize>,
    // pub ecc_transactions: Option<String>,
    // pub ecc_throughput: Option<String>,
    pub dram_read_bytes: Metric<usize>,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct NvprofAllColumns {
    Device: Option<String>,
    Context: Option<usize>,
    Stream: Option<usize>,
    Kernel: Option<String>,
    Correlation_ID: Option<usize>,
    elapsed_cycles_sm: Option<String>,
    inst_per_warp: Option<String>,
    branch_efficiency: Option<String>,
    warp_execution_efficiency: Option<String>,
    warp_nonpred_execution_efficiency: Option<String>,
    inst_replay_overhead: Option<String>,
    shared_load_transactions_per_request: Option<f32>,
    shared_store_transactions_per_request: Option<f32>,
    local_load_transactions_per_request: Option<f32>,
    local_store_transactions_per_request: Option<f32>,
    gld_transactions_per_request: Option<f32>,
    gst_transactions_per_request: Option<f32>,
    shared_store_transactions: Option<usize>,
    shared_load_transactions: Option<usize>,
    local_load_transactions: Option<usize>,
    local_store_transactions: Option<usize>,
    gld_transactions: Option<usize>,
    gst_transactions: Option<usize>,
    sysmem_read_transactions: Option<usize>,
    sysmem_write_transactions: Option<usize>,
    l2_read_transactions: Option<usize>,
    l2_write_transactions: Option<usize>,
    global_hit_rate: Option<String>,
    local_hit_rate: Option<String>,
    gld_requested_throughput: Option<String>,
    gst_requested_throughput: Option<String>,
    gld_throughput: Option<String>,
    gst_throughput: Option<String>,
    local_memory_overhead: Option<String>,
    tex_cache_hit_rate: Option<String>,
    l2_tex_read_hit_rate: Option<String>,
    l2_tex_write_hit_rate: Option<String>,
    tex_cache_throughput: Option<String>,
    l2_tex_read_throughput: Option<String>,
    l2_tex_write_throughput: Option<String>,
    l2_read_throughput: Option<String>,
    l2_write_throughput: Option<String>,
    sysmem_read_throughput: Option<String>,
    sysmem_write_throughput: Option<String>,
    local_load_throughput: Option<String>,
    local_store_throughput: Option<String>,
    shared_load_throughput: Option<String>,
    shared_store_throughput: Option<String>,
    gld_efficiency: Option<String>,
    gst_efficiency: Option<String>,
    tex_cache_transactions: Option<String>,
    flop_count_dp: Option<String>,
    flop_count_dp_add: Option<String>,
    flop_count_dp_fma: Option<String>,
    flop_count_dp_mul: Option<String>,
    flop_count_sp: Option<String>,
    flop_count_sp_add: Option<String>,
    flop_count_sp_fma: Option<String>,
    flop_count_sp_mul: Option<String>,
    flop_count_sp_special: Option<String>,
    inst_executed: Option<String>,
    inst_issued: Option<String>,
    sysmem_utilization: Option<String>,
    stall_inst_fetch: Option<String>,
    stall_exec_dependency: Option<String>,
    stall_memory_dependency: Option<String>,
    stall_texture: Option<String>,
    stall_sync: Option<String>,
    stall_other: Option<String>,
    stall_constant_memory_dependency: Option<String>,
    stall_pipe_busy: Option<String>,
    shared_efficiency: Option<String>,
    inst_fp_32: Option<String>,
    inst_fp_64: Option<String>,
    inst_integer: Option<String>,
    inst_bit_convert: Option<String>,
    inst_control: Option<String>,
    inst_compute_ld_st: Option<String>,
    inst_misc: Option<String>,
    inst_inter_thread_communication: Option<String>,
    issue_slots: Option<String>,
    cf_issued: Option<String>,
    cf_executed: Option<String>,
    ldst_issued: Option<String>,
    ldst_executed: Option<String>,
    atomic_transactions: Option<usize>,
    atomic_transactions_per_request: Option<f32>,
    l2_atomic_throughput: Option<String>,
    l2_atomic_transactions: Option<String>,
    l2_tex_read_transactions: Option<String>,
    stall_memory_throttle: Option<String>,
    stall_not_selected: Option<String>,
    l2_tex_write_transactions: Option<String>,
    flop_count_hp: Option<String>,
    flop_count_hp_add: Option<String>,
    flop_count_hp_mul: Option<String>,
    flop_count_hp_fma: Option<String>,
    inst_fp_16: Option<String>,
    sysmem_read_utilization: Option<String>,
    sysmem_write_utilization: Option<String>,
    pcie_total_data_transmitted: Option<String>,
    pcie_total_data_received: Option<String>,
    inst_executed_global_loads: Option<String>,
    inst_executed_local_loads: Option<String>,
    inst_executed_shared_loads: Option<String>,
    inst_executed_surface_loads: Option<String>,
    inst_executed_global_stores: Option<String>,
    inst_executed_local_stores: Option<String>,
    inst_executed_shared_stores: Option<String>,
    inst_executed_surface_stores: Option<String>,
    inst_executed_global_atomics: Option<String>,
    inst_executed_global_reductions: Option<String>,
    inst_executed_surface_atomics: Option<String>,
    inst_executed_surface_reductions: Option<String>,
    inst_executed_shared_atomics: Option<String>,
    inst_executed_tex_ops: Option<String>,
    l2_global_load_bytes: Option<usize>,
    l2_local_load_bytes: Option<usize>,
    l2_surface_load_bytes: Option<usize>,
    l2_local_global_store_bytes: Option<usize>,
    l2_global_reduction_bytes: Option<usize>,
    l2_global_atomic_store_bytes: Option<usize>,
    l2_surface_store_bytes: Option<usize>,
    l2_surface_reduction_bytes: Option<usize>,
    l2_surface_atomic_store_bytes: Option<usize>,
    global_load_requests: Option<usize>,
    local_load_requests: Option<usize>,
    surface_load_requests: Option<usize>,
    global_store_requests: Option<usize>,
    local_store_requests: Option<usize>,
    surface_store_requests: Option<usize>,
    global_atomic_requests: Option<usize>,
    global_reduction_requests: Option<usize>,
    surface_atomic_requests: Option<usize>,
    surface_reduction_requests: Option<usize>,
    sysmem_read_bytes: Option<usize>,
    sysmem_write_bytes: Option<usize>,
    l2_tex_hit_rate: Option<String>,
    texture_load_requests: Option<String>,
    unique_warps_launched: Option<String>,
    sm_efficiency: Option<String>,
    achieved_occupancy: Option<String>,
    ipc: Option<String>,
    issued_ipc: Option<String>,
    issue_slot_utilization: Option<String>,
    eligible_warps_per_cycle: Option<String>,
    tex_utilization: Option<String>,
    l2_utilization: Option<String>,
    shared_utilization: Option<String>,
    ldst_fu_utilization: Option<String>,
    cf_fu_utilization: Option<String>,
    special_fu_utilization: Option<String>,
    tex_fu_utilization: Option<String>,
    single_precision_fu_utilization: Option<String>,
    double_precision_fu_utilization: Option<String>,
    flop_hp_efficiency: Option<String>,
    flop_sp_efficiency: Option<String>,
    flop_dp_efficiency: Option<String>,
    dram_read_transactions: Option<usize>,
    dram_write_transactions: Option<usize>,
    dram_read_throughput: Option<f32>,
    dram_write_throughput: Option<f32>,
    dram_utilization: Option<String>,
    half_precision_fu_utilization: Option<String>,
    dram_write_bytes: Option<usize>,
    ecc_transactions: Option<String>,
    ecc_throughput: Option<String>,
    dram_read_bytes: Option<usize>,
}

/// Profile test application using nvbprof profiler.
///
/// Note: The nvbprof compiler is not recommended for newer devices.
///
/// # Errors
/// When creating temp dir fails.
/// When profiling fails.
/// When application fails.
#[allow(clippy::too_many_lines)]
pub fn nvprof<P, A>(executable: P, args: A) -> Result<ProfilingResult, Error>
where
    P: AsRef<Path>,
    A: IntoIterator,
    <A as IntoIterator>::Item: AsRef<std::ffi::OsStr>,
{
    let tmp_dir = tempfile::tempdir()?;
    let log_file_path = tmp_dir.path().join("log_file.csv");
    let mut cmd = Command::new("nvprof");
    cmd.args([
        "--unified-memory-profiling",
        "off",
        "--concurrent-kernels",
        "off",
        "--print-gpu-trace",
        "--events",
        "elapsed_cycles_sm",
        "-u",
        "us",
        "--metrics",
        "all",
        "--demangling",
        "off",
        "--csv",
        "--log-file",
    ])
    .arg(&log_file_path)
    .arg(executable.as_ref())
    .args(args.into_iter());
    // dbg!(&cmd);

    let result = cmd.output()?;
    if !result.status.success() {
        return Err(Error::Command(result));
    }

    let log_file = std::fs::OpenOptions::new()
        .read(true)
        .open(&log_file_path)?;

    let mut original_log = String::new();
    let mut log_reader = std::io::BufReader::new(log_file);
    log_reader.read_to_string(&mut original_log)?;
    log_reader.rewind()?;
    println!("{original_log}");

    // seek to valid start of csv data
    let mut lines = log_reader.by_ref().lines();
    for line in &mut lines {
        let Ok(line) = line else {
            continue
        };
        lazy_static! {
            pub static ref PROFILE_RESULT_REGEX: Regex =
                Regex::new(r"^==\d*==\s*Profiling result:\s*$").unwrap();
        }
        lazy_static! {
            pub static ref PROFILER_DISCONNECTED_REGEX: Regex =
                Regex::new(r"^==PROF== Disconnected\s*$").unwrap();
        }
        // println!("line: {:#?}", &line);
        if PROFILE_RESULT_REGEX.is_match(&line) {
            break;
        }
    }

    // upgrade reader to a csv reader, keeping the current position
    let mut csv_reader = csv::ReaderBuilder::new()
        .flexible(false)
        .from_reader(log_reader);

    let mut records = csv_reader.deserialize();
    // let units: HashMap<String, String> = records.next().ok_or(Error::MissingUnits)??;
    // let units: NvprofAllColumns = records.next().ok_or(Error::MissingUnits)??;
    records.next().ok_or(Error::MissingUnits)?;
    let values: NvprofAllColumns = records.next().ok_or(Error::MissingMetrics)??;

    // dbg!(&units);
    dbg!(&values);
    let metrics = NvprofMetrics {
        Device: Metric {
            value: values.Device,
            ..Metric::default() // unit: units.Device,
        },
        Context: Metric {
            value: values.Context,
            ..Metric::default() // unit: units.Context,
        },
        Stream: Metric {
            value: values.Stream,
            ..Metric::default() // unit: units.Stream,
        },
        Kernel: Metric {
            value: values.Kernel,
            ..Metric::default() // unit: units.Kernel,
        },
        Correlation_ID: Metric {
            value: values.Correlation_ID,
            ..Metric::default() // unit: units.Correlation_ID,
        },
        shared_load_transactions_per_request: Metric {
            value: values.shared_load_transactions_per_request,
            ..Metric::default() // unit: units.shared_load_transactions_per_request,
        },
        shared_store_transactions_per_request: Metric {
            value: values.shared_store_transactions_per_request,
            ..Metric::default() // unit: units.shared_store_transactions_per_request,
        },
        local_load_transactions_per_request: Metric {
            value: values.local_load_transactions_per_request,
            ..Metric::default() // unit: units.local_load_transactions_per_request,
        },
        local_store_transactions_per_request: Metric {
            value: values.local_store_transactions_per_request,
            ..Metric::default() // unit: units.local_store_transactions_per_request,
        },
        gld_transactions_per_request: Metric {
            value: values.gld_transactions_per_request,
            ..Metric::default() // unit: units.gld_transactions_per_request,
        },
        gst_transactions_per_request: Metric {
            value: values.gst_transactions_per_request,
            ..Metric::default() // unit: units.gst_transactions_per_request,
        },
        shared_store_transactions: Metric {
            value: values.shared_store_transactions,
            ..Metric::default() // unit: units.shared_store_transactions,
        },
        shared_load_transactions: Metric {
            value: values.shared_load_transactions,
            ..Metric::default() // unit: units.shared_load_transactions,
        },
        local_load_transactions: Metric {
            value: values.local_load_transactions,
            ..Metric::default() // unit: units.local_load_transactions,
        },
        local_store_transactions: Metric {
            value: values.local_store_transactions,
            ..Metric::default() // unit: units.local_store_transactions,
        },
        gld_transactions: Metric {
            value: values.gld_transactions,
            ..Metric::default() // unit: units.gld_transactions,
        },
        gst_transactions: Metric {
            value: values.gst_transactions,
            ..Metric::default() // unit: units.gst_transactions,
        },
        sysmem_read_transactions: Metric {
            value: values.sysmem_read_transactions,
            ..Metric::default() // unit: units.sysmem_read_transactions,
        },
        sysmem_write_transactions: Metric {
            value: values.sysmem_write_transactions,
            ..Metric::default() // unit: units.sysmem_write_transactions,
        },
        l2_read_transactions: Metric {
            value: values.l2_read_transactions,
            ..Metric::default() // unit: units.l2_read_transactions,
        },
        l2_write_transactions: Metric {
            value: values.l2_write_transactions,
            ..Metric::default() // unit: units.l2_write_transactions,
        },
        atomic_transactions: Metric {
            value: values.atomic_transactions,
            ..Metric::default() // unit: units.atomic_transactions,
        },
        atomic_transactions_per_request: Metric {
            value: values.atomic_transactions_per_request,
            ..Metric::default() // unit: units.atomic_transactions_per_request,
        },
        l2_global_load_bytes: Metric {
            value: values.l2_global_load_bytes,
            ..Metric::default() // unit: units.l2_global_load_bytes,
        },
        l2_local_load_bytes: Metric {
            value: values.l2_local_load_bytes,
            ..Metric::default() // unit: units.l2_local_load_bytes,
        },
        l2_surface_load_bytes: Metric {
            value: values.l2_surface_load_bytes,
            ..Metric::default() // unit: units.l2_surface_load_bytes,
        },
        l2_local_global_store_bytes: Metric {
            value: values.l2_local_global_store_bytes,
            ..Metric::default() // unit: units.l2_local_global_store_bytes,
        },
        l2_global_reduction_bytes: Metric {
            value: values.l2_global_reduction_bytes,
            // unit: units.l2_global_reduction_bytes,
            ..Metric::default()
        },
        l2_global_atomic_store_bytes: Metric {
            value: values.l2_global_atomic_store_bytes,
            ..Metric::default() // unit: units.l2_global_atomic_store_bytes,
        },
        l2_surface_store_bytes: Metric {
            value: values.l2_surface_store_bytes,
            ..Metric::default() // unit: units.l2_surface_store_bytes,
        },
        l2_surface_reduction_bytes: Metric {
            value: values.l2_surface_reduction_bytes,
            ..Metric::default() // unit: units.l2_surface_reduction_bytes,
        },
        l2_surface_atomic_store_bytes: Metric {
            value: values.l2_surface_atomic_store_bytes,
            ..Metric::default() // unit: units.l2_surface_atomic_store_bytes,
        },
        global_load_requests: Metric {
            value: values.global_load_requests,
            ..Metric::default() // unit: units.global_load_requests,
        },
        local_load_requests: Metric {
            value: values.local_load_requests,
            ..Metric::default() // unit: units.local_load_requests,
        },
        surface_load_requests: Metric {
            value: values.surface_load_requests,
            ..Metric::default() // unit: units.surface_load_requests,
        },
        global_store_requests: Metric {
            value: values.global_store_requests,
            ..Metric::default() // unit: units.global_store_requests,
        },
        local_store_requests: Metric {
            value: values.local_store_requests,
            ..Metric::default() // unit: units.local_store_requests,
        },
        surface_store_requests: Metric {
            value: values.surface_store_requests,
            ..Metric::default() // unit: units.surface_store_requests,
        },
        global_atomic_requests: Metric {
            value: values.global_atomic_requests,
            ..Metric::default() // unit: units.global_atomic_requests,
        },
        global_reduction_requests: Metric {
            value: values.global_reduction_requests,
            ..Metric::default() // unit: units.global_reduction_requests,
        },
        surface_atomic_requests: Metric {
            value: values.surface_atomic_requests,
            ..Metric::default() // unit: units.surface_atomic_requests,
        },
        surface_reduction_requests: Metric {
            value: values.surface_reduction_requests,
            ..Metric::default() // unit: units.surface_reduction_requests,
        },
        dram_read_transactions: Metric {
            value: values.dram_read_transactions,
            ..Metric::default() // unit: units.dram_read_transactions,
        },
        dram_write_transactions: Metric {
            value: values.dram_write_transactions,
            ..Metric::default() // unit: units.dram_write_transactions,
        },
        dram_read_throughput: Metric {
            value: values.dram_read_throughput,
            ..Metric::default() // unit: units.dram_read_throughput,
        },
        dram_write_throughput: Metric {
            value: values.dram_write_throughput,
            ..Metric::default() // unit: units.dram_write_throughput,
        },
        dram_write_bytes: Metric {
            value: values.dram_write_bytes,
            ..Metric::default() // unit: units.dram_write_bytes,
        },
        dram_read_bytes: Metric {
            value: values.dram_read_bytes,
            ..Metric::default() // unit: units.dram_read_bytes,
        },
    };
    Ok(ProfilingResult {
        raw: original_log,
        metrics,
    })
}

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
