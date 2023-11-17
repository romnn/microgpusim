mod device;
mod dram;
mod l1_tex;
mod l2;
mod metrics;
mod profiler;
mod scheduler;
mod sm;
mod tex;

use indexmap::IndexMap;
use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashMap;
use std::io::BufRead;
use std::path::Path;
use std::path::PathBuf;

use crate::{Error, JsonError, Metric, ParseError};
pub use metrics::{Float, Metrics};

#[derive(PartialEq, Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Output {
    pub raw_metrics_log: String,
    pub metrics: Vec<Metrics>,
}

macro_rules! optional {
    ($x:expr) => {
        if $x.is_empty() {
            None
        } else {
            Some($x)
        }
    };
}

static NO_PERMISSION_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^==ERROR==\s*ERR_NVGPUCTRPERM\s*-\s*The user does not have permission").unwrap()
});

static PROFILE_RESULT_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^==PROF==\s*Disconnected from process").unwrap());

pub fn seek_to_csv<R>(reader: &mut R) -> Result<csv::Reader<&mut R>, ParseError>
where
    R: std::io::BufRead,
{
    // seek to valid start of csv data
    let mut lines = reader.by_ref().lines();
    for line in &mut lines {
        let Ok(line) = line else { continue };

        if NO_PERMISSION_REGEX.is_match(&line) {
            return Err(ParseError::NoPermission);
        }

        if PROFILE_RESULT_REGEX.is_match(&line) {
            break;
        }
    }

    // upgrade reader to a csv reader and start reading from current position
    let csv_reader = csv::ReaderBuilder::new()
        .flexible(false)
        .from_reader(reader);
    Ok(csv_reader)
}

pub fn parse_nsight_csv<M>(reader: &mut impl std::io::BufRead) -> Result<Vec<M>, ParseError>
where
    M: serde::de::DeserializeOwned,
{
    let mut csv_reader = seek_to_csv(reader)?;
    let mut records = csv_reader.deserialize();

    let mut entries = Vec::new();
    let units: IndexMap<String, String> = records.next().ok_or(ParseError::MissingUnits)??;

    while let Some(values) = records.next().transpose()? {
        assert_eq!(units.len(), values.len());
        let metrics: HashMap<String, Metric<String>> = units
            .iter()
            .zip(values.iter())
            .map(|((unit_metric, unit), (value_metric, value))| {
                assert_eq!(unit_metric, value_metric);
                (
                    unit_metric.clone(),
                    Metric {
                        value: optional!(value).cloned(),
                        unit: optional!(unit).cloned(),
                    },
                )
            })
            .collect();

        {
            let mut metrics: Vec<_> = metrics.clone().into_iter().collect();
            metrics.sort_by_key(|(name, _value)| name.clone());

            for (m, value) in &metrics {
                log::trace!("{m}: {:?}", &value.value);
            }
        }

        // this is kind of hacky..
        let serialized = serde_json::to_string_pretty(&metrics)?;
        let deser = &mut serde_json::Deserializer::from_str(&serialized);
        let metrics: M = serde_path_to_error::deserialize(deser).map_err(|source| {
            let path = source.path().clone();
            ParseError::Json(JsonError {
                source: source.into_inner(),
                values: Some(metrics),
                path: Some(path),
            })
        })?;
        entries.push(metrics);
    }

    Ok(entries)
}

pub const NSIGHT_METRICS: [&str; 113] = [
    // nsight compute for compute <7.0
    "gpu__time_duration",
    "smsp__inst_executed_per_warp",
    "smsp__inst_executed_sum",
    "smsp__inst_issued_sum",
    "smsp__thread_inst_executed_sum",
    "smsp__thread_inst_executed_not_pred_off_sum",
    "sm__active_cycles_avg",
    "sm__inst_issued_sum",
    "sm__inst_executed_sum",
    "sm__inst_executed_avg",
    // dram reads
    "dram__read_sectors",
    // dram writes
    "dram__write_sectors",
    // l2 total accesses
    "lts__request_tex_sectors",
    // l2 reads
    "lts__request_tex_read_sectors",
    // l2 writes
    "lts__request_tex_write_sectors",
    // l2 hit rate
    "lts__request_total_sectors_hitrate_pct",
    // l1 TAG hits
    "tex__t_sectors_hit_global_ld_cached",
    "tex__t_sectors_hit_local_ld_cached",
    // l1 TAG misses
    "tex__t_sectors_miss_global_ld_cached",
    "tex__t_sectors_miss_global_ld_uncached",
    "tex__t_sectors_miss_local_ld_cached",
    "tex__t_sectors_miss_local_ld_uncached",
    "tex__t_sectors_miss_surface_ld",
    // l1 MISS writes
    "tex__m_wr_sectors_atom_red",
    "tex__m_wr_sectors_global_atom",
    "tex__m_wr_sectors_global_nonatom",
    "tex__m_wr_sectors_global_red",
    "tex__m_wr_sectors_local_st",
    "tex__m_wr_sectors_surface_atom",
    "tex__m_wr_sectors_surface_nonatom",
    "tex__m_wr_sectors_surface_red",
    // l1 MISS reads
    "tex__m_rd_sectors_miss_surface_ld",
    "tex__m_rd_sectors_miss_surface_ld_pct",
    "tex__m_rd_sectors_miss_local_ld_uncached_pct",
    "tex__m_rd_sectors_miss_local_ld_uncached",
    "tex__m_rd_sectors_miss_local_ld_cached_pct",
    "tex__m_rd_sectors_miss_local_ld_cached",
    "tex__m_rd_sectors_miss_global_ld_uncached_pct",
    "tex__m_rd_sectors_miss_global_ld_uncached",
    "tex__m_rd_sectors_miss_global_ld_cached_pct",
    "tex__m_rd_sectors_miss_global_ld_cached",
    // l1 global hitrate
    "tex__hitrate_pct",
    // bytes
    "tex__t_bytes_miss_global_ld",
    "tex__t_bytes_miss_global_ld_cached",
    "tex__t_bytes_miss_global_ld_uncached",
    "tex__t_bytes_miss_local_ld",
    "tex__t_bytes_miss_local_ld_cached",
    "tex__t_bytes_miss_local_ld_uncached",
    "tex__t_bytes_miss_surface_ld",
    "tex__t_bytes_miss_surface_ld_cached",
    "tex__t_bytes_miss_surface_ld_uncached",
    // nsight compute for compute 7.0+
    "gpc__cycles_elapsed.avg",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_miss.sum",
    "lts__t_sector_op_read_hit_rate.pct", // ?
    "lts__t_sector_op_write_hit_rate.pct",
    "lts__t_sectors_srcunit_tex_op_read.sum",
    "lts__t_sectors_srcunit_tex_op_write.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
    "lts__t_sectors_srcunit_tex_op_write_lookup_hit.sum",
    "lts__t_sectors_srcunit_tex_op_read.sum.per_second",
    "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
    "lts__t_sectors_srcunit_tex_op_write_lookup_miss.sum",
    "dram__sectors_read.sum",
    "dram__sectors_write.sum",
    "dram__bytes_read.sum",
    "smsp__inst_executed.sum",
    "smsp__thread_inst_executed_per_inst_executed.ratio",
    "smsp__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "idc__requests.sum",
    "idc__requests_lookup_hit.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_alu_cycles_active.sum",
    "sm__pipe_fma_cycles_active.sum",
    "sm__pipe_fp64_cycles_active.sum",
    "sm__pipe_shared_cycles_active.sum",
    "sm__pipe_tensor_cycles_active.sum",
    "sm__pipe_tensor_op_hmma_cycles_active.sum",
    "sm__cycles_elapsed.sum",
    "sm__cycles_active.sum",
    "sm__cycles_active.avg",
    "sm__cycles_elapsed.avg",
    "sm__sass_thread_inst_executed_op_integer_pred_on.sum",
    "sm__sass_thread_inst_executed_ops_dadd_dmul_dfma_pred_on.sum",
    "sm__sass_thread_inst_executed_ops_fadd_fmul_ffma_pred_on.sum",
    "sm__sass_thread_inst_executed_ops_hadd_hmul_hfma_pred_on.sum",
    "sm__inst_executed.sum",
    "sm__inst_executed_pipe_alu.sum",
    "sm__inst_executed_pipe_fma.sum",
    "sm__inst_executed_pipe_fp16.sum",
    "sm__inst_executed_pipe_fp64.sum",
    "sm__inst_executed_pipe_tensor.sum",
    "sm__inst_executed_pipe_tex.sum",
    "sm__inst_executed_pipe_xu.sum",
    "sm__inst_executed_pipe_lsu.sum",
    "sm__sass_thread_inst_executed_op_fp16_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fp32_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fp64_pred_on.sum",
    "sm__sass_thread_inst_executed_op_dmul_pred_on.sum",
    "sm__sass_thread_inst_executed_op_dfma_pred_on.sum",
    "sm__sass_thread_inst_executed.sum",
    "sm__sass_inst_executed_op_shared_st.sum",
    "sm__sass_inst_executed_op_shared_ld.sum",
    "sm__sass_inst_executed_op_memory_128b.sum",
    "sm__sass_inst_executed_op_memory_64b.sum",
    "sm__sass_inst_executed_op_memory_32b.sum",
    "sm__sass_inst_executed_op_memory_16b.sum",
    "sm__sass_inst_executed_op_memory_8b.sum",
];

pub fn build_nsight_args(executable: &Path, args: &[String]) -> Result<Vec<String>, Error> {
    let metrics = NSIGHT_METRICS.join(",");
    let mut cmd_args: Vec<String> = [
        "--metrics",
        &metrics,
        "--csv",
        "--page",
        "raw",
        "--kernel-regex-base",
        "mangled",
        "--target-processes",
        "all",
        "--units",
        "base",
        "--fp",
    ]
    .into_iter()
    .map(str::to_string)
    .collect();

    cmd_args.extend([executable.to_string_lossy().to_string()]);
    cmd_args.extend(args.into_iter().cloned());
    Ok(cmd_args)
}

pub async fn profile_all_metrics<A>(
    nsight: impl AsRef<Path>,
    executable: impl AsRef<Path>,
    args: A,
) -> Result<(String, Vec<Metrics>), Error>
where
    A: IntoIterator,
    <A as IntoIterator>::Item: AsRef<std::ffi::OsStr>,
{
    let mut cmd = async_process::Command::new(nsight.as_ref());
    let args: Vec<String> = args
        .into_iter()
        .map(|arg| arg.as_ref().to_string_lossy().to_string())
        .collect();

    let cmd_args = build_nsight_args(executable.as_ref(), &*args)?;
    cmd.args(&cmd_args);

    log::debug!(
        "profile command: {} {}",
        nsight.as_ref().display(),
        cmd_args.join(" ")
    );

    let result = cmd.output().await?;

    let stdout = utils::decode_utf8!(result.stdout);
    let stderr = utils::decode_utf8!(result.stderr);
    log::debug!("profile stdout: {}", stdout);
    log::debug!("profile stderr: {}", stderr);

    if !result.status.success() {
        return Err(Error::Command {
            raw_log: "".to_string(),

            source: utils::CommandError::new(&cmd, result),
        });
    }

    let mut log_reader = std::io::Cursor::new(&stdout);
    match parse_nsight_csv(&mut log_reader) {
        Err(source) => Err(Error::Parse {
            raw_log: stdout,
            source,
        }),
        Ok(metrics) => Ok((stdout, metrics)),
    }
}

#[derive(Debug, Clone)]
pub struct Options {
    pub nsight_path: Option<PathBuf>,
}

/// Profile test application using `nv-nsight-cu-cli` profiler.
///
/// Note: `nv-nsight-cu-cli` is not compatible with newer devices.
///
/// # Errors
/// - When creating temp dir fails.
/// - When profiling fails.
/// - When application fails.
pub async fn nsight<A>(
    executable: impl AsRef<Path>,
    args: A,
    options: &Options,
) -> Result<Output, Error>
where
    A: Clone + IntoIterator,
    <A as IntoIterator>::Item: AsRef<std::ffi::OsStr>,
{
    let nsight: Result<_, Error> = options
        .nsight_path
        .clone()
        .map(|path| {
            if path.is_file() {
                path
            } else {
                path.join("nv-nsight-cu-cli")
            }
        })
        .or_else(|| which::which("nv-nsight-cu-cli").ok())
        .ok_or_else(|| Error::MissingProfiler("nv-nsight-cu-cli".into()));

    let nsight = nsight.or_else(|_| {
        let cuda = utils::find_cuda().ok_or(Error::MissingCUDA)?;
        Ok::<_, Error>(cuda.join("bin/nv-nsight-cu-cli"))
    })?;
    let nsight = nsight
        .canonicalize()
        .map_err(|_| Error::MissingProfiler(nsight))?;

    let executable = executable
        .as_ref()
        .canonicalize()
        .map_err(|_| Error::MissingExecutable(executable.as_ref().into()))?;

    let (raw_metrics_log, metrics) =
        profile_all_metrics(&nsight, &executable, args.clone()).await?;

    Ok(Output {
        raw_metrics_log,
        metrics,
    })
}

#[cfg(test)]
mod tests {
    use super::{parse_nsight_csv, Float, Metric};
    use color_eyre::eyre;
    use similar_asserts as diff;
    use std::io::Cursor;

    #[test]
    fn parse_all_metrics() -> eyre::Result<()> {
        let bytes = include_bytes!("../../tests/nsight_vectoradd_100_32_metrics_all.txt");
        let log = String::from_utf8_lossy(bytes).to_string();
        println!("{}", &log);
        let mut log_reader = Cursor::new(bytes);
        let metrics: Vec<super::Metrics> = parse_nsight_csv(&mut log_reader)?;
        dbg!(&metrics);
        diff::assert_eq!(metrics.len(), 3);

        diff::assert_eq!(
            metrics[0].device.attribute_display_name,
            Some(Metric::new("NVIDIA GeForce GTX 1080".to_string(), None))
        );
        diff::assert_eq!(
            metrics[0].kernel_name,
            Some(Metric::new("vecAdd".to_string(), None))
        );
        diff::assert_eq!(metrics[0].context, Some(Metric::new(1, None)));
        diff::assert_eq!(metrics[0].stream, Some(Metric::new(7, None)));
        diff::assert_eq!(
            metrics[0].device.attribute_clock_rate,
            Some(Metric::new(Float(1_759_000.0), None))
        );
        diff::assert_eq!(
            metrics[0].dram.write_bytes_per_sec,
            Some(Metric::new(
                Float(141_176_470.59),
                "byte/second".to_string()
            ))
        );
        diff::assert_eq!(
            metrics[0].gpu_time_duration,
            Some(Metric::new(Float(2_720.00), "nsecond".to_string()))
        );
        diff::assert_eq!(
            metrics[0].gpc_elapsed_cycles_max,
            Some(Metric::new(Float(5_774.00), "cycle".to_string()))
        );
        diff::assert_eq!(
            metrics[0].sm_scheduler.maximum_warps_avg_per_active_cycle,
            Some(Metric::new(Float(16.00), "warp/cycle".to_string()))
        );
        Ok(())
    }
}
