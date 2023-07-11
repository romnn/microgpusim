mod metrics;

use async_process::Command;
use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashMap;
use std::io::{BufRead, Read, Seek};
use std::path::Path;

use crate::{Error, Metric, ParseError};
pub use metrics::Metrics;

pub type ProfilingResult = super::ProfilingResult<Metrics>;

macro_rules! optional {
    ($x:expr) => {
        if $x.is_empty() {
            None
        } else {
            Some($x)
        }
    };
}

static NO_PERMISSION_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"user does not have permission").unwrap());

static PROFILE_RESULT_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^==\d*==\s*Profiling result:\s*$").unwrap());

pub fn parse_nvprof_csv(reader: &mut impl std::io::BufRead) -> Result<Metrics, ParseError> {
    // seek to valid start of csv data
    let mut lines = reader.by_ref().lines();
    for line in &mut lines {
        let Ok(line) = line else {
            continue
        };

        if NO_PERMISSION_REGEX.is_match(&line) {
            return Err(ParseError::NoPermission);
        }

        if PROFILE_RESULT_REGEX.is_match(&line) {
            break;
        }
    }

    // upgrade reader to a csv reader, keeping the current position
    let mut csv_reader = csv::ReaderBuilder::new()
        .flexible(false)
        .from_reader(reader);

    let mut records = csv_reader.deserialize();

    let mut metrics: HashMap<String, Metric<String>> = HashMap::new();
    let units: HashMap<String, String> = records.next().ok_or(ParseError::MissingUnits)??;
    let values: HashMap<String, String> = records.next().ok_or(ParseError::MissingMetrics)??;
    assert_eq!(units.len(), values.len());

    for (metric, unit) in units {
        metrics.entry(metric).or_default().unit = optional!(unit);
    }
    for (metric, value) in values {
        metrics.entry(metric).or_default().value = optional!(value);
    }

    // this is kind of hacky..
    let metrics = serde_json::to_string(&metrics)?;
    let metrics: Metrics = serde_json::from_str(&metrics)?;
    Ok(metrics)
}

#[derive(Debug, Clone)]
pub struct Options {}

/// Profile test application using nvbprof profiler.
///
/// Note: The nvbprof compiler is not recommended for newer devices.
///
/// # Errors
/// - When creating temp dir fails.
/// - When profiling fails.
/// - When application fails.
#[allow(clippy::too_many_lines)]
pub async fn nvprof<A>(
    executable: impl AsRef<Path>,
    args: A,
    _options: &Options,
) -> Result<ProfilingResult, Error>
where
    A: IntoIterator,
    <A as IntoIterator>::Item: AsRef<std::ffi::OsStr>,
{
    let tmp_dir = tempfile::tempdir()?;
    let log_file_path = tmp_dir.path().join("log_file.csv");

    let nvprof = which::which("nvprof").map_err(|_| Error::MissingProfiler("nvprof".into()));
    let nvprof = nvprof.or_else(|_| {
        let cuda = utils::find_cuda().ok_or(Error::MissingCUDA)?;
        Ok::<_, Error>(cuda.join("bin/nvprof"))
    })?;
    let nvprof = nvprof
        .canonicalize()
        .map_err(|_| Error::MissingProfiler(nvprof))?;

    let executable = executable
        .as_ref()
        .canonicalize()
        .map_err(|_| Error::MissingExecutable(executable.as_ref().into()))?;

    let mut cmd = Command::new(nvprof);
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
    .arg(&executable)
    .args(args.into_iter());

    let result = cmd.output().await?;
    if !result.status.success() {
        return Err(Error::Command(utils::CommandError::new(&cmd, result)));
    }

    let log_file = std::fs::OpenOptions::new()
        .read(true)
        .open(&log_file_path)?;

    let mut log_reader = std::io::BufReader::new(log_file);

    let mut original_log = String::new();
    log_reader.read_to_string(&mut original_log)?;
    log_reader.rewind()?;

    let metrics = parse_nvprof_csv(&mut log_reader).map_err(|source| Error::Parse {
        raw_log: original_log.clone(),
        source,
    })?;
    Ok(ProfilingResult {
        raw: original_log,
        metrics,
    })
}

#[cfg(test)]
mod tests {
    use super::{parse_nvprof_csv, Metric};
    use color_eyre::eyre;
    use std::io::Cursor;

    #[test]
    fn parse_all_metrics() -> eyre::Result<()> {
        let bytes = include_bytes!("../../tests/nvprof_vectoradd_100_32_metrics_all.txt");
        let log = String::from_utf8_lossy(bytes).to_string();
        dbg!(&log);
        let mut log_reader = Cursor::new(bytes);
        let metrics = parse_nvprof_csv(&mut log_reader)?;
        dbg!(&metrics);
        assert_eq!(
            metrics.device,
            Metric::new("NVIDIA GeForce GTX 1080 (0)".to_string(), None)
        );
        assert_eq!(
            metrics.kernel,
            Metric::new("_Z6vecAddIfEvPT_S1_S1_i".to_string(), None)
        );
        assert_eq!(metrics.context, Metric::new(1, None));
        assert_eq!(metrics.stream, Metric::new(7, None));
        assert_eq!(metrics.dram_write_bytes, Metric::new(0, None));
        assert_eq!(metrics.dram_read_bytes, Metric::new(7136, None));
        assert_eq!(metrics.dram_read_transactions, Metric::new(223, None));
        assert_eq!(metrics.dram_write_transactions, Metric::new(0, None));
        assert_eq!(metrics.l2_read_transactions, Metric::new(66, None));
        assert_eq!(metrics.l2_write_transactions, Metric::new(26, None));
        Ok(())
    }
}
