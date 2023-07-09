mod metrics;

use async_process::Command;
use regex::Regex;
use std::cell::OnceCell;
use std::collections::HashMap;
use std::io::{BufRead, Read, Seek};
use std::path::{Path, PathBuf};

use crate::{CommandError, Error, Metric, ParseError};
pub use metrics::NvprofMetrics;

pub type ProfilingResult = super::ProfilingResult<NvprofMetrics>;

macro_rules! optional {
    ($x:expr) => {
        if $x.is_empty() {
            None
        } else {
            Some($x)
        }
    };
}

pub fn parse_nvprof_csv(reader: &mut impl std::io::BufRead) -> Result<NvprofMetrics, ParseError> {
    // seek to valid start of csv data
    let mut lines = reader.by_ref().lines();
    for line in &mut lines {
        let Ok(line) = line else {
            continue
        };

        const NO_PERMISSION_REGEX: OnceCell<Regex> = OnceCell::new();
        if NO_PERMISSION_REGEX
            .get_or_init(|| Regex::new(r"user does not have permission").unwrap())
            .is_match(&line)
        {
            return Err(ParseError::NoPermission);
        }

        const PROFILE_RESULT_REGEX: OnceCell<Regex> = OnceCell::new();
        if PROFILE_RESULT_REGEX
            .get_or_init(|| Regex::new(r"^==\d*==\s*Profiling result:\s*$").unwrap())
            .is_match(&line)
        {
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

    for (metric, unit) in units.into_iter() {
        metrics.entry(metric).or_default().unit = optional!(unit);
    }
    for (metric, value) in values.into_iter() {
        metrics.entry(metric).or_default().value = optional!(value);
    }

    // this is kind of hacky..
    let metrics = serde_json::to_string(&metrics)?;
    let metrics: NvprofMetrics = serde_json::from_str(&metrics)?;
    Ok(metrics)
}

/// Profile test application using nvbprof profiler.
///
/// Note: The nvbprof compiler is not recommended for newer devices.
///
/// # Errors
/// - When creating temp dir fails.
/// - When profiling fails.
/// - When application fails.
#[allow(clippy::too_many_lines)]
pub async fn nvprof<P, A>(executable: P, args: A) -> Result<ProfilingResult, Error>
where
    P: AsRef<Path>,
    A: IntoIterator,
    <A as IntoIterator>::Item: AsRef<std::ffi::OsStr>,
{
    let tmp_dir = tempfile::tempdir()?;
    let log_file_path = tmp_dir.path().join("log_file.csv");

    let nvprof = which::which("nvprof").map_err(|_| Error::MissingProfiler("nvprof".to_string()));
    let nvprof = nvprof.or_else(|_| {
        let cuda = utils::find_cuda().ok_or(Error::MissingCUDA)?;
        Ok::<_, Error>(cuda.join("bin/nvprof"))
    })?;
    let nvprof = nvprof
        .canonicalize()
        .map_err(|_| Error::MissingProfiler(nvprof.to_string_lossy().to_string()))?;

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
    .arg(executable.as_ref())
    .args(args.into_iter());

    let result = cmd.output().await?;
    if !result.status.success() {
        return Err(Error::Command(CommandError {
            command: format!("{:?}", cmd),
            log: std::fs::read_to_string(&log_file_path).ok(),
            output: result,
        }));
    }

    let log_file = std::fs::OpenOptions::new()
        .read(true)
        .open(&log_file_path)?;

    let mut log_reader = std::io::BufReader::new(log_file);

    let mut original_log = String::new();
    log_reader.read_to_string(&mut original_log)?;
    log_reader.rewind()?;
    // println!("{original_log}");

    if let Ok(keep_log_file_path) = std::env::var("KEEP_LOG_FILE_PATH").map(PathBuf::from) {
        use std::io::Write;

        // println!("writing log to {}", keep_log_file_path.display());
        if let Some(parent) = keep_log_file_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let mut keep_log_file = std::fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(&keep_log_file_path)?;
        keep_log_file.write_all(original_log.as_bytes())?;
    }

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
    // use pretty_assertions::assert_eq as diff_assert_eq;
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
