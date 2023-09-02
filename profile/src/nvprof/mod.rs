mod metrics;

use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashMap;
use std::io::{BufRead, Read};
use std::path::Path;

use crate::{Error, Metric, ParseError};
pub use metrics::{Command, Metrics};

#[derive(PartialEq, Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Output {
    pub raw_metrics_log: String,
    pub raw_commands_log: String,
    pub metrics: Vec<Metrics>,
    pub commands: Vec<Command>,
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

static NO_PERMISSION_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"user does not have permission").unwrap());

static PROFILE_RESULT_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^==\d*==\s*Profiling result:\s*$").unwrap());

pub fn seek_to_csv<R>(reader: &mut R) -> Result<csv::Reader<&mut R>, ParseError>
where
    R: std::io::BufRead,
{
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

    // upgrade reader to a csv reader and start reading from current position
    let csv_reader = csv::ReaderBuilder::new()
        .flexible(false)
        .from_reader(reader);
    Ok(csv_reader)
}

pub fn parse_nvprof_csv<M>(reader: &mut impl std::io::BufRead) -> Result<Vec<M>, ParseError>
where
    M: serde::de::DeserializeOwned,
{
    let mut csv_reader = seek_to_csv(reader)?;
    let mut records = csv_reader.deserialize();

    use indexmap::IndexMap;
    let mut entries = Vec::new();
    let units: IndexMap<String, String> = records.next().ok_or(ParseError::MissingUnits)??;

    while let Some(values) = records.next().transpose()? {
        assert_eq!(units.len(), values.len());
        let metrics: HashMap<String, Metric<String>> = units
            .iter()
            .zip(values.iter())
            .flat_map(|((unit_metric, unit), (value_metric, value))| {
                assert_eq!(unit_metric, value_metric);
                Some((
                    unit_metric.clone(),
                    Metric {
                        value: optional!(value).cloned(),
                        unit: optional!(unit).cloned(),
                    },
                ))
            })
            .collect();

        {
            let mut metrics: Vec<_> = metrics.clone().into_iter().collect();
            metrics.sort_by_key(|(name, _value)| name.clone());

            for (m, value) in metrics.iter() {
                log::trace!("{m}: {:?}", &value.value);
            }
        }

        // this is kind of hacky..
        let serialized = serde_json::to_string(&metrics)?;
        let deser = &mut serde_json::Deserializer::from_str(&serialized);
        let metrics: M = serde_path_to_error::deserialize(deser).map_err(|source| {
            let path = source.path().to_string();
            ParseError::JSON {
                source: source.into_inner(),
                values: Some(metrics),
                path: Some(path),
            }
        })?;
        entries.push(metrics);
    }

    Ok(entries)
}

#[derive(Debug, Clone)]
pub struct Options {}

pub async fn profile_all_metrics<A>(
    nvprof: impl AsRef<Path>,
    executable: impl AsRef<Path>,
    args: A,
    log_file_path: impl AsRef<Path>,
) -> Result<(String, Vec<Metrics>), Error>
where
    A: IntoIterator,
    <A as IntoIterator>::Item: AsRef<std::ffi::OsStr>,
{
    let mut cmd = async_process::Command::new(nvprof.as_ref());
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
    .arg(log_file_path.as_ref())
    .arg(executable.as_ref())
    .args(args.into_iter());

    let result = cmd.output().await?;
    if !result.status.success() {
        return Err(Error::Command(utils::CommandError::new(&cmd, result)));
    }

    let log_file = std::fs::OpenOptions::new()
        .read(true)
        .open(&log_file_path)?;

    let mut log_reader = std::io::BufReader::new(log_file);

    let mut raw_log = String::new();
    log_reader.read_to_string(&mut raw_log)?;

    let mut log_reader = std::io::Cursor::new(&raw_log);
    match parse_nvprof_csv(&mut log_reader) {
        Err(source) => Err(Error::Parse { raw_log, source }),
        // Ok(metrics) if metrics.len() != 1 => Err(Error::Parse {
        //     raw_log,
        //     source: ParseError::MissingMetrics,
        // }),
        Ok(metrics) => Ok((raw_log, metrics)),
    }
}

pub async fn profile_commands<A>(
    nvprof: impl AsRef<Path>,
    executable: impl AsRef<Path>,
    args: A,
    log_file_path: impl AsRef<Path>,
) -> Result<(String, Vec<Command>), Error>
where
    A: IntoIterator,
    <A as IntoIterator>::Item: AsRef<std::ffi::OsStr>,
{
    let mut cmd = async_process::Command::new(nvprof.as_ref());
    cmd.args([
        "--unified-memory-profiling",
        "off",
        "--concurrent-kernels",
        "off",
        "--print-gpu-trace",
        "-u",
        "us",
        "--demangling",
        "off",
        "--csv",
        "--log-file",
    ])
    .arg(log_file_path.as_ref())
    .arg(executable.as_ref())
    .args(args.into_iter());

    let result = cmd.output().await?;
    if !result.status.success() {
        return Err(Error::Command(utils::CommandError::new(&cmd, result)));
    }

    let log_file = std::fs::OpenOptions::new()
        .read(true)
        .open(&log_file_path)?;

    let mut log_reader = std::io::BufReader::new(log_file);

    let mut raw_log = String::new();
    log_reader.read_to_string(&mut raw_log)?;

    let mut log_reader = std::io::Cursor::new(&raw_log);
    match parse_nvprof_csv(&mut log_reader) {
        Err(source) => Err(Error::Parse { raw_log, source }),
        Ok(commands) => Ok((raw_log, commands)),
    }
}

/// Profile test application using nvprof profiler.
///
/// Note: `nvprof` is not compatible with newer devices.
///
/// # Errors
/// - When creating temp dir fails.
/// - When profiling fails.
/// - When application fails.
pub async fn nvprof<A>(
    executable: impl AsRef<Path>,
    args: A,
    _options: &Options,
) -> Result<Output, Error>
where
    A: Clone + IntoIterator,
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

    let (raw_metrics_log, metrics) =
        profile_all_metrics(&nvprof, &executable, args.clone(), &log_file_path).await?;

    let (raw_commands_log, commands) =
        profile_commands(&nvprof, &executable, args, &log_file_path).await?;

    Ok(Output {
        raw_metrics_log,
        raw_commands_log,
        metrics,
        commands,
    })
}

#[cfg(test)]
mod tests {
    use super::{parse_nvprof_csv, Metric};
    use color_eyre::eyre;
    use similar_asserts as diff;
    use std::io::Cursor;

    #[test]
    fn parse_all_metrics() -> eyre::Result<()> {
        let bytes = include_bytes!("../../tests/nvprof_vectoradd_100_32_metrics_all.txt");
        let log = String::from_utf8_lossy(bytes).to_string();
        dbg!(&log);
        let mut log_reader = Cursor::new(bytes);
        let mut metrics: Vec<super::Metrics> = parse_nvprof_csv(&mut log_reader)?;
        diff::assert_eq!(metrics.len(), 1);
        let metrics = metrics.remove(0);
        dbg!(&metrics);
        diff::assert_eq!(
            metrics.device,
            Metric::new("NVIDIA GeForce GTX 1080 (0)".to_string(), None)
        );
        diff::assert_eq!(
            metrics.kernel,
            Metric::new("_Z6vecAddIfEvPT_S1_S1_i".to_string(), None)
        );
        diff::assert_eq!(metrics.context, Metric::new(1, None));
        diff::assert_eq!(metrics.stream, Metric::new(7, None));
        diff::assert_eq!(metrics.dram_write_bytes, Metric::new(0, None));
        diff::assert_eq!(metrics.dram_read_bytes, Metric::new(7136, None));
        diff::assert_eq!(metrics.dram_read_transactions, Metric::new(223, None));
        diff::assert_eq!(metrics.dram_write_transactions, Metric::new(0, None));
        diff::assert_eq!(metrics.l2_read_transactions, Metric::new(66, None));
        diff::assert_eq!(metrics.l2_write_transactions, Metric::new(26, None));
        Ok(())
    }

    #[test]
    fn parse_commands() -> eyre::Result<()> {
        use super::metrics::Command;
        let bytes = include_bytes!("../../tests/nvprof_vectoradd_100_32_commands.txt");
        let log = String::from_utf8_lossy(bytes).to_string();
        dbg!(&log);
        let mut log_reader = Cursor::new(bytes);
        let metrics: Vec<Command> = parse_nvprof_csv(&mut log_reader)?;
        dbg!(&metrics);
        diff::assert_eq!(metrics.len(), 5);

        diff::assert_eq!(
            have: metrics[0],
            want: Command {
                start: Metric::new(245729.104000, "us".to_string()),
                duration: Metric::new(1.088000, "us".to_string()),
                grid_x: Metric::new(None, None),
                grid_y: Metric::new(None, None),
                grid_z: Metric::new(None, None),
                block_x: Metric::new(None, None),
                block_y: Metric::new(None, None),
                block_z: Metric::new(None, None),
                registers_per_thread: Metric::new(None, None),
                static_shared_memory: Metric::new(None, "B".to_string()),
                dynamic_shared_memory: Metric::new(None, "B".to_string()),
                size: Metric::new(400.0, "B".to_string()),
                throughput: Metric::new(350.615557, "MB/s".to_string()),
                src_mem_type: Metric::new("Pageable".to_string(), None),
                dest_mem_type: Metric::new("Device".to_string(), None),
                device: Metric::new("NVIDIA GeForce GTX 1080 (0)".to_string(), None),
                context: Metric::new(1, None),
                stream: Metric::new(7, None),
                name: Metric::new("[CUDA memcpy HtoD]".to_string(), None),
                correlation_id: Metric::new(117, None),
            },
        );
        diff::assert_eq!(
            have: metrics[3],
            want: Command {
                start: Metric::new(245767.824000, "us".to_string()),
                duration: Metric::new(3.264000, "us".to_string()),
                grid_x: Metric::new(1, None),
                grid_y: Metric::new(1, None),
                grid_z: Metric::new(1, None),
                block_x: Metric::new(1024, None),
                block_y: Metric::new(1, None),
                block_z: Metric::new(1, None),
                registers_per_thread: Metric::new(8, None),
                static_shared_memory: Metric::new(0.0, "B".to_string()),
                dynamic_shared_memory: Metric::new(0.0, "B".to_string()),
                size: Metric::new(None, "B".to_string()),
                throughput: Metric::new(None, "MB/s".to_string()),
                src_mem_type: Metric::new(None, None),
                dest_mem_type: Metric::new(None, None),
                device: Metric::new("NVIDIA GeForce GTX 1080 (0)".to_string(), None),
                context: Metric::new(1, None),
                stream: Metric::new(7, None),
                name: Metric::new("_Z6vecAddIfEvPT_S1_S1_i".to_string(), None),
                correlation_id: Metric::new(123, None),
            },
        );
        Ok(())
    }
}
