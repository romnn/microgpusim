mod metrics;

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
        let serialized = serde_json::to_string(&metrics)?;
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

pub async fn profile_all_metrics<A>(
    nsight: impl AsRef<Path>,
    executable: impl AsRef<Path>,
    args: A,
) -> Result<(String, Vec<Metrics>), Error>
where
    A: IntoIterator,
    <A as IntoIterator>::Item: AsRef<std::ffi::OsStr>,
{
    let mut cmd_args = vec![
        "--csv",
        "--page",
        "raw",
        "--target-processes",
        "all",
        "--units",
        "base",
        "--fp",
        executable.as_ref().to_str().unwrap(),
    ];
    let args: Vec<String> = args
        .into_iter()
        .map(|arg| arg.as_ref().to_string_lossy().to_string())
        .collect();
    cmd_args.extend(args.iter().map(String::as_str));
    let mut cmd = async_process::Command::new(nsight.as_ref());
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
            metrics[0].device_attribute_display_name,
            Metric::new("NVIDIA GeForce GTX 1080".to_string(), None)
        );
        diff::assert_eq!(
            metrics[0].kernel_name,
            Metric::new("vecAdd".to_string(), None)
        );
        diff::assert_eq!(metrics[0].context, Metric::new(1, None));
        diff::assert_eq!(metrics[0].stream, Metric::new(7, None));
        diff::assert_eq!(
            metrics[0].device_attribute_clock_rate,
            Metric::new(Float(1_759_000.0), None)
        );
        diff::assert_eq!(
            metrics[0].dram_write_bytes_per_sec,
            Metric::new(Float(141_176_470.59), "byte/second".to_string())
        );
        diff::assert_eq!(
            metrics[0].gpu_time_duration,
            Metric::new(Float(2_720.00), "nsecond".to_string())
        );
        diff::assert_eq!(
            metrics[0].gpc_elapsed_cycles_max,
            Metric::new(Float(5_774.00), "cycle".to_string())
        );
        diff::assert_eq!(
            metrics[0].smsp_maximum_warps_avg_per_active_cycle,
            Metric::new(Float(16.00), "warp/cycle".to_string())
        );
        Ok(())
    }
}
