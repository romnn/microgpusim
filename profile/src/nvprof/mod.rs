mod metrics;

use async_process::Command;
use regex::Regex;
use std::cell::OnceCell;
use std::collections::HashMap;
use std::io::{BufRead, Read, Seek};
use std::path::{Path, PathBuf};

use crate::{CommandError, Error, Metric};
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

pub fn parse_nvprof_csv(reader: &mut impl std::io::BufRead) -> Result<NvprofMetrics, Error> {
    // seek to valid start of csv data
    let mut lines = reader.by_ref().lines();
    for line in &mut lines {
        let Ok(line) = line else {
            continue
        };
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
    let units: HashMap<String, String> = records.next().ok_or(Error::MissingUnits)??;
    let values: HashMap<String, String> = records.next().ok_or(Error::MissingUnits)??;
    assert_eq!(units.len(), values.len());

    for (metric, unit) in units.into_iter() {
        metrics.entry(metric).or_default().unit = optional!(unit);
    }
    for (metric, value) in values.into_iter() {
        metrics.entry(metric).or_default().value = optional!(value);
    }
    // dbg!(&metrics);
    // println!("{}", serde_json::to_string_pretty(&metrics).unwrap());

    let temp = serde_json::to_string(&metrics).unwrap();
    let back: NvprofMetrics = serde_json::from_str(&temp).unwrap();
    // let metrics = serde_json::to_value(metrics).unwrap();
    // let back: NvprofMetrics = serde_json::from_value(metrics).unwrap();
    // let back = NvprofMetrics::deserialize(metrics).unwrap();

    // dbg!(&units.iter().cloned().sorted());
    // dbg!(&values.iter().cloned().sorted());

    // let units: NvprofAllColumns = records.next().ok_or(Error::MissingUnits)??;
    // ignore units for now
    // let _ = records.next().ok_or(Error::MissingUnits)?;
    // let values: NvprofAllColumns = records.next().ok_or(Error::MissingMetrics)??;

    // NvprofMetrics

    Ok(back)
    // Ok(NvprofMetrics::default())
    // dbg!(&units);
    // dbg!(&values);
    // Ok(NvprofMetrics {
    //     Device: Metric {
    //         value: values.Device,
    //         ..Metric::default() // unit: units.Device,
    //     },
    //     Context: Metric {
    //         value: values.Context,
    //         ..Metric::default() // unit: units.Context,
    //     },
    //     Stream: Metric {
    //         value: values.Stream,
    //         ..Metric::default() // unit: units.Stream,
    //     },
    //     Kernel: Metric {
    //         value: values.Kernel,
    //         ..Metric::default() // unit: units.Kernel,
    //     },
    //     Correlation_ID: Metric {
    //         value: values.Correlation_ID,
    //         ..Metric::default() // unit: units.Correlation_ID,
    //     },
    //     shared_load_transactions_per_request: Metric {
    //         value: values.shared_load_transactions_per_request,
    //         ..Metric::default() // unit: units.shared_load_transactions_per_request,
    //     },
    //     shared_store_transactions_per_request: Metric {
    //         value: values.shared_store_transactions_per_request,
    //         ..Metric::default() // unit: units.shared_store_transactions_per_request,
    //     },
    //     local_load_transactions_per_request: Metric {
    //         value: values.local_load_transactions_per_request,
    //         ..Metric::default() // unit: units.local_load_transactions_per_request,
    //     },
    //     local_store_transactions_per_request: Metric {
    //         value: values.local_store_transactions_per_request,
    //         ..Metric::default() // unit: units.local_store_transactions_per_request,
    //     },
    //     gld_transactions_per_request: Metric {
    //         value: values.gld_transactions_per_request,
    //         ..Metric::default() // unit: units.gld_transactions_per_request,
    //     },
    //     gst_transactions_per_request: Metric {
    //         value: values.gst_transactions_per_request,
    //         ..Metric::default() // unit: units.gst_transactions_per_request,
    //     },
    //     shared_store_transactions: Metric {
    //         value: values.shared_store_transactions,
    //         ..Metric::default() // unit: units.shared_store_transactions,
    //     },
    //     shared_load_transactions: Metric {
    //         value: values.shared_load_transactions,
    //         ..Metric::default() // unit: units.shared_load_transactions,
    //     },
    //     local_load_transactions: Metric {
    //         value: values.local_load_transactions,
    //         ..Metric::default() // unit: units.local_load_transactions,
    //     },
    //     local_store_transactions: Metric {
    //         value: values.local_store_transactions,
    //         ..Metric::default() // unit: units.local_store_transactions,
    //     },
    //     gld_transactions: Metric {
    //         value: values.gld_transactions,
    //         ..Metric::default() // unit: units.gld_transactions,
    //     },
    //     gst_transactions: Metric {
    //         value: values.gst_transactions,
    //         ..Metric::default() // unit: units.gst_transactions,
    //     },
    //     sysmem_read_transactions: Metric {
    //         value: values.sysmem_read_transactions,
    //         ..Metric::default() // unit: units.sysmem_read_transactions,
    //     },
    //     sysmem_write_transactions: Metric {
    //         value: values.sysmem_write_transactions,
    //         ..Metric::default() // unit: units.sysmem_write_transactions,
    //     },
    //     l2_read_transactions: Metric {
    //         value: values.l2_read_transactions,
    //         ..Metric::default() // unit: units.l2_read_transactions,
    //     },
    //     l2_write_transactions: Metric {
    //         value: values.l2_write_transactions,
    //         ..Metric::default() // unit: units.l2_write_transactions,
    //     },
    //     atomic_transactions: Metric {
    //         value: values.atomic_transactions,
    //         ..Metric::default() // unit: units.atomic_transactions,
    //     },
    //     atomic_transactions_per_request: Metric {
    //         value: values.atomic_transactions_per_request,
    //         ..Metric::default() // unit: units.atomic_transactions_per_request,
    //     },
    //     l2_global_load_bytes: Metric {
    //         value: values.l2_global_load_bytes,
    //         ..Metric::default() // unit: units.l2_global_load_bytes,
    //     },
    //     l2_local_load_bytes: Metric {
    //         value: values.l2_local_load_bytes,
    //         ..Metric::default() // unit: units.l2_local_load_bytes,
    //     },
    //     l2_surface_load_bytes: Metric {
    //         value: values.l2_surface_load_bytes,
    //         ..Metric::default() // unit: units.l2_surface_load_bytes,
    //     },
    //     l2_local_global_store_bytes: Metric {
    //         value: values.l2_local_global_store_bytes,
    //         ..Metric::default() // unit: units.l2_local_global_store_bytes,
    //     },
    //     l2_global_reduction_bytes: Metric {
    //         value: values.l2_global_reduction_bytes,
    //         // unit: units.l2_global_reduction_bytes,
    //         ..Metric::default()
    //     },
    //     l2_global_atomic_store_bytes: Metric {
    //         value: values.l2_global_atomic_store_bytes,
    //         ..Metric::default() // unit: units.l2_global_atomic_store_bytes,
    //     },
    //     l2_surface_store_bytes: Metric {
    //         value: values.l2_surface_store_bytes,
    //         ..Metric::default() // unit: units.l2_surface_store_bytes,
    //     },
    //     l2_surface_reduction_bytes: Metric {
    //         value: values.l2_surface_reduction_bytes,
    //         ..Metric::default() // unit: units.l2_surface_reduction_bytes,
    //     },
    //     l2_surface_atomic_store_bytes: Metric {
    //         value: values.l2_surface_atomic_store_bytes,
    //         ..Metric::default() // unit: units.l2_surface_atomic_store_bytes,
    //     },
    //     global_load_requests: Metric {
    //         value: values.global_load_requests,
    //         ..Metric::default() // unit: units.global_load_requests,
    //     },
    //     local_load_requests: Metric {
    //         value: values.local_load_requests,
    //         ..Metric::default() // unit: units.local_load_requests,
    //     },
    //     surface_load_requests: Metric {
    //         value: values.surface_load_requests,
    //         ..Metric::default() // unit: units.surface_load_requests,
    //     },
    //     global_store_requests: Metric {
    //         value: values.global_store_requests,
    //         ..Metric::default() // unit: units.global_store_requests,
    //     },
    //     local_store_requests: Metric {
    //         value: values.local_store_requests,
    //         ..Metric::default() // unit: units.local_store_requests,
    //     },
    //     surface_store_requests: Metric {
    //         value: values.surface_store_requests,
    //         ..Metric::default() // unit: units.surface_store_requests,
    //     },
    //     global_atomic_requests: Metric {
    //         value: values.global_atomic_requests,
    //         ..Metric::default() // unit: units.global_atomic_requests,
    //     },
    //     global_reduction_requests: Metric {
    //         value: values.global_reduction_requests,
    //         ..Metric::default() // unit: units.global_reduction_requests,
    //     },
    //     surface_atomic_requests: Metric {
    //         value: values.surface_atomic_requests,
    //         ..Metric::default() // unit: units.surface_atomic_requests,
    //     },
    //     surface_reduction_requests: Metric {
    //         value: values.surface_reduction_requests,
    //         ..Metric::default() // unit: units.surface_reduction_requests,
    //     },
    //     dram_read_transactions: Metric {
    //         value: values.dram_read_transactions,
    //         ..Metric::default() // unit: units.dram_read_transactions,
    //     },
    //     dram_write_transactions: Metric {
    //         value: values.dram_write_transactions,
    //         ..Metric::default() // unit: units.dram_write_transactions,
    //     },
    //     dram_read_throughput: Metric {
    //         value: values.dram_read_throughput,
    //         ..Metric::default() // unit: units.dram_read_throughput,
    //     },
    //     dram_write_throughput: Metric {
    //         value: values.dram_write_throughput,
    //         ..Metric::default() // unit: units.dram_write_throughput,
    //     },
    //     dram_write_bytes: Metric {
    //         value: values.dram_write_bytes,
    //         ..Metric::default() // unit: units.dram_write_bytes,
    //     },
    //     dram_read_bytes: Metric {
    //         value: values.dram_read_bytes,
    //         ..Metric::default() // unit: units.dram_read_bytes,
    //     },
    // })
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

        println!("writing log to {}", keep_log_file_path.display());
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

    let metrics = parse_nvprof_csv(&mut log_reader)?;
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
        let mut log = Cursor::new(include_bytes!(
            "../../tests/nvprof_vectoradd_100_32_metrics_all.txt"
        ));
        let metrics = parse_nvprof_csv(&mut log)?;
        dbg!(&metrics);
        assert_eq!(
            metrics.device,
            Metric::new("NVIDIA GeForce GTX 1080 (0)".to_string(), None)
        );
        assert_eq!(
            metrics.kernel,
            Metric::new("_Z6vecAddIfEvPT_S1_S1_i".to_string(), None)
        );
        assert_eq!(
            metrics.context,
            Metric::new(1, None) // Metric {
                                 //     value: Some(1),
                                 //     unit: None
                                 // }
        );
        assert_eq!(
            metrics.stream,
            Metric::new(7, None) // Metric {
                                 //     value: Some(7),
                                 //     unit: None
                                 // }
        );
        assert_eq!(
            metrics.dram_write_bytes,
            Metric::new(0, None) // Metric {
                                 //     value: Some(66),
                                 //     unit: None
                                 // }
        );
        assert_eq!(
            metrics.dram_read_bytes,
            Metric::new(7136, None) // Metric {
                                    //     value: Some(66),
                                    //     unit: None,
                                    // }
        );
        assert_eq!(
            metrics.dram_read_transactions,
            Metric::new(223, None) // Metric {
                                   //     value: Some(66),
                                   //     unit: None
                                   // }
        );
        assert_eq!(
            metrics.dram_write_transactions,
            Metric::new(0, None) // Metric {
                                 //     value: Some(0),
                                 //     unit: None
                                 // }
        );
        assert_eq!(
            metrics.l2_read_transactions,
            Metric::new(66, None) // Metric {
                                  // Metric {
                                  //     value: Some(66),
                                  //     unit: None
                                  // }
        );
        assert_eq!(
            metrics.l2_write_transactions,
            Metric::new(26, None) // Metric {
                                  // Metric {
                                  //     value: Some(26),
                                  //     unit: None
                                  // }
        );
        Ok(())
    }
}
