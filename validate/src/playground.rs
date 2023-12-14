use crate::{
    materialized::{self, BenchmarkConfig, TargetBenchmarkConfig},
    options::{self, Options},
    RunError, TraceProvider,
};
use color_eyre::{eyre, Help};
use std::io::Read;
use std::time::Duration;

#[must_use]
pub fn is_debug() -> bool {
    playground::is_debug()
}

pub fn simulate_bench_config<A>(
    bench: &BenchmarkConfig,
    trace_provider: TraceProvider,
    extra_args: A,
    accelsim_compat_mode: bool,
) -> Result<(String, playground::stats::StatsBridge, Duration), RunError>
where
    A: IntoIterator,
    <A as IntoIterator>::Item: Into<String>,
{
    let (TargetBenchmarkConfig::PlaygroundSimulate {
        ref traces_dir,
        ref configs,
        ..
    }
    | TargetBenchmarkConfig::AccelsimSimulate {
        ref traces_dir,
        ref configs,
        ..
    }) = bench.target_config
    else {
        unreachable!();
    };

    let kernelslist = match trace_provider {
        TraceProvider::Native | TraceProvider::Accelsim => traces_dir.join("kernelslist.g"),
        TraceProvider::Box => {
            accelsim::tracegen::convert_box_to_accelsim_traces(&accelsim::tracegen::Conversion {
                native_commands_path: &traces_dir.join("../trace/commands.json"),
                box_traces_dir: &traces_dir.join("../trace"),
                accelsim_traces_dir: traces_dir,
            })?
        }
    };
    if !kernelslist.is_file() {
        return Err(RunError::Failed(
            eyre::eyre!("missing {}", kernelslist.display()).with_suggestion(|| {
                let trace_cmd = format!(
                    "cargo validate -b {}@{} accelsim-trace",
                    bench.name,
                    bench.input_idx + 1
                );
                format!("generate traces first using: `{trace_cmd}`")
            }),
        ));
    }

    let materialized::config::AccelsimSimConfigFiles {
        config,
        trace_config,
        inter_config,
        ..
    } = configs.clone();

    let kernelslist = kernelslist.to_string_lossy().to_string();
    let gpgpusim_config = config.to_string_lossy().to_string();
    let trace_config = trace_config.to_string_lossy().to_string();
    let inter_config = inter_config.to_string_lossy().to_string();

    let mut args = vec![
        "-trace".to_string(),
        kernelslist.to_string(),
        "-config".to_string(),
        gpgpusim_config,
        "-config".to_string(),
        trace_config,
        "-inter_config_file".to_string(),
        inter_config,
    ];
    let extra_args: Vec<String> = extra_args.into_iter().map(Into::into).collect();
    args.extend(extra_args.into_iter());
    // args.extend(extra_args.iter().map(String::as_str));

    let tmp_dir = tempfile::tempdir().map_err(eyre::Report::from)?;
    let log_file_path = tmp_dir.path().join("log.txt");
    log::debug!("playground log file: {}", log_file_path.display());

    let config = playground::Config {
        accelsim_compat_mode,
        print_stats: true,
        stats_file: Some(
            std::ffi::CString::new(log_file_path.to_string_lossy().to_string().to_string())
                .unwrap(),
        ),
    };

    let start = std::time::Instant::now();
    let stats = playground::run(config, args).map_err(eyre::Report::from)?;
    let dur = start.elapsed();

    let mut raw_log = String::new();

    let mut log_reader = utils::fs::open_readable(log_file_path).map_err(eyre::Report::from)?;
    log_reader
        .read_to_string(&mut raw_log)
        .map_err(eyre::Report::from)?;

    Ok((raw_log, stats, dur))
}

pub async fn simulate(
    bench: BenchmarkConfig,
    options: &Options,
    _sim_options: &options::PlaygroundSim,
    _bar: &indicatif::ProgressBar,
) -> Result<Duration, RunError> {
    let TargetBenchmarkConfig::PlaygroundSimulate { ref stats_dir, .. } = bench.target_config
    else {
        unreachable!();
    };

    let detailed_stats_dir = &stats_dir.join("detailed");
    if options.clean {
        utils::fs::remove_dir(stats_dir).map_err(eyre::Report::from)?;
        utils::fs::remove_dir(detailed_stats_dir).map_err(eyre::Report::from)?;
    }

    utils::fs::create_dirs(stats_dir).map_err(eyre::Report::from)?;

    if !options.force
        && [stats_dir.as_path(), detailed_stats_dir.as_path()]
            .iter()
            .all(|stat_dir| crate::stats::already_exist(&bench.common, stat_dir))
    {
        return Err(RunError::Skipped);
    }

    let mut total_dur = Duration::ZERO;
    for repetition in 0..bench.common.repetitions {
        let accelsim_compat_mode = true;
        let extra_args = vec!["-gpgpu_perf_sim_memcpy", "0"];
        let bench = bench.clone();
        let (log, stats, dur) = tokio::task::spawn_blocking(move || {
            let (log, stats, dur) = simulate_bench_config(
                &bench,
                TraceProvider::Native,
                extra_args,
                accelsim_compat_mode,
            )?;
            Ok::<_, eyre::Report>((log, stats, dur))
        })
        .await
        .unwrap()?;

        total_dur += dur;
        let mut converted_stats: stats::Stats = stats.into();

        converted_stats.sim.elapsed_millis = dur.as_millis();
        converted_stats.sim.is_release_build = playground::is_debug();

        // cannot report per kernel for now...
        // however, we process per kernel stats from the log file the same as for accelsim
        let per_kernel_stats = vec![converted_stats];

        super::accelsim::process_stats(log.into_bytes(), &dur, stats_dir, repetition)?;
        let full = false;
        super::simulate::process_stats(&per_kernel_stats, detailed_stats_dir, repetition, full)?;
    }
    Ok(total_dur)
}
