use crate::{
    materialize::{self, BenchmarkConfig},
    options::{self, Options},
    RunError, TraceProvider,
};
use color_eyre::{eyre, Help};
use std::io::Read;
use std::time::Duration;

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
    let traces_dir = &bench.accelsim_trace.traces_dir;
    let kernelslist = traces_dir.join(match trace_provider {
        TraceProvider::Native | TraceProvider::Accelsim => "kernelslist.g",
        TraceProvider::Box => "box-kernelslist.g",
    });
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

    let materialize::AccelsimSimConfigFiles {
        config,
        trace_config,
        inter_config,
        ..
    } = bench.playground_simulate.configs.clone();

    let kernelslist = kernelslist.to_string_lossy().to_string();
    let gpgpusim_config = config.to_string_lossy().to_string();
    let trace_config = trace_config.to_string_lossy().to_string();
    let inter_config = inter_config.to_string_lossy().to_string();

    let mut args = vec![
        "-trace",
        &kernelslist,
        "-config",
        &gpgpusim_config,
        "-config",
        &trace_config,
        "-inter_config_file",
        &inter_config,
    ];
    let extra_args: Vec<String> = extra_args.into_iter().map(Into::into).collect();
    args.extend(extra_args.iter().map(String::as_str));

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
    let stats = playground::run(config, args.as_slice()).map_err(eyre::Report::from)?;
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
    _sim_opts: &options::PlaygroundSim,
) -> Result<(), RunError> {
    // get traces dir from accelsim trace config
    let stats_dir = bench.playground_simulate.stats_dir.clone();
    let detailed_stats_dir = stats_dir.join("detailed");

    if !options.force
        && [stats_dir.as_path(), detailed_stats_dir.as_path()]
            .iter()
            .all(crate::stats::already_exist)
    {
        return Err(RunError::Skipped);
    }

    let accelsim_compat_mode = true;
    let extra_args = vec!["-gpgpu_perf_sim_memcpy", "0"];
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

    let detailed_stats: stats::Stats = stats.into();

    super::accelsim::process_stats(log.into_bytes(), &dur, &stats_dir)?;
    super::simulate::process_stats(detailed_stats, &dur, &detailed_stats_dir)?;
    Ok(())
}
