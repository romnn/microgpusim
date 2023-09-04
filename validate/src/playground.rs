use crate::{
    materialize::{self, BenchmarkConfig},
    options::{self, Options},
    RunError, TraceProvider,
};
use color_eyre::{eyre, Help};
use std::io::Read;

pub fn simulate_bench_config(
    bench: &BenchmarkConfig,
    trace_provider: TraceProvider,
    accelsim_compat_mode: bool,
) -> Result<(String, playground::stats::StatsBridge), RunError> {
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

    let args = [
        "-trace",
        &kernelslist,
        "-config",
        &gpgpusim_config,
        "-config",
        &trace_config,
        "-inter_config_file",
        &inter_config,
    ];

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
    let stats = playground::run(config, args.as_slice()).map_err(eyre::Report::from)?;

    let mut raw_log = String::new();

    let mut log_reader = utils::fs::open_readable(log_file_path).map_err(eyre::Report::from)?;
    log_reader
        .read_to_string(&mut raw_log)
        .map_err(eyre::Report::from)?;

    Ok((raw_log, stats))
}

pub async fn simulate(
    bench: BenchmarkConfig,
    options: &Options,
    _sim_opts: &options::PlaygroundSim,
) -> Result<(), RunError> {
    // get traces dir from accelsim trace config
    let stats_dir = bench.playground_simulate.stats_dir.clone();

    if !options.force && crate::stats::already_exist(&stats_dir) {
        return Err(RunError::Skipped);
    }

    let accelsim_compat_mode = true;
    let (log, _stats, dur) = tokio::task::spawn_blocking(move || {
        let start = std::time::Instant::now();
        let (log, stats) =
            simulate_bench_config(&bench, TraceProvider::Native, accelsim_compat_mode)?;
        Ok::<_, eyre::Report>((log, stats, start.elapsed()))
    })
    .await
    .unwrap()?;

    // dbg!(&log);

    super::accelsim::process_stats(log.into_bytes(), dur, &stats_dir)?;

    // let stats = stats::Stats {
    //     accesses: stats.accesses.into(),
    //     instructions: stats.instructions.into(),
    //     // pub sim: Sim,
    //     // pub dram: DRAM,
    //     // pub l1i_stats: PerCache,
    //     // pub l1c_stats: PerCache,
    //     // pub l1t_stats: PerCache,
    //     // pub l1d_stats: PerCache,
    //     // pub l2d_stats: PerCache,
    //     ..stats::Stats::default()
    // };

    // create_dirs(&stats_dir).map_err(eyre::Report::from)?;
    // let _stats_out_file = stats_dir.join("stats.json");
    //
    // // let flat_stats: Vec<_> = stats.into_iter().collect();
    // // serde_json::to_writer_pretty(open_writable(&stats_out_file)?, &flat_stats)?;
    //
    // #[cfg(debug_assertions)]
    // let exec_time_file_path = stats_dir.join("exec_time.debug.json");
    // #[cfg(not(debug_assertions))]
    // let exec_time_file_path = stats_dir.join("exec_time.release.json");
    //
    // serde_json::to_writer_pretty(open_writable(exec_time_file_path)?, &dur.as_millis())
    //     .map_err(eyre::Report::from)?;
    Ok(())
}
