use crate::{
    open_writable,
    options::{self, Options},
    RunError,
};
use color_eyre::{eyre, Help};
use std::time::Instant;
use utils::fs::create_dirs;
use validate::materialize::{self, BenchmarkConfig};

pub async fn simulate(
    bench: &BenchmarkConfig,
    options: &Options,
    _sim_opts: &options::PlaygroundSim,
) -> Result<(), RunError> {
    // get traces dir from accelsim trace config
    let traces_dir = &bench.accelsim_trace.traces_dir;
    let stats_dir = &bench.playground_simulate.stats_dir;

    if !options.force && crate::stats::already_exist(stats_dir) {
        return Err(RunError::Skipped);
    }

    let kernelslist = traces_dir.join("kernelslist.g");
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

    let (_stats, dur) = tokio::task::spawn_blocking(move || {
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
        dbg!(&args);

        let start = Instant::now();
        let config = playground::Config::default();
        let stats = playground::run(&config, args.as_slice())?;
        Ok::<_, eyre::Report>((stats, start.elapsed()))
    })
    .await
    .map_err(eyre::Report::from)??;

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

    create_dirs(stats_dir).map_err(eyre::Report::from)?;
    let _stats_out_file = stats_dir.join("stats.json");
    let exec_dur_file = stats_dir.join("exec_time.json");

    // let flat_stats: Vec<_> = stats.into_iter().collect();
    // serde_json::to_writer_pretty(open_writable(&stats_out_file)?, &flat_stats)?;
    serde_json::to_writer_pretty(open_writable(exec_dur_file)?, &dur.as_millis())
        .map_err(eyre::Report::from)?;
    Ok(())
}
