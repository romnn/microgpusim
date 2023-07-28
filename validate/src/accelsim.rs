use crate::{
    open_writable,
    options::{self, Options},
    RunError,
};
use color_eyre::{eyre, Help};
use std::io::Write;
use utils::fs::create_dirs;
use validate::materialize::{self, BenchmarkConfig};

pub async fn trace(
    bench: BenchmarkConfig,
    options: &Options,
    _trace_opts: &options::AccelsimTrace,
) -> Result<(), RunError> {
    let traces_dir = &bench.accelsim_trace.traces_dir;
    create_dirs(traces_dir).map_err(eyre::Report::from)?;

    if !options.force && traces_dir.join("kernelslist.g").is_file() {
        return Err(RunError::Skipped);
    }

    let options = accelsim_trace::Options {
        traces_dir: traces_dir.clone(),
        nvbit_tracer_tool: None, // auto detect
        ..accelsim_trace::Options::default()
    };
    let dur = accelsim_trace::trace(&bench.executable, &bench.args, &options).await?;

    let trace_dur_file = traces_dir.join("trace_time.json");
    serde_json::to_writer_pretty(open_writable(trace_dur_file)?, &dur.as_millis())
        .map_err(eyre::Report::from)?;
    Ok(())
}

pub async fn simulate(
    bench: BenchmarkConfig,
    options: &Options,
    _sim_options: &options::AccelsimSim,
) -> Result<(), RunError> {
    let traces_dir = &bench.accelsim_trace.traces_dir;
    let stats_dir = &bench.accelsim_simulate.stats_dir;

    if !options.force && crate::stats::already_exist(stats_dir) {
        return Err(RunError::Skipped);
    }

    // todo: allow setting this in test-apps.yml ?
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

    let common = &bench.accelsim_simulate.common;

    let timeout = common.timeout.map(Into::into);

    let materialize::AccelsimSimConfigFiles {
        config,
        config_dir,
        trace_config,
        inter_config,
    } = bench.accelsim_simulate.configs.clone();

    let config = accelsim::SimConfig {
        config: Some(config),
        config_dir: Some(config_dir),
        trace_config: Some(trace_config),
        inter_config: Some(inter_config),
    };

    let (log, dur) =
        accelsim_sim::simulate_trace(&traces_dir, &kernelslist, config, timeout).await?;

    // parse stats
    let parse_options = accelsim::parser::Options::default();
    let log_reader = std::io::Cursor::new(&log.stdout);
    let stats = accelsim::parser::parse_stats(log_reader, &parse_options)?;

    create_dirs(stats_dir).map_err(eyre::Report::from)?;

    open_writable(stats_dir.join("log.txt"))?
        .write_all(&log.stdout)
        .map_err(eyre::Report::from)?;

    let converted_stats: stats::Stats = stats.clone().try_into()?;
    dbg!(&converted_stats);
    crate::stats::write_stats_as_csv(stats_dir, converted_stats)?;
    // validate::write_csv_rows(
    //     open_writable(stats_dir.join("stats.csv"))?,
    //     &stats.into_inner().into_iter().collect::<Vec<_>>(),
    // )?;

    validate::write_csv_rows(
        open_writable(stats_dir.join("raw.stats.csv"))?,
        &stats.into_inner().into_iter().collect::<Vec<_>>(),
    )?;

    // let flat_stats: Vec<_> = stats.into_inner().into_iter().collect();
    // serde_json::to_writer_pretty(open_writable(&stats_out_file)?, &flat_stats)?;

    serde_json::to_writer_pretty(
        open_writable(stats_dir.join("exec_time.json"))?,
        &dur.as_millis(),
    )
    .map_err(eyre::Report::from)?;
    Ok(())
}
