use crate::{
    open_writable,
    options::{self, Options},
    RunError,
};
use color_eyre::{eyre, Help};
use std::time::Instant;
use utils::fs::create_dirs;
use validate::materialize::BenchmarkConfig;

pub async fn simulate(
    bench: BenchmarkConfig,
    options: &Options,
    _trace_opts: &options::Sim,
) -> Result<(), RunError> {
    // get traces dir from trace config
    let traces_dir = bench.trace.traces_dir;
    let stats_dir = bench.simulate.stats_dir;

    if !options.force && crate::stats::already_exist(&stats_dir) {
        return Err(RunError::Skipped);
    }

    let commandlist = traces_dir.join("commands.json");
    if !commandlist.is_file() {
        return Err(RunError::Failed(
            eyre::eyre!("missing {}", commandlist.display()).with_suggestion(|| {
                let trace_cmd = format!(
                    "cargo validate -b {}@{} trace",
                    bench.name,
                    bench.input_idx + 1
                );
                format!("generate traces first using: `{trace_cmd}`")
            }),
        ));
    }

    let (stats, dur) = tokio::task::spawn_blocking(move || {
        let start = Instant::now();
        let stats = casimu::ported::accelmain(traces_dir, None)?;
        Ok::<_, eyre::Report>((stats, start.elapsed()))
    })
    .await
    .map_err(eyre::Report::from)??;

    create_dirs(&stats_dir).map_err(eyre::Report::from)?;

    crate::stats::write_stats_as_csv(&stats_dir, stats)?;

    serde_json::to_writer_pretty(
        open_writable(stats_dir.join("exec_time.json"))?,
        &dur.as_millis(),
    )
    .map_err(eyre::Report::from)?;

    // let json_stats: stats::FlatStats = stats.clone().into();
    // let json_stats_out_file = stats_dir.join("stats.json");
    // serde_json::to_writer_pretty(open_writable(&csv_stats_out_file)?, &stats)?;

    // let csv_stats_out_file = stats_dir.join("stats.csv");
    // let mut csv_writer = csv::WriterBuilder::new()
    //     .flexible(false)
    //     .from_writer(open_writable(&csv_stats_out_file)?);
    // csv_writer.serialize(stats)?;
    // let data: Vec<(&str, &[&dyn serde::Serialize])> = vec![
    //     ("sim", &[stats.sim]),
    //     // ("dram", &[stats.dram]),
    //     ("accesses", &stats.accesses.flatten()),
    // ];
    // for (stat_name, rows) in data {
    //     validate::write_csv_rows(
    //         open_writable(stats_dir.join(format!("stats.{}.csv", stat_name)))?,
    //         rows,
    //     )?;

    // serde_json::to_writer_pretty(open_writable(&json_stats_out_file)?, &json_stats)?;
    Ok(())
}