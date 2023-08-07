use super::materialize::BenchmarkConfig;
use super::{
    open_writable,
    options::{self, Options},
    RunError,
};
use color_eyre::{eyre, Help};
use std::time::Instant;
use utils::fs::create_dirs;

pub fn simulate_bench_config(bench: &BenchmarkConfig) -> Result<stats::Stats, RunError> {
    // get traces dir from trace config
    let traces_dir = bench.trace.traces_dir.clone();

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

    // let config = Arc::new(config::GPUConfig {
    //     num_simt_clusters: 20,                   // 20
    //     num_cores_per_simt_cluster: 4,           // 1
    //     num_schedulers_per_core: 2,              // 1
    //     num_memory_controllers: 8,               // 8
    //     num_sub_partition_per_memory_channel: 2, // 2
    //     fill_l2_on_memcopy: true,                // true
    //     ..config::GPUConfig::default()
    // });

    let stats = casimu::accelmain(traces_dir, None)?;
    Ok(stats)
}

pub async fn simulate(
    bench: BenchmarkConfig,
    options: &Options,
    _trace_opts: &options::Sim,
) -> Result<(), RunError> {
    let stats_dir = bench.simulate.stats_dir.clone();

    if !options.force && crate::stats::already_exist(&stats_dir) {
        return Err(RunError::Skipped);
    }

    let (stats, dur) = tokio::task::spawn_blocking(move || {
        let start = Instant::now();
        let stats = simulate_bench_config(&bench)?;
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
