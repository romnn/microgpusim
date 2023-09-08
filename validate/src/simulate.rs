use super::materialize::BenchmarkConfig;
use super::{
    open_writable,
    options::{self, Options},
    RunError,
};
use color_eyre::{eyre, Help};
use gpucachesim::{interconn as ic, mem_fetch, MockSimulator};
use std::path::Path;
use std::time::Instant;
use utils::fs::create_dirs;

#[allow(clippy::module_name_repetitions)]
pub fn simulate_bench_config(
    bench: &BenchmarkConfig,
) -> Result<MockSimulator<ic::ToyInterconnect<ic::Packet<mem_fetch::MemFetch>>>, RunError> {
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

    let non_deterministic: Option<usize> = std::env::var("NONDET")
        .ok()
        .as_deref()
        .map(str::parse)
        .transpose()
        .unwrap();

    let parallelization = match (bench.simulate.parallel, non_deterministic) {
        (false, _) => gpucachesim::config::Parallelization::Serial,
        #[cfg(feature = "parallel")]
        (true, None) => gpucachesim::config::Parallelization::Deterministic,
        #[cfg(feature = "parallel")]
        (true, Some(n)) => gpucachesim::config::Parallelization::Nondeterministic(n),
        #[cfg(not(feature = "parallel"))]
        _ => {
            return Err(RunError::Failed(
                eyre::eyre!("parallel feature is disabled")
                    .with_suggestion(|| format!(r#"enable the "parallel" feature"#)),
            ))
        }
    };

    let config = gpucachesim::config::GPU {
        num_simt_clusters: 20,                   // 20
        num_cores_per_simt_cluster: 1,           // 1
        num_schedulers_per_core: 2,              // 1
        num_memory_controllers: 8,               // 8
        num_sub_partition_per_memory_channel: 2, // 2
        fill_l2_on_memcopy: false,               // true
        parallelization,
        log_after_cycle: None,
        ..gpucachesim::config::GPU::default()
    };
    // dbg!(&config);

    let sim = gpucachesim::accelmain(traces_dir, config)?;
    dbg!(&gpucachesim::WIP_STATS.lock());

    Ok(sim)
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

    let (sim, dur) = tokio::task::spawn_blocking(move || {
        let start = Instant::now();
        let stats = simulate_bench_config(&bench)?;
        Ok::<_, eyre::Report>((stats, start.elapsed()))
    })
    .await
    .map_err(eyre::Report::from)??;

    let stats = sim.stats();
    let profile = if gpucachesim::is_debug() {
        "debug"
    } else {
        "release"
    };
    process_stats(stats, &dur, &stats_dir, profile)?;

    Ok(())
}

#[inline]
pub fn process_stats(
    stats: stats::Stats,
    dur: &std::time::Duration,
    stats_dir: &Path,
    profile: &str,
) -> Result<(), RunError> {
    create_dirs(stats_dir).map_err(eyre::Report::from)?;
    crate::stats::write_stats_as_csv(stats_dir, stats)?;

    let exec_time_file_path = stats_dir.join(format!("exec_time.{profile}.json"));
    serde_json::to_writer_pretty(open_writable(exec_time_file_path)?, &dur.as_millis())
        .map_err(eyre::Report::from)?;
    Ok(())
}
