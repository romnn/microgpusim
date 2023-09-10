use super::materialized::{BenchmarkConfig, TargetBenchmarkConfig};
use super::{
    open_writable,
    options::{self, Options},
    RunError,
};
use color_eyre::{eyre, Help};
use gpucachesim::{config::Parallelization, interconn as ic, mem_fetch, MockSimulator};
use serde_json_merge::Index;
use std::path::Path;
use std::time::Instant;
use utils::fs::create_dirs;

#[allow(clippy::module_name_repetitions)]
pub fn simulate_bench_config(
    bench: &BenchmarkConfig,
) -> Result<MockSimulator<ic::ToyInterconnect<ic::Packet<mem_fetch::MemFetch>>>, RunError> {
    let TargetBenchmarkConfig::Simulate { ref traces_dir, .. } = bench.target_config else {
        unreachable!();
    };

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

    let values: serde_json::Value = serde_json::to_value(&bench.values).unwrap();

    // TODO: decide on a parallel implementation using the inputs
    // let non_deterministic: Option<usize> = std::env::var("NONDET")
    //     .ok()
    //     .as_deref()
    //     .map(str::parse)
    //     .transpose()
    //     .unwrap();

    let parallelism_mode = values
        .get_index(serde_json_merge::index!("mode"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_lowercase);

    let parallelism_threads = values
        .get_index(serde_json_merge::index!("threads"))
        .and_then(serde_json::Value::as_u64)
        .map(|threads| threads as usize);

    let parallelism_run_ahead = values
        .get_index(serde_json_merge::index!("run_ahead"))
        .and_then(serde_json::Value::as_u64);

    let parallelization = match (parallelism_mode.as_deref(), parallelism_run_ahead) {
        (Some("serial") | None, _) => Parallelization::Serial,
        #[cfg(feature = "parallel")]
        (Some("deterministic"), _) => Parallelization::Deterministic,
        #[cfg(feature = "parallel")]
        (Some("nondeterministic"), run_ahead) => {
            Parallelization::Nondeterministic(run_ahead.unwrap_or(10) as usize)
        }
        (Some(other), _) => panic!("unknown parallelization mode: {}", other),
        #[cfg(not(feature = "parallel"))]
        _ => {
            return Err(RunError::Failed(
                eyre::eyre!("parallel feature is disabled")
                    .with_suggestion(|| format!(r#"enable the "parallel" feature"#)),
            ))
        }
    };

    dbg!(&parallelization);

    let config = gpucachesim::config::GPU {
        num_simt_clusters: 20,                       // 20
        num_cores_per_simt_cluster: 1,               // 1
        num_schedulers_per_core: 2,                  // 1
        num_memory_controllers: 8,                   // 8
        num_dram_chips_per_memory_controller: 1,     // 1
        num_sub_partitions_per_memory_controller: 2, // 2
        fill_l2_on_memcopy: false,                   // true
        parallelization,
        log_after_cycle: None,
        simulation_threads: parallelism_threads,
        ..gpucachesim::config::GPU::default()
    };

    let sim = gpucachesim::accelmain(traces_dir, config)?;
    let stats = sim.stats();
    let mut wip_stats = gpucachesim::WIP_STATS.lock();
    dbg!(&wip_stats);
    dbg!(wip_stats.warp_instructions as f32 / wip_stats.num_warps as f32);
    for kernel_stats in stats.inner.iter() {
        dbg!(&kernel_stats.sim);
    }

    *wip_stats = gpucachesim::WIPStats::default();

    Ok(sim)
}

pub async fn simulate(
    bench: BenchmarkConfig,
    options: &Options,
    _sim_options: &options::Sim,
    _bar: &indicatif::ProgressBar,
) -> Result<(), RunError> {
    let TargetBenchmarkConfig::Simulate { ref stats_dir, .. } = bench.target_config else {
        unreachable!();
    };

    if options.clean {
        utils::fs::remove_dir(stats_dir).map_err(eyre::Report::from)?;
    }

    create_dirs(stats_dir).map_err(eyre::Report::from)?;

    if !options.force && crate::stats::already_exist(&bench.common, stats_dir) {
        return Err(RunError::Skipped);
    }

    for repetition in 0..bench.common.repetitions {
        let bench = bench.clone();
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
        process_stats(stats.as_ref(), &dur, stats_dir, profile, repetition)?;
    }
    Ok(())
}

#[inline]
pub fn process_stats(
    stats: &[stats::Stats],
    dur: &std::time::Duration,
    stats_dir: &Path,
    profile: &str,
    repetition: usize,
) -> Result<(), RunError> {
    create_dirs(stats_dir).map_err(eyre::Report::from)?;
    crate::stats::write_stats_as_csv(stats_dir, stats, repetition)?;

    let exec_time_file_path = stats_dir.join(format!("exec_time.{profile}.{repetition}.json"));
    serde_json::to_writer_pretty(open_writable(exec_time_file_path)?, &dur.as_millis())
        .map_err(eyre::Report::from)?;
    Ok(())
}
