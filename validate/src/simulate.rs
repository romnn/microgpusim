use super::materialized::{BenchmarkConfig, TargetBenchmarkConfig};
use super::{
    options::{self, Options},
    RunError,
};
use color_eyre::{eyre, Help};
use gpucachesim::{config::Parallelization, interconn as ic, mem_fetch, MockSimulator};
use std::path::Path;
use std::time::Instant;
use utils::fs::create_dirs;

#[derive(Debug, serde::Deserialize)]
struct Input {
    #[serde(rename = "mode")]
    parallelism_mode: Option<String>,
    #[serde(rename = "threads")]
    parallelism_threads: Option<usize>,
    #[serde(rename = "run_ahead")]
    parallelism_run_ahead: Option<usize>,
    cores_per_cluster: Option<usize>,
    memory_only: Option<bool>,
}

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

    let err = |err: serde_json::Error| {
        RunError::Failed(
            eyre::Report::from(err).wrap_err(
                eyre::eyre!(
                    "failed to parse input values for bench config {}@{}",
                    bench.name,
                    bench.input_idx
                )
                .with_section(|| format!("{:#?}", bench.values)),
            ),
        )
    };
    let values: serde_json::Value = serde_json::to_value(&bench.values).map_err(err)?;
    let input: Input = serde_json::from_value(values).map_err(err)?;
    dbg!(&input);

    let parallelization = match (
        input
            .parallelism_mode
            .as_deref()
            .map(str::to_lowercase)
            .as_deref(),
        input.parallelism_run_ahead,
    ) {
        (Some("serial") | None, _) => Parallelization::Serial,
        #[cfg(feature = "parallel")]
        (Some("deterministic"), _) => Parallelization::Deterministic,
        #[cfg(feature = "parallel")]
        (Some("nondeterministic"), run_ahead) => {
            Parallelization::Nondeterministic(run_ahead.unwrap_or(10))
        }
        (Some(other), _) => panic!("unknown parallelization mode: {other}"),
        #[cfg(not(feature = "parallel"))]
        _ => {
            return Err(RunError::Failed(
                eyre::eyre!("parallel feature is disabled")
                    .with_suggestion(|| format!(r#"enable the "parallel" feature"#)),
            ))
        }
    };

    let config = gpucachesim::config::GPU {
        num_simt_clusters: 20,                                            // 20
        num_cores_per_simt_cluster: input.cores_per_cluster.unwrap_or(1), // 1
        num_schedulers_per_core: 2,                                       // 1
        num_memory_controllers: 8,                                        // 8
        num_dram_chips_per_memory_controller: 1,                          // 1
        num_sub_partitions_per_memory_controller: 2,                      // 2
        fill_l2_on_memcopy: false,                                        // true
        memory_only: input.memory_only.unwrap_or(false),
        parallelization,
        log_after_cycle: None,
        simulation_threads: input.parallelism_threads,
        ..gpucachesim::config::GPU::default()
    };

    let sim = gpucachesim::accelmain(traces_dir, config)?;
    let stats = sim.stats();
    let mut wip_stats = gpucachesim::WIP_STATS.lock();
    dbg!(&wip_stats);
    dbg!(wip_stats.warp_instructions as f32 / wip_stats.num_warps as f32);
    for kernel_stats in &stats.inner {
        dbg!(&kernel_stats.sim);
    }
    dbg!(gpucachesim::is_debug());

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
        process_stats(stats.as_ref(), &dur, stats_dir, repetition)?;
    }
    Ok(())
}

#[inline]
pub fn process_stats(
    stats: &[stats::Stats],
    _dur: &std::time::Duration,
    stats_dir: &Path,
    repetition: usize,
) -> Result<(), RunError> {
    create_dirs(stats_dir).map_err(eyre::Report::from)?;
    crate::stats::write_stats_as_csv(stats_dir, stats, repetition)?;
    Ok(())
}
