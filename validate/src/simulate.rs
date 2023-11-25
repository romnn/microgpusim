use crate::materialized::{BenchmarkConfig, TargetBenchmarkConfig};
use crate::{
    options::{self, Options},
    RunError,
};
use color_eyre::{eyre, Help};
pub use gpucachesim::config::{self, Parallelization};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use utils::fs::create_dirs;

#[allow(clippy::module_name_repetitions)]
pub fn simulate_bench_config(bench: &BenchmarkConfig) -> Result<config::GTX1080, RunError> {
    let TargetBenchmarkConfig::Simulate { ref traces_dir, l2_prefill, .. } = bench.target_config else {
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

    let input = gpucachesim::config::parse_input(&bench.values).map_err(|err| {
        eyre::Report::from(err).wrap_err(
            eyre::eyre!(
                "failed to parse input values for bench config {}@{}",
                bench.name,
                bench.input_idx
            )
            .with_section(|| format!("{:#?}", bench.values)),
        )
    })?;

    // dbg!(&input);

    let mut sim_config = gpucachesim::config::gtx1080::build_config(&input)?;
    if let Some(l2_prefill) = l2_prefill {
        sim_config.fill_l2_on_memcopy = l2_prefill;
    }
    gpucachesim::init_deadlock_detector();
    let mut sim = gpucachesim::config::GTX1080::new(Arc::new(sim_config));

    let (traces_dir, commands_path) = if traces_dir.is_dir() {
        (traces_dir.to_path_buf(), traces_dir.join("commands.json"))
    } else {
        (
            traces_dir.parent().map(Path::to_path_buf).ok_or_else(|| {
                eyre::eyre!(
                    "could not determine trace dir from file {}",
                    traces_dir.display()
                )
            })?,
            traces_dir.to_path_buf(),
        )
    };
    sim.add_commands(commands_path, traces_dir)?;
    sim.run()?;

    let stats = sim.stats();
    // let mut wip_stats = gpucachesim::WIP_STATS.lock();
    // dbg!(&wip_stats);
    // dbg!(wip_stats.warp_instructions as f32 / wip_stats.num_warps as f32);
    // dbg!(&stats.inner.len());
    for kernel_stats in &stats.inner {
        // dbg!(&kernel_stats.sim);
        // dbg!(&kernel_stats.l1d_stats);
        // dbg!(&kernel_stats.l1d_stats.reduce());
        // dbg!(&kernel_stats.l1d_stats.reduce().num_accesses());
        // dbg!(&kernel_stats
        //     .l1d_stats
        //     .reduce()
        //     .iter()
        //     .map(|(_, num)| num)
        //     .sum::<usize>());
        // dbg!(&kernel_stats.l2d_stats);
        // dbg!(&kernel_stats.l2d_stats.reduce());
        // dbg!(&kernel_stats.l2d_stats.reduce().num_accesses());
        // dbg!(&kernel_stats.dram.reduce());
        // dbg!(&kernel_stats.sim);
        eprintln!("SIM: {:#?}", &kernel_stats.sim);
        eprintln!("DRAM: {:#?}", &kernel_stats.dram.reduce());
        eprintln!("L1D: {:#?}", &kernel_stats.l1d_stats.reduce());
        eprintln!("L2D: {:#?}", &kernel_stats.l2d_stats.reduce());
        eprintln!(
            "L2D hit rate: {:4.2}% ({} hits / {} accesses)",
            &kernel_stats.l2d_stats.reduce().hit_rate() * 100.0,
            &kernel_stats.l2d_stats.reduce().num_hits(),
            &kernel_stats.l2d_stats.reduce().num_accesses(),
        );
        eprintln!(
            "L2D write hit rate: {:4.2}% ({} write hits / {} writes)",
            &kernel_stats.l2d_stats.reduce().write_hit_rate() * 100.0,
            &kernel_stats.l2d_stats.reduce().num_write_hits(),
            &kernel_stats.l2d_stats.reduce().num_writes(),
        );
    }

    // let reduced = stats.clone().reduce();
    // dbg!(&reduced.dram.reduce());
    let num_kernels_launched = stats.inner.len();
    dbg!(num_kernels_launched);

    // *wip_stats = gpucachesim::WIPStats::default();

    Ok(sim)
}

pub fn process_stats(
    stats: &[stats::Stats],
    _dur: &std::time::Duration,
    stats_dir: &Path,
    repetition: usize,
    full: bool,
) -> Result<(), RunError> {
    create_dirs(stats_dir).map_err(eyre::Report::from)?;
    crate::stats::write_stats_as_csv(stats_dir, stats, repetition, full)?;
    Ok(())
}

pub async fn simulate(
    bench: BenchmarkConfig,
    options: &Options,
    _sim_options: &options::Sim,
    _bar: &indicatif::ProgressBar,
) -> Result<Duration, RunError> {
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

    let mut total_dur = Duration::ZERO;
    for repetition in 0..bench.common.repetitions {
        let bench = bench.clone();
        let (sim, dur) = tokio::task::spawn_blocking(move || {
            let start = Instant::now();
            let stats = simulate_bench_config(&bench)?;
            Ok::<_, eyre::Report>((stats, start.elapsed()))
        })
        .await
        .map_err(eyre::Report::from)??;

        total_dur += dur;
        let stats = sim.stats();
        let full = false;
        process_stats(stats.as_ref(), &dur, stats_dir, repetition, full)?;
    }
    Ok(total_dur)
}

pub mod exec {
    use crate::materialized::{BenchmarkConfig, TargetBenchmarkConfig};
    use crate::{
        options::{self, Options},
        RunError, Target,
    };
    use color_eyre::{eyre, Help};
    use gpucachesim_benchmarks as benchmarks;
    use std::time::Duration;
    use utils::fs::create_dirs;

    pub async fn simulate(
        bench: BenchmarkConfig,
        options: &Options,
        _sim_options: &options::Sim,
        _bar: &indicatif::ProgressBar,
    ) -> Result<Duration, RunError> {
        let (TargetBenchmarkConfig::Simulate { ref stats_dir, parallel, l2_prefill, .. } | TargetBenchmarkConfig::ExecDrivenSimulate { ref stats_dir, parallel, l2_prefill, .. }) = bench.target_config else {
            unreachable!();
        };
        if options.clean {
            utils::fs::remove_dir(&stats_dir).map_err(eyre::Report::from)?;
        }

        create_dirs(&stats_dir).map_err(eyre::Report::from)?;
        if !options.force && crate::stats::already_exist(&bench.common, &stats_dir) {
            return Err(RunError::Skipped);
        }

        let parse_err = |err: serde_json::Error| {
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
        let values: serde_json::Value = serde_json::to_value(&bench.values).map_err(parse_err)?;
        // dbg!(&values);

        let (commands, kernel_traces) = match bench.name.to_lowercase().as_str() {
            "vectoradd" => {
                #[derive(Debug, serde::Deserialize)]
                struct VectoraddInput {
                    dtype: usize,
                    length: usize,
                }
                let VectoraddInput { dtype, length } =
                    serde_json::from_value(values.clone()).map_err(parse_err)?;

                match dtype {
                    32 => benchmarks::vectoradd::benchmark::<f32>(length).await,
                    64 => benchmarks::vectoradd::benchmark::<f64>(length).await,
                    other => return Err(RunError::Failed(eyre::eyre!("invalid dtype {other:?}"))),
                }
            }
            "simple_matrixmul" => {
                #[derive(Debug, serde::Deserialize)]
                struct SimpleMatrixmulInput {
                    dtype: usize,
                    m: usize,
                    n: usize,
                    p: usize,
                }
                let SimpleMatrixmulInput { dtype, m, n, p } =
                    serde_json::from_value(values.clone()).map_err(parse_err)?;

                match dtype {
                    32 => benchmarks::simple_matrixmul::benchmark::<f32>(m, n, p).await,
                    64 => benchmarks::simple_matrixmul::benchmark::<f64>(m, n, p).await,
                    other => return Err(RunError::Failed(eyre::eyre!("invalid dtype {other:?}"))),
                }
            }
            "matrixmul" => {
                #[derive(Debug, serde::Deserialize)]
                struct MatrixmulInput {
                    dtype: usize,
                    rows: usize,
                }
                let MatrixmulInput { dtype, rows } =
                    serde_json::from_value(values.clone()).map_err(parse_err)?;

                match dtype {
                    32 => benchmarks::matrixmul::benchmark::<f32>(rows).await,
                    64 => benchmarks::matrixmul::benchmark::<f64>(rows).await,
                    other => return Err(RunError::Failed(eyre::eyre!("invalid dtype {other:?}"))),
                }
            }
            "transpose" => {
                #[derive(Debug, serde::Deserialize)]
                struct TransposeInput {
                    dim: usize,
                    repeat: Option<usize>,
                    variant: benchmarks::transpose::Variant,
                }
                let TransposeInput {
                    dim,
                    variant,
                    repeat,
                } = serde_json::from_value(values.clone()).map_err(parse_err)?;
                benchmarks::transpose::benchmark::<f32>(dim, variant, repeat.unwrap_or(0)).await
            }
            "babelstream" => return Err(RunError::Skipped),
            other => {
                return Err(RunError::Failed(eyre::eyre!(
                    "unknown benchmark: {}",
                    other
                )))
            }
        }?;

        let traces_dir = stats_dir.join("traces");
        gpucachesim::exec::write_traces(commands, kernel_traces, &traces_dir)?;

        let mut total_dur = Duration::ZERO;
        for repetition in 0..bench.common.repetitions {
            let bench = BenchmarkConfig {
                target: Target::Simulate,
                target_config: TargetBenchmarkConfig::Simulate {
                    traces_dir: traces_dir.clone(),
                    stats_dir: stats_dir.clone(),
                    accelsim_traces_dir: traces_dir.clone(),
                    parallel,
                    l2_prefill,
                },
                ..bench.clone()
            };
            let (sim, dur) = tokio::task::spawn_blocking(move || {
                let start = std::time::Instant::now();
                let stats = super::simulate_bench_config(&bench)?;
                Ok::<_, eyre::Report>((stats, start.elapsed()))
            })
            .await
            .map_err(eyre::Report::from)??;

            total_dur += dur;
            let stats = sim.stats();
            let full = false;
            super::process_stats(stats.as_ref(), &dur, &stats_dir, repetition, full)?;
        }
        Ok(total_dur)
    }
}
