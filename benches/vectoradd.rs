#![allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]

use clap::{Parser, Subcommand};
use color_eyre::eyre;
use console::{style, Style};
use criterion::{black_box, Criterion};
use gpucachesim::config::Parallelization;
use itertools::Itertools;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use validate::benchmark::find_first;
use validate::{
    benchmark::Input,
    input,
    materialized::{BenchmarkConfig, TargetBenchmarkConfig},
    Target, TraceProvider,
};

#[derive(Debug, Clone, Copy, Subcommand, Hash, PartialEq, Eq, PartialOrd, Ord, strum::EnumIter)]
enum TargetCommand {
    Accelsim,
    Playground,
    Serial,
    Deterministic,
    Nondeterministic { run_ahead: usize },
}

impl std::fmt::Display for TargetCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TargetCommand::Accelsim => write!(f, "accelsim"),
            TargetCommand::Playground => write!(f, "playground"),
            TargetCommand::Serial => write!(f, "gpucachesim[serial]"),
            TargetCommand::Deterministic => write!(f, "gpucachesim[deterministic]"),
            TargetCommand::Nondeterministic { run_ahead } => {
                write!(f, "gpucachesim[nondeterministic({run_ahead})]")
            }
        }
    }
}

impl From<TargetCommand> for Target {
    fn from(value: TargetCommand) -> Self {
        match value {
            TargetCommand::Accelsim => Target::AccelsimSimulate,
            TargetCommand::Playground => Target::PlaygroundSimulate,
            TargetCommand::Serial
            | TargetCommand::Deterministic
            | TargetCommand::Nondeterministic { .. } => Target::Simulate,
        }
    }
}

impl TargetCommand {
    pub fn is_gpucachesim(&self) -> bool {
        matches!(
            self,
            TargetCommand::Serial
                | TargetCommand::Deterministic
                | TargetCommand::Nondeterministic { .. }
        )
    }
}

#[derive(thiserror::Error, Debug)]
#[error("invalid target {0:?}")]
pub struct InvalidTarget(String);

impl std::str::FromStr for TargetCommand {
    type Err = InvalidTarget;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().trim() {
            "serial" => Ok(TargetCommand::Serial),
            "play" | "playground" => Ok(TargetCommand::Playground),
            "accel" | "accelsim" => Ok(TargetCommand::Accelsim),
            "det" | "deterministic" => Ok(TargetCommand::Deterministic),
            "nondet" | "nondeterministic" => Ok(TargetCommand::Nondeterministic { run_ahead: 10 }),
            other => Err(InvalidTarget(other.to_string())),
        }
    }
}

#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
struct Options {
    #[clap(subcommand)]
    pub command: Option<TargetCommand>,

    #[arg(long = "threads", help = "number of threads")]
    pub threads: Option<usize>,
}

pub fn run_box(
    bench_config: &BenchmarkConfig,
    parallelization: Parallelization,
    threads: Option<usize>,
) -> eyre::Result<(stats::PerKernel, Duration)> {
    use gpucachesim::config::{gtx1080::build_config, Input, GPU};
    let TargetBenchmarkConfig::Simulate { ref traces_dir, .. } = bench_config.target_config else {
        unreachable!();
    };

    let mut config: GPU = build_config(&Input::default())?;
    config.parallelization = parallelization;
    config.simulation_threads = threads;
    config.fill_l2_on_memcopy = false;
    config.perfect_inst_const_cache = true;
    assert!(!gpucachesim::is_debug());
    let start = Instant::now();
    let sim = gpucachesim::accelmain(traces_dir, config)?;
    let dur = start.elapsed();

    let stats = sim.stats();
    Ok((stats, dur))
}

pub async fn run_accelsim(
    bench_config: Arc<BenchmarkConfig>,
) -> eyre::Result<(String, accelsim::Stats, Duration)> {
    assert!(!validate::accelsim::is_debug());
    let (log, dur) = validate::accelsim::simulate_bench_config(&bench_config).await?;
    let parse_options = accelsim::parser::Options::default();
    let log_reader = std::io::Cursor::new(&log.stdout);
    let stats = accelsim::Stats {
        is_release_build: !validate::accelsim::is_debug(),
        ..accelsim::parser::parse_stats(log_reader, &parse_options)?
    };

    Ok((utils::decode_utf8!(&log.stdout), stats, dur))
}

pub fn run_playground(
    bench_config: &BenchmarkConfig,
) -> eyre::Result<(String, playground::stats::Stats, Duration)> {
    let accelsim_compat_mode = false;
    let extra_args: &[String] = &[];
    assert!(!validate::playground::is_debug());
    let result = validate::playground::simulate_bench_config(
        bench_config,
        TraceProvider::Box,
        extra_args,
        accelsim_compat_mode,
    )?;
    Ok(result)
}

pub fn accelsim_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("accelsim");
    group.sample_size(10);
    group.sampling_mode(criterion::SamplingMode::Flat);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");

    let input: Input = input!({ "dtype": 32, "length": 10000 }).unwrap();
    let bench_config = validate::benchmark::find_all(Target::AccelsimSimulate, "vectorAdd", &input)
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    let bench_config = Arc::new(bench_config);
    group.bench_function("vectoradd/10000", |b| {
        b.to_async(&runtime)
            .iter(|| run_accelsim(black_box(bench_config.clone())));
    });
}

pub fn play_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("play");
    group.sample_size(10);
    group.sampling_mode(criterion::SamplingMode::Flat);

    let bench_config = validate::benchmark::find_first(
        Target::PlaygroundSimulate,
        "vectorAdd",
        &input!({ "dtype": 32, "length": 10000 }).unwrap(),
    )
    .unwrap()
    .unwrap();
    group.bench_function("vectoradd/10000", |b| {
        b.iter(|| run_playground(black_box(&bench_config)));
    });
}

pub fn box_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("box");
    group.sample_size(10);
    group.sampling_mode(criterion::SamplingMode::Flat);

    let bench_config = validate::benchmark::find_first(
        Target::Simulate,
        "vectorAdd",
        &input!({ "dtype": 32, "length": 10000 }).unwrap(),
    )
    .unwrap()
    .unwrap();
    group.bench_function("vectoradd/10000", |b| {
        b.iter(|| run_box(black_box(&bench_config), Parallelization::Serial, None));
    });
}

fn print_timings() {
    let timings = gpucachesim::TIMINGS.lock();
    if timings.is_empty()
        || timings
            .iter()
            .map(|(_, dur)| dur.total())
            .sum::<Duration>()
            .is_zero()
    {
        return;
    }

    println!("sorted by NAME");
    for (name, dur) in timings.iter().sorted_by_key(|(&name, _dur)| name) {
        println!(
            "\t{name:<30}: {:>6.5} ms avg ({:>2.6} sec total)",
            dur.mean().as_secs_f64() * 1000.0,
            dur.total().as_secs_f64(),
        );
    }
    println!();
    println!("sorted by TOTAL DURATION");
    for (name, dur) in timings.iter().sorted_by_key(|(_name, dur)| dur.total()) {
        println!(
            "\t{name:<30}: {:>6.5} ms avg ({:>2.6} sec total)",
            dur.mean().as_secs_f64() * 1000.0,
            dur.total().as_secs_f64(),
        );
    }
    println!();
}

fn configure_tracing() -> Option<tracing_chrome::FlushGuard> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let enable_tracing = std::env::var("TRACE").unwrap_or_default().to_lowercase() == "yes";
    if enable_tracing {
        // tracing_subscriber::fmt::init();
        let (chrome_layer, guard) = ChromeLayerBuilder::new().file("bench.trace.json").build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        gpucachesim::init_logging();
        None
    }
}

criterion::criterion_group!(benches, box_benchmark, play_benchmark, accelsim_benchmark);
// criterion::criterion_main!(benches);

#[allow(dead_code, clippy::too_many_lines)]
fn main() -> eyre::Result<()> {
    #[allow(unused_imports)]
    use std::io::Write;

    color_eyre::install()?;

    // clap parsing does not work when running "cargo bench ..."
    // let options = Options::parse();
    //
    let command: Option<TargetCommand> = std::env::var("TARGET")
        .ok()
        .as_deref()
        .map(std::str::FromStr::from_str)
        .transpose()?;
    let threads = std::env::var("THREADS")
        .ok()
        .as_deref()
        .map(str::parse)
        .transpose()?;

    let options = Options { command, threads };

    // takes 34 sec (accel same)
    let (bench_name, input_query): (_, Input) =
        ("transpose", input!({ "dim": 256, "variant": "naive"})?);

    // let (bench_name, input_query): (_, Input) = ("vectorAdd", input!({ "length": 500_000 })?);

    let (bench_name, input_query): (_, Input) =
        ("vectorAdd", input!({ "dtype": 32, "length": 100 })?);
    // let (bench_name, input_query): (_, Input) =
    //     ("vectorAdd", input!({ "dtype": 32, "length": 1_000 })?);
    // let (bench_name, input_query): (_, Input) =
    //     ("vectorAdd", input!({ "dtype": 32, "length": 500_000 })?);

    // let (bench_name, input_num) = ("simple_matrixmul", 26); // takes 22 sec
    // let (bench_name, input_num) = ("matrixmul", 3); // takes 54 sec (accel 76)
    // let (bench_name, input_query) = (
    //     "vectorAdd",
    //     input!({ "dtype": 32, "length": 10000 })?,
    // );

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    let mut tracing_guard = match options.command {
        Some(
            TargetCommand::Serial
            | TargetCommand::Nondeterministic { .. }
            | TargetCommand::Deterministic,
        ) => configure_tracing(),
        _ => None,
    };

    let commands = options.command.map(|cmd| vec![cmd]).unwrap_or(vec![
        TargetCommand::Accelsim,
        // TargetCommand::Playground,
        TargetCommand::Serial,
        // TargetCommand::Deterministic,
        // TargetCommand::Nondeterministic { run_ahead: 10 },
    ]);

    #[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
    struct TargetStats {
        pub duration: Duration,
        pub cycles: u64,
    }

    let mut target_stats: HashMap<TargetCommand, TargetStats> = HashMap::new();

    for cmd in commands.into_iter().dedup() {
        // clear timing measurements
        gpucachesim::TIMINGS.lock().clear();

        let bench_config = find_first(cmd.into(), bench_name, &input_query)?.unwrap();
        println!(
            "{}: running {}",
            style(cmd).cyan(),
            style(bench_config.to_string()).cyan()
        );

        let stats = match cmd {
            TargetCommand::Accelsim => {
                let bench_config = Arc::new(bench_config);
                let (_, stats, duration) =
                    runtime.block_on(run_accelsim(black_box(bench_config)))?;
                let stats: stats::PerKernel = stats.try_into()?;
                TargetStats {
                    duration,
                    cycles: stats.reduce().sim.cycles,
                }
            }
            TargetCommand::Playground => {
                let (_, stats, duration) = run_playground(black_box(&bench_config))?;
                let stats = stats::Stats::from(stats);
                TargetStats {
                    duration,
                    cycles: stats.sim.cycles,
                }
            }
            TargetCommand::Serial => {
                let (stats, duration) = run_box(
                    black_box(&bench_config),
                    Parallelization::Serial,
                    options.threads,
                )?;
                TargetStats {
                    duration,
                    cycles: stats.reduce().sim.cycles,
                }
            }
            TargetCommand::Deterministic => {
                let (stats, duration) = run_box(
                    black_box(&bench_config),
                    Parallelization::Deterministic,
                    options.threads,
                )?;
                TargetStats {
                    duration,
                    cycles: stats.reduce().sim.cycles,
                }
            }
            TargetCommand::Nondeterministic { run_ahead } => {
                let (stats, duration) = run_box(
                    black_box(&bench_config),
                    Parallelization::Nondeterministic { run_ahead },
                    options.threads,
                )?;
                TargetStats {
                    duration,
                    cycles: stats.reduce().sim.cycles,
                }
            }
        };

        println!(
            "{}",
            style(format!("{} took {:?}", cmd, stats.duration)).red()
        );
        target_stats.insert(cmd, stats);

        if cmd.is_gpucachesim() {
            print_timings();
        }
    }

    tracing_guard.take();

    let sorted_target_stats: Vec<_> = target_stats
        .iter()
        .sorted_by_key(|(_, stats)| stats.duration)
        .collect();

    println!("\n\n==== RESULTS (fastest to slowest) ====");
    for (target, stats) in sorted_target_stats {
        let accelsim_baseline_duration = target_stats
            .get(&TargetCommand::Accelsim)
            .map(|stats| stats.duration);
        let serial_baseline_duration = target_stats
            .get(&TargetCommand::Serial)
            .map(|stats| stats.duration);

        let speedup_over_accelsim = accelsim_baseline_duration
            .map(|baseline| baseline.as_secs_f64() / stats.duration.as_secs_f64())
            .unwrap_or(0.0);
        let speedup_over_serial = serial_baseline_duration
            .map(|baseline| baseline.as_secs_f64() / stats.duration.as_secs_f64())
            .unwrap_or(0.0);

        let target_color = |target: &TargetCommand| -> Style {
            if target.is_gpucachesim() {
                Style::new().cyan()
            } else {
                Style::new()
            }
        };

        let speedup_color = |speedup: f64| -> Style {
            if speedup > 1.0 {
                Style::new().green()
            } else if speedup == 1.0 || speedup == 0.0 {
                Style::new()
            } else {
                Style::new().red()
            }
        };

        print!(
            "     {}{:>20?}\t{:>10} cycles",
            target_color(target).apply_to(format!("{: <45}", target.to_string())),
            stats.duration,
            stats.cycles,
        );
        if target.is_gpucachesim() {
            print!(
                "\t=> speedup(serial)={}\tspeedup(accelsim)={}",
                speedup_color(speedup_over_serial)
                    .apply_to(format!("{: >6.3}x", speedup_over_serial)),
 
                speedup_color(speedup_over_accelsim)
                    .apply_to(format!("{: >6.3}x", speedup_over_accelsim)),
            );
        }

        print!("\n");
    }
    Ok(())
}
