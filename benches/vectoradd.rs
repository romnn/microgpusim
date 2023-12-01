#![allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]

use clap::{Parser, Subcommand};
use color_eyre::eyre;
use criterion::{black_box, Criterion};
use gpucachesim::config::Parallelization;
use std::sync::Arc;
use std::time::Duration;
use validate::{
    benchmark::Input,
    input,
    materialized::{BenchmarkConfig, TargetBenchmarkConfig},
    Target, TraceProvider,
};

#[derive(Debug, Subcommand, strum::EnumIter)]
enum Command {
    Accelsim,
    Playground,
    Serial,
    Deterministic,
    Nondeterministic {
        run_ahead: Option<usize>,
        interleaved: Option<bool>,
    },
}

#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
struct Options {
    #[clap(subcommand)]
    pub command: Option<Command>,

    #[arg(long = "threads", help = "number of threads")]
    pub threads: Option<usize>,
}

pub fn run_box(
    bench_config: &BenchmarkConfig,
    parallelization: Parallelization,
    threads: Option<usize>,
) -> eyre::Result<stats::PerKernel> {
    use gpucachesim::config::{gtx1080::build_config, Input, GPU};
    let TargetBenchmarkConfig::Simulate { ref traces_dir, .. } = bench_config.target_config else {
        unreachable!();
    };

    // if let TargetBenchmarkConfig::Simulate {
    //     ref mut parallel, ..
    // } = bench_config.target_config
    // {
    //     *parallel = Some(!serial);
    // }
    // if std::env::var("PARALLEL").unwrap_or_default().to_lowercase() == "yes";
    // println!("parallel: {}", bench_config.simulate.parallel);
    // let sim = validate::simulate::simulate_bench_config(&*bench_config)?;
    // let config = gpucachesim::config::GPU {
    //     num_simt_clusters: 20,                       // 20
    //     num_cores_per_simt_cluster: 1,               // 1
    //     num_schedulers_per_core: 2,                  // 1
    //     num_memory_controllers: 8,                   // 8
    //     num_dram_chips_per_memory_controller: 1,     // 1
    //     num_sub_partitions_per_memory_controller: 2, // 2
    //     fill_l2_on_memcopy: false,                   // true
    //     parallelization,
    //     log_after_cycle: None,
    //     simulation_threads: None, // can use env variables still
    //     ..gpucachesim::config::GPU::default()
    // };

    let mut config: GPU = build_config(&Input::default())?;
    config.parallelization = parallelization;
    config.simulation_threads = threads;
    config.fill_l2_on_memcopy = false;
    assert!(!gpucachesim::is_debug());
    let sim = gpucachesim::accelmain(traces_dir, config)?;

    // fast parallel:   cycle loop time: 558485 ns
    // serial:          cycle loop time: 2814591 ns (speedup 5x)
    // have 80 cores and 16 threads
    //
    // parallel: dram cycle time: 229004 ns
    // println!();
    // let timings = gpucachesim::TIMINGS.lock().unwrap();
    // let total = timings["total_cycle"].mean();
    // for (name, dur) in [
    //     ("core cycle", &timings["core_cycle"]),
    //     ("icnt cycle", &timings["icnt_cycle"]),
    //     ("dram cycle", &timings["dram_cycle"]),
    //     ("l2 cycle", &timings["l2_cycle"]),
    // ] {
    //     let dur = dur.mean();
    //     let percent = (dur.as_secs_f64() / total.as_secs_f64()) * 100.0;
    //     let ms = dur.as_secs_f64() * 1000.0;
    //     println!("{name} time: {ms:.5} ms ({percent:>2.2}%)");
    // }
    // println!();
    let stats = sim.stats();
    Ok(stats)
}

pub async fn run_accelsim(bench_config: Arc<BenchmarkConfig>) -> eyre::Result<accelsim::Stats> {
    assert!(!validate::accelsim::is_debug());
    let (log, _dur) = validate::accelsim::simulate_bench_config(&bench_config).await?;
    let parse_options = accelsim::parser::Options::default();
    let log_reader = std::io::Cursor::new(log.stdout);
    let stats = accelsim::Stats {
        is_release_build: !validate::accelsim::is_debug(),
        ..accelsim::parser::parse_stats(log_reader, &parse_options)?
    };

    Ok(stats)
}

pub fn run_playground(
    bench_config: &BenchmarkConfig,
) -> eyre::Result<(String, playground::stats::Stats, Duration)> {
    let accelsim_compat_mode = false;
    let extra_args: &[String] = &[];
    assert!(!validate::playground::is_debug());
    let (log, stats, dur) = validate::playground::simulate_bench_config(
        bench_config,
        TraceProvider::Box,
        extra_args,
        accelsim_compat_mode,
    )?;
    Ok((log, stats, dur))
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
    use itertools::Itertools;
    let timings = gpucachesim::TIMINGS.lock();
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
    use std::time::Instant;
    use validate::benchmark::find_first;

    color_eyre::install()?;

    // let options = Options::parse();
    let options = Options {
        command: None,
        threads: None,
    };

    // takes 34 sec (accel same)
    let (bench_name, input_query): (_, Input) =
        ("transpose", input!({ "dim": 256, "variant": "naive"})?);
    let (bench_name, input_query): (_, Input) = ("vectorAdd", input!({ "length": 500_000 })?);

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
        Some(Command::Serial | Command::Nondeterministic { .. } | Command::Deterministic) => {
            configure_tracing()
        }
        _ => None,
    };

    let commands = options
        .command
        .map(|cmd| vec![cmd])
        // .unwrap_or(<Command as strum::IntoEnumIterator>::iter().collect());
        .unwrap_or(vec![
            Command::Accelsim,
            Command::Playground,
            Command::Serial,
            Command::Deterministic,
            Command::Nondeterministic {
                run_ahead: Some(10),
                interleaved: Some(false),
            },
            Command::Nondeterministic {
                run_ahead: Some(10),
                interleaved: Some(true),
            },
        ]);

    let mut box_baseline: Option<Duration> = None;
    for cmd in commands {
        // clear timing measurements
        gpucachesim::TIMINGS.lock().clear();

        match cmd {
            Command::Accelsim => {
                let bench_config = Arc::new(
                    find_first(Target::AccelsimSimulate, bench_name, &input_query)?.unwrap(),
                );
                println!("running {}@{}", bench_config.name, bench_config.input_idx);
                let start = Instant::now();
                let stats = runtime.block_on(async {
                    let stats = run_accelsim(black_box(bench_config)).await?;
                    let stats: stats::PerKernel = stats.try_into()?;
                    Ok::<_, eyre::Report>(stats)
                })?;
                let accel_dur = start.elapsed();
                dbg!(&stats.reduce().sim);
                println!("accelsim took: {accel_dur:?}");
            }
            Command::Playground => {
                let bench_config =
                    find_first(Target::PlaygroundSimulate, bench_name, &input_query)?.unwrap();
                println!("running {}@{}", bench_config.name, bench_config.input_idx);
                let (_, stats, play_dur) = run_playground(black_box(&bench_config))?;
                dbg!(&stats::Stats::from(stats).sim);
                println!("playground took: {play_dur:?}");
            }
            Command::Serial => {
                let bench_config = find_first(Target::Simulate, bench_name, &input_query)?.unwrap();
                println!("running {}@{}", bench_config.name, bench_config.input_idx);
                let start = Instant::now();
                let stats = run_box(
                    black_box(&bench_config),
                    Parallelization::Serial,
                    options.threads,
                )?;
                let serial_box_dur = start.elapsed();
                dbg!(&stats.reduce().sim);
                // for (kernel_launch_id, kernel_stats) in stats.as_ref().iter().enumerate() {
                //     dbg!(kernel_launch_id);
                //     dbg!(&kernel_stats.sim);
                // }
                box_baseline = Some(serial_box_dur);
                println!("box[serial] took: {serial_box_dur:?}");
                if let Some(baseline) = box_baseline {
                    println!(
                        "speedup is: {:.2}x",
                        baseline.as_secs_f64() / serial_box_dur.as_secs_f64()
                    );
                }
            }
            Command::Deterministic => {
                let bench_config = find_first(Target::Simulate, bench_name, &input_query)?.unwrap();
                println!("running {}@{}", bench_config.name, bench_config.input_idx);
                let start = Instant::now();
                let stats = run_box(
                    black_box(&bench_config),
                    Parallelization::Deterministic,
                    options.threads,
                )?;
                let box_dur = start.elapsed();
                println!("box[deterministic] took: {box_dur:?}");
                if let Some(baseline) = box_baseline {
                    println!(
                        "speedup is: {:.2}x",
                        baseline.as_secs_f64() / box_dur.as_secs_f64()
                    );
                }
            }
            Command::Nondeterministic {
                run_ahead,
                interleaved,
            } => {
                let start = Instant::now();
                let bench_config = find_first(Target::Simulate, bench_name, &input_query)?.unwrap();
                let stats = run_box(
                    black_box(&bench_config),
                    Parallelization::Nondeterministic {
                        run_ahead: run_ahead.unwrap_or(5),
                        interleave: interleaved.unwrap_or(false),
                    },
                    options.threads,
                )?;
                let box_dur = start.elapsed();
                println!("box[nondeterministic][run_ahead={run_ahead:?}][interleave={interleaved:?}] took: {box_dur:?}");
                if let Some(baseline) = box_baseline {
                    println!(
                        "speedup is: {:.2}x",
                        baseline.as_secs_f64() / box_dur.as_secs_f64()
                    );
                }
            }
        }

        print_timings();
    }

    tracing_guard.take();

    Ok(())
}
