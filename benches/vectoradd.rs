#![allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]

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

// fn get_bench_config(
//     target: Target,
//     benchmark_name: &str,
//     input_query: validate::benchmark::Input,
//     // input_idx: usize,
// ) -> eyre::Result<BenchmarkConfig> {
//     use std::path::PathBuf;
//
//     let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
//     let benchmarks_path = manifest_dir.join("test-apps/test-apps-materialized.yml");
//     let reader = utils::fs::open_readable(benchmarks_path)?;
//     let benchmarks = Benchmarks::from_reader(reader)?;
//     let bench_config = benchmarks
//         .find_all(target, benchmark_name, input_idx)
//         // .get_single_config(target, benchmark_name, input_idx)
//         // .ok_or_else(|| {
//         //     eyre::eyre!(
//         //         "no benchmark {:?} or input index {}",
//         //         benchmark_name,
//         //         input_idx
//         //     )
//         // })?;
//     Ok(bench_config.clone())
// }

pub fn run_box(
    bench_config: Arc<BenchmarkConfig>,
    parallelization: Parallelization,
) -> eyre::Result<stats::PerKernel> {
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
        simulation_threads: None, // can use env variables still
        ..gpucachesim::config::GPU::default()
    };

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
    bench_config: Arc<BenchmarkConfig>,
) -> eyre::Result<(String, playground::stats::Stats, Duration)> {
    let accelsim_compat_mode = false;
    let extra_args: &[String] = &[];
    assert!(!validate::playground::is_debug());
    let (log, stats, dur) = validate::playground::simulate_bench_config(
        &bench_config,
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
    // group.bench_function("transpose/256/naive", |b| {
    //     b.iter(|| run_accelsim(black_box(get_bench_config("transpose", 0).unwrap())))
    // });
}

pub fn play_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("play");
    group.sample_size(10);
    group.sampling_mode(criterion::SamplingMode::Flat);

    let bench_config = validate::benchmark::find_all(
        Target::PlaygroundSimulate,
        "vectorAdd",
        &input!({ "dtype": 32, "length": 10000 }).unwrap(),
    )
    .unwrap()
    .into_iter()
    .next()
    .unwrap();
    let bench_config = Arc::new(bench_config);
    group.bench_function("vectoradd/10000", |b| {
        b.iter(|| run_playground(black_box(bench_config.clone())));
    });
    // group.bench_function("transpose/256/naive", |b| {
    //     b.iter(|| run_playground(black_box(get_bench_config("transpose", 0).unwrap())))
    // });
}

pub fn box_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("box");
    group.sample_size(10);
    group.sampling_mode(criterion::SamplingMode::Flat);

    let bench_config = validate::benchmark::find_all(
        Target::Simulate,
        "vectorAdd",
        &input!({ "dtype": 32, "length": 10000 }).unwrap(),
    )
    .unwrap()
    .into_iter()
    .next()
    .unwrap();
    let bench_config = Arc::new(bench_config);
    group.bench_function("vectoradd/10000", |b| {
        b.iter(|| run_box(black_box(bench_config.clone()), Parallelization::Serial));
    });
    // group.bench_function("transpose/256/naive", |b| {
    //     b.iter(|| run_box(black_box(get_bench_config("transpose", 0).unwrap())))
    // });
}

criterion::criterion_group!(benches, box_benchmark, play_benchmark, accelsim_benchmark);
// criterion::criterion_main!(benches);

#[allow(dead_code)]
fn main() -> eyre::Result<()> {
    use itertools::Itertools;
    #[allow(unused_imports)]
    use std::io::Write;
    use std::time::Instant;
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    color_eyre::install()?;

    let profile = std::env::var("TRACE").unwrap_or_default().to_lowercase() == "yes";

    let mut generate_trace = if profile {
        // tracing_subscriber::fmt::init();
        let (chrome_layer, guard) = ChromeLayerBuilder::new().file("bench.trace.json").build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        env_logger::init();
        // let mut log_builder = env_logger::Builder::new();
        // log_builder.format(|buf, record| writeln!(buf, "{}", record.args()));
        None
    };

    // takes 34 sec (accel same)
    let (bench_name, input_query): (_, Input) =
        ("transpose", input!({ "dim": 256, "variant": "naive"})?);
    // let (bench_name, input_num) = ("simple_matrixmul", 26); // takes 22 sec
    // let (bench_name, input_num) = ("matrixmul", 3); // takes 54 sec (accel 76)
    // let (bench_name, input_query) = (
    //     "vectorAdd",
    //     input!({ "dtype": 32, "length": 10000 })?,
    // );

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    let non_deterministic: Option<usize> = std::env::var("NONDET")
        .ok()
        .as_deref()
        .map(str::parse)
        .transpose()?;
    let parallelization = match non_deterministic {
        None => Parallelization::Deterministic,
        Some(run_ahead) => Parallelization::Nondeterministic(run_ahead),
    };

    let start = Instant::now();
    let bench_config = validate::benchmark::find_all(Target::Simulate, bench_name, &input_query)?
        .into_iter()
        .next()
        .unwrap();
    let bench_config = Arc::new(bench_config);
    println!("running {}@{}", bench_config.name, bench_config.input_idx);

    let stats = run_box(black_box(bench_config), parallelization)?;
    dbg!(&stats.reduce().sim);
    // for (kernel_launch_id, kernel_stats) in stats.as_ref().iter().enumerate() {
    //     dbg!(kernel_launch_id);
    //     dbg!(&kernel_stats.sim);
    // }
    let box_dur = start.elapsed();
    println!("box took:\t\t{box_dur:?}");

    drop(generate_trace.take());
    if profile {
        return Ok(());
    }

    {
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

    // clear timing measurements
    gpucachesim::TIMINGS.lock().clear();

    let bench_config = validate::benchmark::find_all(Target::Simulate, bench_name, &input_query)?
        .into_iter()
        .next()
        .unwrap();
    let bench_config = Arc::new(bench_config);
    let start = Instant::now();
    let stats = run_box(black_box(bench_config), Parallelization::Serial)?;
    dbg!(&stats.reduce().sim);
    // for (kernel_launch_id, kernel_stats) in stats.as_ref().iter().enumerate() {
    //     dbg!(kernel_launch_id);
    //     dbg!(&kernel_stats.sim);
    // }
    let serial_box_dur = start.elapsed();
    println!("serial box took:\t\t{serial_box_dur:?}");
    println!(
        "speedup is :\t\t{:.2}",
        serial_box_dur.as_secs_f64() / box_dur.as_secs_f64()
    );
    {
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

    let bench_config =
        validate::benchmark::find_all(Target::PlaygroundSimulate, bench_name, &input_query)?
            .into_iter()
            .next()
            .unwrap();
    let bench_config = Arc::new(bench_config);
    let (_, stats, play_dur) = run_playground(black_box(bench_config))?;
    dbg!(&stats::Stats::from(stats).sim);
    println!("play took:\t\t{play_dur:?}");

    let bench_config =
        validate::benchmark::find_all(Target::AccelsimSimulate, bench_name, &input_query)?
            .into_iter()
            .next()
            .unwrap();
    let bench_config = Arc::new(bench_config);
    let start = Instant::now();
    let stats = runtime.block_on(async {
        let stats = run_accelsim(black_box(bench_config)).await?;
        let stats: stats::Stats = stats.try_into()?;
        Ok::<_, eyre::Report>(stats)
    })?;
    dbg!(&stats.sim);

    let accel_dur = start.elapsed();
    println!("accel took:\t\t{accel_dur:?}");

    Ok(())
}
