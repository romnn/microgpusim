#![allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]

use color_eyre::eyre;
use criterion::{black_box, Criterion};
use std::time::Duration;
use validate::{
    materialize::{BenchmarkConfig, Benchmarks},
    TraceProvider,
};

fn get_bench_config(benchmark_name: &str, input_idx: usize) -> eyre::Result<BenchmarkConfig> {
    use std::path::PathBuf;

    let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
    let benchmarks_path = manifest_dir.join("test-apps/test-apps-materialized.yml");
    let reader = utils::fs::open_readable(benchmarks_path)?;
    let benchmarks = Benchmarks::from_reader(reader)?;
    let bench_config = benchmarks
        .get_single_config(benchmark_name, input_idx)
        .ok_or_else(|| {
            eyre::eyre!(
                "no benchmark {:?} or input index {}",
                benchmark_name,
                input_idx
            )
        })?;
    Ok(bench_config.clone())
}

pub fn run_box(mut bench_config: BenchmarkConfig, serial: bool) -> eyre::Result<stats::Stats> {
    bench_config.simulate.parallel = !serial;
    // if std::env::var("PARALLEL").unwrap_or_default().to_lowercase() == "yes";
    // println!("parallel: {}", bench_config.simulate.parallel);
    let sim = validate::simulate::simulate_bench_config(&bench_config)?;
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

pub async fn run_accelsim(bench_config: BenchmarkConfig) -> eyre::Result<()> {
    let (_output, _dur) = validate::accelsim::simulate_bench_config(&bench_config).await?;
    Ok(())
}

pub fn run_playground(
    bench_config: &BenchmarkConfig,
) -> eyre::Result<(String, playground::stats::Stats, Duration)> {
    let accelsim_compat_mode = false;
    let extra_args: &[String] = &[];
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

    group.bench_function("vectoradd/10000", |b| {
        b.to_async(&runtime)
            .iter(|| run_accelsim(black_box(get_bench_config("vectorAdd", 2).unwrap())));
    });
    // group.bench_function("transpose/256/naive", |b| {
    //     b.iter(|| run_accelsim(black_box(get_bench_config("transpose", 0).unwrap())))
    // });
}

pub fn play_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("play");
    group.sample_size(10);
    group.sampling_mode(criterion::SamplingMode::Flat);

    group.bench_function("vectoradd/10000", |b| {
        b.iter(|| run_playground(&black_box(get_bench_config("vectorAdd", 2).unwrap())));
    });
    // group.bench_function("transpose/256/naive", |b| {
    //     b.iter(|| run_playground(black_box(get_bench_config("transpose", 0).unwrap())))
    // });
}

pub fn box_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("box");
    group.sample_size(10);
    group.sampling_mode(criterion::SamplingMode::Flat);

    group.bench_function("vectoradd/10000", |b| {
        b.iter(|| run_box(black_box(get_bench_config("vectorAdd", 2).unwrap()), true));
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

    let (bench_name, input_num) = ("transpose", 0); // takes 34 sec (accel same)

    // let (bench_name, input_num) = ("simple_matrixmul", 26); // takes 22 sec

    // let (bench_name, input_num) = ("matrixmul", 3); // takes 54 sec (accel 76)

    // let (bench_name, input_num) = ("vectorAdd", 0);
    println!("running {bench_name}@{input_num}");

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    let start = Instant::now();
    let stats = run_box(black_box(get_bench_config(bench_name, input_num)?), false)?;
    dbg!(&stats.sim);
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

    let start = Instant::now();
    let stats = run_box(black_box(get_bench_config(bench_name, input_num)?), true)?;
    dbg!(&stats.sim);
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

    let (_, _, play_dur) = run_playground(&black_box(get_bench_config(bench_name, input_num)?))?;
    println!("play took:\t\t{play_dur:?}");

    let start = Instant::now();
    runtime.block_on(async {
        run_accelsim(black_box(get_bench_config(bench_name, input_num)?)).await?;
        Ok::<(), eyre::Report>(())
    })?;
    let accel_dur = start.elapsed();
    println!("accel took:\t\t{accel_dur:?}");

    Ok(())
}
