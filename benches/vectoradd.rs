#![allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]

use color_eyre::eyre;
use criterion::{black_box, Criterion};
use validate::materialize::{BenchmarkConfig, Benchmarks};

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

pub fn run_box(mut bench_config: BenchmarkConfig) -> eyre::Result<()> {
    bench_config.simulate.parallel =
        std::env::var("PARALLEL").unwrap_or_default().to_lowercase() == "yes";
    println!("parallel: {}", bench_config.simulate.parallel);
    let stats = validate::simulate::simulate_bench_config(&bench_config)?;
    let cycles = stats.sim.cycles;
    // fast parallel:   cycle loop time: 558485 ns
    // serial:          cycle loop time: 2814591 ns (speedup 5x)
    // have 80 cores and 16 threads
    //
    // parallel: dram cycle time: 229004 ns
    let total = casimu::TOTAL_CYCLE_DUR.lock().unwrap().clone();
    for (name, dur) in [
        ("core cycle", &casimu::CORE_CYCLE_DUR),
        ("icnt cycle", &casimu::ICNT_CYCLE_DUR),
        ("dram cycle", &casimu::DRAM_CYCLE_DUR),
        ("l2 cycle", &casimu::L2_CYCLE_DUR),
        ("total cycle", &casimu::TOTAL_CYCLE_DUR),
    ] {
        let dur = dur.lock().unwrap();
        let percent = (dur.as_secs_f64() / total.as_secs_f64()) * 100.0;
        let ns = dur.as_nanos() / u128::from(cycles);
        println!("{name} time: {ns} ns ({percent:>2.2}%)");
    }
    Ok(())
}

pub async fn run_accelsim(bench_config: BenchmarkConfig) -> eyre::Result<()> {
    let (_stats, _dur) = validate::accelsim::simulate_bench_config(&bench_config).await?;
    Ok(())
}

pub fn run_playground(bench_config: &BenchmarkConfig) -> eyre::Result<()> {
    let _stats = validate::playground::simulate_bench_config(bench_config);
    Ok(())
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
        b.iter(|| run_box(black_box(get_bench_config("vectorAdd", 2).unwrap())));
    });
    // group.bench_function("transpose/256/naive", |b| {
    //     b.iter(|| run_box(black_box(get_bench_config("transpose", 0).unwrap())))
    // });
}

criterion::criterion_group!(benches, box_benchmark, play_benchmark, accelsim_benchmark);
// criterion::criterion_main!(benches);

#[allow(dead_code)]
fn main() -> eyre::Result<()> {
    use std::time::Instant;

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    let mut start = Instant::now();
    let _ = run_box(black_box(get_bench_config("transpose", 0)?));
    let box_dur = start.elapsed();
    println!("box took:\t\t{:?}", box_dur);

    start = Instant::now();
    let _ = run_playground(&black_box(get_bench_config("transpose", 0)?));
    let play_dur = start.elapsed();
    println!("play took:\t\t{:?}", play_dur);

    start = Instant::now();
    runtime.block_on(async {
        run_accelsim(black_box(get_bench_config("transpose", 0)?)).await?;
        Ok::<(), eyre::Report>(())
    })?;
    let accel_dur = start.elapsed();
    println!("accel took:\t\t{:?}", accel_dur);

    println!(
        "speedup is :\t\t{:.2}",
        play_dur.as_secs_f64() / box_dur.as_secs_f64()
    );

    Ok(())
}
