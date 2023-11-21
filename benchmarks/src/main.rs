use clap::{Parser, Subcommand};
use color_eyre::eyre;
use gpucachesim_benchmarks as benchmarks;
use std::time::Instant;

#[derive(Debug, Subcommand)]
enum Command {
    Vectoradd {
        #[arg(long = "dtype", default_value = "32")]
        dtype: usize,
        #[arg(long = "length")]
        length: usize,
    },
    Transpose {
        #[arg(long = "dim")]
        dim: usize,
        #[arg(long = "variant")]
        variant: benchmarks::transpose::Variant,
        #[arg(long = "repetitions")]
        repetitions: Option<usize>,
    },
    Matrixmul {
        #[arg(long = "dtype", default_value = "32")]
        dtype: usize,
        #[arg(long = "rows")]
        rows: usize,
    },
    Babelstream {},
    SimpleMatrixmul {
        #[arg(long = "dtype", default_value = "32")]
        dtype: usize,
        #[arg(short = 'm', long = "m")]
        m: usize,
        #[arg(short = 'n', long = "n")]
        n: usize,
        #[arg(short = 'p', long = "p")]
        p: usize,
    },
}

#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
struct Options {
    #[clap(subcommand)]
    pub command: Command,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> eyre::Result<()> {
    color_eyre::install()?;
    gpucachesim::init_deadlock_detector();

    let options = Options::parse();

    let deadlock_check = std::env::var("DEADLOCK_CHECK")
        .unwrap_or_default()
        .to_lowercase()
        == "yes";

    let start = Instant::now();

    let (commands, kernel_traces) = match options.command {
        Command::Vectoradd { length, dtype } => match dtype {
            32 => benchmarks::vectoradd::benchmark::<f32>(length).await,
            64 => benchmarks::vectoradd::benchmark::<f64>(length).await,
            other => return Err(eyre::eyre!("invalid dtype {other:?}")),
        },
        Command::SimpleMatrixmul { dtype, m, n, p } => match dtype {
            32 => benchmarks::simple_matrixmul::benchmark::<f32>(m, n, p).await,
            64 => benchmarks::simple_matrixmul::benchmark::<f64>(m, n, p).await,
            other => return Err(eyre::eyre!("invalid dtype {other:?}")),
        },
        Command::Matrixmul { dtype, rows } => match dtype {
            32 => benchmarks::matrixmul::benchmark::<f32>(rows).await,
            64 => benchmarks::matrixmul::benchmark::<f64>(rows).await,
            other => return Err(eyre::eyre!("invalid dtype {other:?}")),
        },
        Command::Transpose {
            dim,
            variant,
            repetitions,
        } => benchmarks::transpose::benchmark::<f32>(dim, variant, repetitions.unwrap_or(0)).await,
        Command::Babelstream { .. } => unimplemented!("babelstream not yet supported"),
    }?;

    // let traces_dir = stats_dir.join("traces");
    // gpucachesim::exec::write_traces(commands, kernel_traces, &traces_dir)?;

    // let bench = BenchmarkConfig {
    //     target: Target::Simulate,
    //     target_config: TargetBenchmarkConfig::Simulate {
    //         traces_dir: traces_dir.clone(),
    //         stats_dir: stats_dir.clone(),
    //         accelsim_traces_dir: traces_dir.clone(),
    //         parallel,
    //     },
    //     ..bench.clone()
    // };
    // let (sim, dur) = tokio::task::spawn_blocking(move || {
    //     let start = std::time::Instant::now();
    //     let stats = super::simulate_bench_config(&bench)?;
    //     Ok::<_, eyre::Report>((stats, start.elapsed()))
    // })
    // .await
    // .map_err(eyre::Report::from)??;
    //
    // total_dur += dur;
    // let stats = sim.stats();
    // let full = false;
    // validate::process_stats(stats.as_ref(), &dur, &stats_dir, repetition, full)?;

    // eprintln!("STATS:\n");
    // for (kernel_launch_id, kernel_stats) in stats.as_ref().iter().enumerate() {
    //     eprintln!(
    //         "\n ===== kernel launch {kernel_launch_id:<3}: {}  =====\n",
    //         kernel_stats.sim.kernel_name
    //     );
    //     eprintln!("DRAM: {:#?}", &kernel_stats.dram.reduce());
    //     eprintln!("SIM: {:#?}", &kernel_stats.sim);
    //     eprintln!("INSTRUCTIONS: {:#?}", &kernel_stats.instructions);
    //     eprintln!("ACCESSES: {:#?}", &kernel_stats.accesses);
    //     eprintln!("L1I: {:#?}", &kernel_stats.l1i_stats.reduce());
    //     eprintln!("L1D: {:#?}", &kernel_stats.l1d_stats.reduce());
    //     eprintln!("L2D: {:#?}", &kernel_stats.l2d_stats.reduce());
    // }
    // eprintln!("completed in {:?}", start.elapsed());
    Ok(())
}
