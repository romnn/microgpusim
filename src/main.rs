// #![allow(warnings)]

use clap::{Parser, Subcommand};
use color_eyre::eyre;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Subcommand)]
enum Command {
    /// does testing things
    Trace {
        #[arg(short, long)]
        output: PathBuf,
    },
    /// plots a trace
    PlotTrace {
        #[arg(short, long)]
        output: PathBuf,
    },
}

#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
struct Options {
    /// Input to operate on
    #[arg(value_name = "TRACE_DIR")]
    pub trace_dir: PathBuf,

    /// Stats output file
    #[arg(short = 'o', long = "stats", value_name = "STATS_OUT")]
    pub stats_out_file: Option<PathBuf>,

    /// Turn debugging information on
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub debug: u8,

    /// Use multi-threading
    #[arg(long = "parallel")]
    pub parallel: bool,

    /// Use non-deterministic simulation
    #[arg(long = "non-deterministic")]
    pub non_deterministic: Option<usize>,

    #[clap(flatten)]
    pub accelsim: gpucachesim::config::accelsim::Config,
}

fn main() -> eyre::Result<()> {
    use std::io::Write;

    color_eyre::install()?;

    let start = Instant::now();
    let options = Options::parse();
    #[cfg(debug_assertions)]
    std::env::set_var("RUST_BACKTRACE", "full");

    let mut log_builder = env_logger::Builder::new();
    log_builder.format(|buf, record| {
        writeln!(
            buf,
            // "{} [{}] - {}",
            "{}",
            // Local::now().format("%Y-%m-%dT%H:%M:%S"),
            // record.level(),
            record.args()
        )
    });

    let log_after_cycle = std::env::var("LOG_AFTER")
        .unwrap_or_default()
        .parse::<u64>()
        .ok();

    if log_after_cycle.is_none() {
        log_builder.parse_default_env();
        log_builder.init();
    }

    let deadlock_check = std::env::var("DEADLOCK_CHECK")
        .unwrap_or_default()
        .to_lowercase()
        == "yes";

    let parallelization = match (options.parallel, options.non_deterministic) {
        (false, _) => gpucachesim::config::Parallelization::Serial,
        #[cfg(feature = "parallel")]
        (true, None) => gpucachesim::config::Parallelization::Deterministic,
        #[cfg(feature = "parallel")]
        (true, Some(n)) => gpucachesim::config::Parallelization::Nondeterministic(n),
        #[cfg(not(feature = "parallel"))]
        _ => eyre::bail!(
            "{} was compiled with parallel simulation disabled",
            env!("CARGO_BIN_NAME")
        ),
    };

    let config = gpucachesim::config::GPU {
        num_simt_clusters: 20,                   // 20
        num_cores_per_simt_cluster: 1,           // 1
        num_schedulers_per_core: 2,              // 1
        num_memory_controllers: 8,               // 8
        num_sub_partition_per_memory_channel: 2, // 2
        fill_l2_on_memcopy: false,               // true
        parallelization,
        deadlock_check,
        log_after_cycle,
        ..gpucachesim::config::GPU::default()
    };

    let sim = gpucachesim::accelmain(&options.trace_dir, config)?;
    let stats = sim.stats();

    // save stats to file
    if let Some(stats_out_file) = options.stats_out_file.as_ref() {
        gpucachesim::save_stats_to_file(&stats, stats_out_file)?;
    }

    eprintln!("STATS:\n");
    eprintln!("DRAM: total reads: {}", &stats.dram.total_reads());
    eprintln!("DRAM: total writes: {}", &stats.dram.total_writes());
    eprintln!("SIM: {:#?}", &stats.sim);
    eprintln!("INSTRUCTIONS: {:#?}", &stats.instructions);
    eprintln!("ACCESSES: {:#?}", &stats.accesses);
    eprintln!("L1I: {:#?}", &stats.l1i_stats.reduce());
    eprintln!("L1D: {:#?}", &stats.l1d_stats.reduce());
    eprintln!("L2D: {:#?}", &stats.l2d_stats.reduce());
    eprintln!("completed in {:?}", start.elapsed());
    Ok(())
}
