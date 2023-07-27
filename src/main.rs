#![allow(warnings)]

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
    #[arg(short = 'p', long = "path", value_name = "TRACE_DIR")]
    trace_dir: PathBuf,
    /// Stats output file
    #[arg(short = 'o', long = "stats", value_name = "STATS_OUT")]
    stats_out_file: Option<PathBuf>,
    /// Turn debugging information on
    #[arg(short, long, action = clap::ArgAction::Count)]
    debug: u8,
    // #[command(subcommand)]
    // command: Option<Command>,
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

    let stats = casimu::ported::accelmain(&options.trace_dir, log_after_cycle)?;

    // save stats to file
    if let Some(stats_out_file) = options.stats_out_file.as_ref() {
        casimu::ported::save_stats_to_file(&stats, &stats_out_file)?;
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
