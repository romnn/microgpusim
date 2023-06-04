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
    /// Turn debugging information on
    #[arg(short, long, action = clap::ArgAction::Count)]
    debug: u8,
    // #[command(subcommand)]
    // command: Option<Command>,
}

fn main() -> eyre::Result<()> {
    color_eyre::install()?;

    let start = Instant::now();
    let options = Options::parse();
    std::env::set_var("RUST_BACKTRACE", "full");
    let res = casimu::ported::accelmain(&options.trace_dir);
    println!("completed in {:?}", start.elapsed());
    res
}
