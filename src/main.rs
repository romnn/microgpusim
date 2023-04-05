#![allow(warnings)]

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

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

fn main() -> Result<()> {
    let options = Options::parse();
    casimu::ported::accelmain(&options.trace_dir)
}
