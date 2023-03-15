use invoke_trace;
use profile;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

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
    // Simulate {
    //     /// lists test values
    //     #[arg(short, long)]
    //     list: bool,
    // },
    // Accel {
    //     /// lists test values
    //     #[arg(short, long)]
    //     list: bool,
    // },
}

#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
struct Options {
    /// Input to operate on
    #[arg(short, long, value_name = "FILE")]
    path: PathBuf,
    /// Turn debugging information on
    #[arg(short, long, action = clap::ArgAction::Count)]
    debug: u8,

    #[command(subcommand)]
    command: Option<Command>,
}

fn open_writable(path: &Path) -> Result<BufWriter<fs::File>, std::io::Error> {
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&path)?;
    Ok(BufWriter::new(file))
}

fn profile_exec(exec: &Path, exec_args: &Vec<&String>, traces_dir: &Path) -> Result<()> {
    let profiling_results = profile::nvprof(exec, exec_args)?;
    let writer = open_writable(&traces_dir.join("nvprof.json"))?;
    serde_json::to_writer_pretty(writer, &profiling_results.metrics)?;
    let mut writer = open_writable(&traces_dir.join("nvprof.log"))?;
    writer.write_all(profiling_results.raw.as_bytes())?;
    Ok(())
}

fn trace_exec(exec: &Path, exec_args: &Vec<&String>, traces_dir: &Path) -> Result<()> {
    invoke_trace::trace(exec, exec_args, traces_dir)?;
    Ok(())
}

fn main() -> Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    // let options = Options::parse();
    // dbg!(&options);

    let args: Vec<_> = std::env::args().collect();
    let exec = PathBuf::from(args.get(1).expect("usage ./profile <executable> [args]"));
    let exec_args = args.iter().skip(2).collect::<Vec<_>>();

    let exec_dir = exec.parent().expect("executable has no parent dir");
    let traces_dir = exec_dir.join("traces");
    // fs::create_dir_all(&traces_dir).ok();
    fs::DirBuilder::new()
        .recursive(true)
        .mode(0o777)
        .create(&traces_dir)?;

    profile_exec(&exec, &exec_args, &traces_dir)?;
    trace_exec(&exec, &exec_args, &traces_dir)?;

    Ok(())
}
