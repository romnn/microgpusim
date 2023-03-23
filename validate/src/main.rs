#[cfg(feature = "remote")]
pub mod remote;

use anyhow::Result;
use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

fn open_writable(path: &Path) -> Result<BufWriter<fs::File>, std::io::Error> {
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)?;
    Ok(BufWriter::new(file))
}

async fn profile_exec(exec: &Path, exec_args: &Vec<&String>, traces_dir: &Path) -> Result<()> {
    let profiling_results = profile::nvprof(exec, exec_args).await?;
    let writer = open_writable(&traces_dir.join("nvprof.json"))?;
    serde_json::to_writer_pretty(writer, &profiling_results.metrics)?;
    let mut writer = open_writable(&traces_dir.join("nvprof.log"))?;
    writer.write_all(profiling_results.raw.as_bytes())?;
    Ok(())
}

async fn trace_exec(exec: &Path, exec_args: &Vec<&String>, traces_dir: &Path) -> Result<()> {
    invoke_trace::trace(exec, exec_args, traces_dir).await?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    use std::os::unix::fs::DirBuilderExt;

    // load env variables from .env files
    dotenv::dotenv().ok();

    let args: Vec<_> = std::env::args().collect();
    let exec = PathBuf::from(args.get(1).expect("usage ./casimu <executable> [args]"));
    let exec_args = args.iter().skip(2).collect::<Vec<_>>();

    let exec_dir = exec.parent().expect("executable has no parent dir");
    let traces_dir = exec_dir.join("traces");

    #[cfg(feature = "remote")]
    remote::connect().await?;

    fs::DirBuilder::new()
        .recursive(true)
        .mode(0o777)
        .create(&traces_dir)?;

    profile_exec(&exec, &exec_args, &traces_dir).await?;
    trace_exec(&exec, &exec_args, &traces_dir).await?;

    Ok(())
}
