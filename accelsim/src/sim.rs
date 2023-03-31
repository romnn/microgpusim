#![allow(warnings)]

use accelsim::parser::{parse, Options as ParseOptions};
use anyhow::Result;
use async_process::Command;
use clap::Parser;
use std::collections::HashMap;
use std::io::Write;
use std::os::unix::fs::DirBuilderExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

async fn sim_trace(
    traces_dir: impl AsRef<Path>,
    config: SimConfig,
    timeout: Option<Duration>,
) -> Result<std::process::Output> {
    let accelsim_path = accelsim::locate()?;
    let sim_root = accelsim_path.join("gpu-simulator/");
    let accelsim_bin = sim_root.join("bin/release/accel-sim.out");
    if !accelsim_bin.is_file() {
        anyhow::bail!("missing {}", accelsim_bin.display());
    }

    let setup_env = sim_root.join("setup_environment.sh");
    if !setup_env.is_file() {
        anyhow::bail!("missing {}", setup_env.display());
    }

    // utils.chmod_x(setup_env)
    let mut tmp_sim_sh = vec!["set -e".to_string()];
    tmp_sim_sh.push(format!("source {}", setup_env.canonicalize()?.display()));

    let gpgpusim_config = config
        .config
        .unwrap_or(config.config_dir.join("gpgpusim.config"));
    let mut trace_cmd: Vec<String> = vec![
        accelsim_bin.canonicalize()?.to_string_lossy().to_string(),
        "-trace".to_string(),
        traces_dir
            .as_ref()
            .join("kernelslist.g")
            .canonicalize()?
            .to_string_lossy()
            .to_string(),
        "-config".to_string(),
        gpgpusim_config
            .canonicalize()?
            .to_string_lossy()
            .to_string(),
    ];

    // if let Some(inter_config) = gpgpusim_trace_config {
    let gpgpusim_trace_config = config
        .trace_config
        .unwrap_or(config.config_dir.join("gpgpusim.trace.config"));

    if gpgpusim_trace_config.is_file() {
        trace_cmd.extend([
            "-config".to_string(),
            gpgpusim_trace_config
                .canonicalize()?
                .to_string_lossy()
                .to_string(),
        ]);
    }
    tmp_sim_sh.push(trace_cmd.join(" "));
    let tmp_sim_sh = tmp_sim_sh.join("\n");
    dbg!(&tmp_sim_sh);

    let tmp_sim_sh_path = traces_dir.as_ref().join("sim.tmp.sh");
    let mut tmp_sim_sh_file = std::fs::OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(&tmp_sim_sh_path)?;
    tmp_sim_sh_file.write_all(tmp_sim_sh.as_bytes())?;

    if !config.config_dir.is_dir() {
        anyhow::bail!(
            "config dir {} is not a directory",
            config.config_dir.display()
        );
    }

    let mut cmd = Command::new("bash");
    cmd.current_dir(config.config_dir);
    cmd.arg(&*tmp_sim_sh_path.canonicalize()?.to_string_lossy());
    if let Some(cuda_path) = utils::find_cuda().first() {
        cmd.env("CUDA_INSTALL_PATH", &*cuda_path.to_string_lossy());
    }
    dbg!(&cmd);

    let result = match timeout {
        Some(timeout) => tokio::time::timeout(timeout.into(), cmd.output()).await,
        None => Ok(cmd.output().await),
    };
    let result = result??;

    std::fs::remove_file(&tmp_sim_sh_path);
    if !result.status.success() {
        anyhow::bail!("cmd failed with code {:?}", result.status.code());
    }
    Ok(result)
}

fn parse_duration_string(duration: &str) -> Result<Duration> {
    let res = duration_string::DurationString::from_string(duration.into())
        .map_err(|msg| anyhow::anyhow!("invalid duration string {}", duration))?;
    Ok(res.into())
}

#[derive(Parser, Debug)]
struct Options {
    traces_dir: PathBuf,

    #[clap(flatten)]
    sim_config: SimConfig,

    log_file: Option<PathBuf>,
    stats_file: Option<PathBuf>,

    #[clap(
        help = "timeout",
        value_parser = parse_duration_string,
    )]
    timeout: Option<Duration>,
}

#[derive(Parser, Debug)]
struct SimConfig {
    config_dir: PathBuf,
    config: Option<PathBuf>,
    trace_config: Option<PathBuf>,
    inter_config: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let options = Options::parse();
    dbg!(&options.traces_dir);

    let start = Instant::now();
    let output = sim_trace(&options.traces_dir, options.sim_config, options.timeout).await?;
    println!("simulating took {:?}", start.elapsed());

    // write log
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    // dbg!(&stdout);
    // dbg!(&stderr);

    let log_file_path = options
        .log_file
        .unwrap_or(options.traces_dir.join("accelsim_log.txt"));
    {
        let mut log_file = std::fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(&log_file_path)?;
        log_file.write_all(stdout.as_bytes())?;
    }

    // parse stats
    let stats_file_path = options
        .stats_file
        .unwrap_or(log_file_path.with_extension("csv"));
    let stats = parse(ParseOptions::new(log_file_path, stats_file_path))?;

    let mut preview: Vec<_> = stats
        .iter()
        .map(|(idx, val)| (format!("{} / {} / {}", idx.0, idx.1, idx.2), val))
        .collect();
    preview.sort_by(|a, b| a.0.cmp(&b.0));

    for (key, val) in preview {
        println!(" => {key}: {val}");
    }
    Ok(())
}
