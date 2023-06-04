#![allow(warnings)]

use accelsim::parser::{parse, Options as ParseOptions};
use async_process::Command;
use clap::Parser;
use color_eyre::eyre;
use std::collections::HashMap;
use std::io::Write;
use std::os::unix::fs::DirBuilderExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

async fn sim_trace(
    traces_dir: impl AsRef<Path>,
    config: SimConfig,
    timeout: Option<Duration>,
) -> eyre::Result<std::process::Output> {
    let accelsim_path = accelsim::locate()?;
    let sim_root = accelsim_path.join("gpu-simulator/");

    #[cfg(debug_assertions)]
    let profile = "debug";
    #[cfg(not(debug_assertions))]
    let profile = "release";

    let accelsim_bin = sim_root.join("bin").join(profile).join("accel-sim.out");
    if !accelsim_bin.is_file() {
        eyre::eyre!("missing {}", accelsim_bin.display());
    }

    let setup_env = sim_root.join("setup_environment.sh");
    if !setup_env.is_file() {
        eyre::eyre!("missing {}", setup_env.display());
    }

    let mut tmp_sim_sh = vec!["set -e".to_string()];

    // change current working dir
    tmp_sim_sh.push(format!(
        "cd {}",
        config.config_dir.canonicalize()?.display()
    ));

    // source simulator setup
    tmp_sim_sh.push(format!(
        "source {} {}",
        &*setup_env.canonicalize()?.to_string_lossy(),
        &profile,
    ));

    // run accelsim binary
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
    // dbg!(&tmp_sim_sh);

    let tmp_sim_sh_path = traces_dir.as_ref().join("sim.tmp.sh");
    let mut tmp_sim_sh_file = std::fs::OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(&tmp_sim_sh_path)?;
    tmp_sim_sh_file.write_all(tmp_sim_sh.as_bytes())?;

    if !config.config_dir.is_dir() {
        eyre::eyre!(
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
    // dbg!(&cmd);

    let result = match timeout {
        Some(timeout) => tokio::time::timeout(timeout, cmd.output()).await,
        None => Ok(cmd.output().await),
    };
    let result = result??;

    if !result.status.success() {
        println!("{}", String::from_utf8_lossy(&result.stdout).to_string());
        println!("{}", String::from_utf8_lossy(&result.stderr).to_string());

        // if we want to debug, we leave the file in place
        // gdb --args bash test-apps/vectoradd/traces/vectoradd-100-32-trace/sim.tmp.sh
        eyre::eyre!("cmd failed with code {:?}", result.status.code());
    }

    // for now, we want to keep the file
    // std::fs::remove_file(&tmp_sim_sh_path);
    Ok(result)
}

fn parse_duration_string(duration: &str) -> eyre::Result<Duration> {
    let res = duration_string::DurationString::from_string(duration.into())
        .map_err(|msg| eyre::eyre!("invalid duration string {}", duration))?;
    Ok(res.into())
}

#[derive(Parser, Debug)]
struct Options {
    #[clap(
        // long = "traces-dir",
        help = "directory containing accelsim traces (kernelslist.g)"
    )]
    traces_dir: PathBuf,

    #[clap(flatten)]
    sim_config: SimConfig,

    #[clap(long = "log-file", help = "write simuation output to log file")]
    log_file: Option<PathBuf>,

    #[clap(long = "stats-file", help = "parse simulation stats into csv file")]
    stats_file: Option<PathBuf>,

    #[clap(
        long = "timeout",
        help = "timeout",
        value_parser = parse_duration_string,
    )]
    timeout: Option<Duration>,
}

#[derive(Parser, Debug)]
struct SimConfig {
    // #[clap(long = "config-dir", help = "config directory")]
    #[clap(help = "config directory")]
    config_dir: PathBuf,
    #[clap(long = "config", help = "config file")]
    config: Option<PathBuf>,
    #[clap(long = "trace-config", help = "trace config file")]
    trace_config: Option<PathBuf>,
    #[clap(long = "inter-config", help = "interconnect config file")]
    inter_config: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    color_eyre::install()?;

    let mut options = Options::parse();

    // make paths absolute
    // options.traces_dir = options.traces_dir.canonicalize()?;
    // options.sim_config.config_dir = options.sim_config.config_dir.canonicalize()?;
    //
    // if let Some(log_file) = options.log_file {
    //     options.log_file = Some(log_file.canonicalize()?);
    // }
    // if let Some(stats_file) = options.stats_file {
    //     options.stats_file = Some(stats_file.canonicalize()?);
    // }
    // if let Some(config) = options.sim_config.config {
    //     options.sim_config.config = Some(config.canonicalize()?);
    // }
    // if let Some(trace_config) = options.sim_config.trace_config {
    //     options.sim_config.trace_config = Some(trace_config.canonicalize()?);
    // }
    // if let Some(inter_config) = options.sim_config.inter_config {
    //     options.sim_config.inter_config = Some(inter_config.canonicalize()?);
    // }

    // dbg!(&options.traces_dir);

    let start = Instant::now();
    let output = sim_trace(&options.traces_dir, options.sim_config, options.timeout).await?;
    println!("simulating took {:?}", start.elapsed());

    // write log
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    println!("\n\n STDOUT \n\n");
    println!("{}", &stdout);

    eprintln!("\n\n STDERR \n\n");
    eprintln!("{}", &stderr);

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
    let parse_options = ParseOptions::new(log_file_path, stats_file_path);
    let stats = parse(parse_options)?;

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
