#![allow(clippy::missing_errors_doc)]
// #![allow(warnings)]

use accelsim::SimConfig;
use async_process::Command;
use color_eyre::eyre::{self, WrapErr};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub fn locate_accelsim_bin(accel_path: &Path, profile: &str) -> eyre::Result<PathBuf> {
    let use_box = std::env::var("USE_BOX").unwrap_or_default().to_lowercase() == "yes";
    let accelsim_bin = if use_box {
        accelsim::build::manifest_path()?
            .join("../target")
            .join(profile)
            .join("playground")
    } else {
        accelsim::executable(accel_path)
    };
    Ok(accelsim_bin)
}

pub fn render_sim_script(
    accelsim_bin: &Path,
    traces_dir: &Path,
    config_dir: &Path,
    profile: &str,
    config: &SimConfig,
    setup_env_path: &Path,
) -> eyre::Result<String> {
    let mut sim_sh = vec![];
    sim_sh.push("#!/usr/bin/env bash".to_string());
    sim_sh.push("set -e".to_string());
    // sim_sh.push(r#"echo "simulating...""#.to_string());
    sim_sh.push(format!("cd {}", config_dir.display()));

    // source simulator setup
    sim_sh.push(format!(
        "source {} {}",
        &*setup_env_path.to_string_lossy(),
        &profile,
    ));

    // run accelsim binary
    let gpgpusim_config = config
        .config()
        .ok_or(eyre::eyre!("missing gpgpusim config"))?;
    let gpgpusim_config = gpgpusim_config.canonicalize().wrap_err_with(|| {
        format!(
            "gpgpusim config at {} does not exist",
            gpgpusim_config.display()
        )
    })?;

    let kernelslist = traces_dir.join("kernelslist.g");
    let kernelslist = kernelslist
        .canonicalize()
        .wrap_err_with(|| format!("kernelslist at {} does not exist", kernelslist.display()))?;

    let mut trace_cmd: Vec<String> = vec![
        accelsim_bin.to_string_lossy().to_string(),
        "-trace".to_string(),
        kernelslist.to_string_lossy().to_string(),
        "-config".to_string(),
        gpgpusim_config.to_string_lossy().to_string(),
    ];

    let trace_config = config.trace_config().as_deref().map(Path::canonicalize);
    match trace_config {
        Some(Ok(config)) if config.is_file() => {
            trace_cmd.extend(["-config".to_string(), config.to_string_lossy().to_string()]);
        }
        _ => {}
    }
    sim_sh.push(trace_cmd.join(" "));
    Ok(sim_sh.join("\n"))
}

pub async fn simulate_trace(
    traces_dir: impl AsRef<Path>,
    config: SimConfig,
    timeout: Option<Duration>,
) -> eyre::Result<std::process::Output> {
    #[cfg(feature = "upstream")]
    let use_upstream = true;
    #[cfg(not(feature = "upstream"))]
    let use_upstream = false;

    let accelsim_path = accelsim::locate(use_upstream)?;
    let profile = accelsim::profile();

    let accelsim_bin = locate_accelsim_bin(&accelsim_path, profile)?;
    let accelsim_bin = accelsim_bin
        .canonicalize()
        .wrap_err_with(|| format!("{} does not exist", accelsim_bin.display()))?;

    let sim_root = accelsim_path.join("gpu-simulator/");
    let setup_env_path = sim_root.join("setup_environment.sh");
    let setup_env_path = setup_env_path
        .canonicalize()
        .wrap_err_with(|| format!("{} does not exist", setup_env_path.display()))?;

    // change current working dir
    let config_dir = config
        .config_dir
        .as_ref()
        .ok_or(eyre::eyre!("missing config dir"))?;
    let config_dir = config_dir
        .canonicalize()
        .wrap_err_with(|| format!("{} does not exist", config_dir.display()))?;

    if !config_dir.is_dir() {
        eyre::bail!("config dir {} is not a directory", config_dir.display());
    }

    let tmp_sim_sh = render_sim_script(
        &accelsim_bin,
        traces_dir.as_ref(),
        &config_dir,
        profile,
        &config,
        &setup_env_path,
    )?;
    log::debug!("{}", &tmp_sim_sh);

    let tmp_sim_sh_path = traces_dir.as_ref().join("sim.tmp.sh");
    {
        let mut tmp_sim_sh_file = utils::fs::open_writable(&tmp_sim_sh_path)?;
        tmp_sim_sh_file.write_all(tmp_sim_sh.as_bytes())?;
    }
    let tmp_sim_sh_path = tmp_sim_sh_path
        .canonicalize()
        .wrap_err_with(|| format!("{} does not exist", tmp_sim_sh_path.display()))?;

    let cuda_path = utils::find_cuda().ok_or(eyre::eyre!("CUDA not found"))?;

    let mut cmd = Command::new("bash");
    cmd.current_dir(config_dir);

    let args = [tmp_sim_sh_path.to_string_lossy().to_string()];
    cmd.args(&args);
    cmd.env("CUDA_INSTALL_PATH", &*cuda_path.to_string_lossy());
    log::debug!("command: {:?}", &cmd);

    let result = match timeout {
        Some(timeout) => tokio::time::timeout(timeout, cmd.output()).await,
        None => Ok(cmd.output().await),
    };
    let result = result??;

    if !result.status.success() {
        use color_eyre::Section;
        return Err(utils::CommandError::new(&cmd, result)
            .into_eyre()
            .with_suggestion(|| {
                format!(
                    "to debug, use: `gdb --args bash {}`",
                    tmp_sim_sh_path.display(),
                )
            }));
    }

    // for now, we want to keep the file
    // std::fs::remove_file(&tmp_sim_sh_path);
    Ok(result)
}
