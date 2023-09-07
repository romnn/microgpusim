#![allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
// #![allow(warnings)]

use accelsim::SimConfig;
use async_process::Command;
use color_eyre::eyre::{self, WrapErr};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[must_use]
pub fn has_upstream() -> bool {
    #[cfg(feature = "upstream")]
    let upstream = true;
    #[cfg(not(feature = "upstream"))]
    let upstream = false;
    upstream
}

pub fn locate_accelsim_bin(accel_path: &Path, profile: &str) -> eyre::Result<PathBuf> {
    let use_box = std::env::var("USE_BOX").unwrap_or_default().to_lowercase() == "yes";
    let accelsim_bin = if use_box {
        accelsim::build::manifest_path()?
            .join("../target")
            .join(profile)
            .join("playground")
    } else {
        accelsim::executable(accel_path, profile)
    };
    Ok(accelsim_bin)
}

pub struct SimulationScript<'a> {
    accelsim_bin: PathBuf,
    kernelslist: PathBuf,
    cwd: &'a Path,
    profile: &'a str,
    config: &'a SimConfig,
    setup_env_path: PathBuf,
    extra_sim_args: Vec<String>,
}

impl<'a> SimulationScript<'a> {
    pub fn render(self) -> eyre::Result<String> {
        let Self {
            accelsim_bin,
            kernelslist,
            cwd,
            profile,
            config,
            setup_env_path,
            extra_sim_args,
        } = self;

        let mut sim_sh = vec![];
        sim_sh.push("#!/usr/bin/env bash".to_string());
        sim_sh.push("set -e".to_string());
        // sim_sh.push(r#"echo "simulating...""#.to_string());
        sim_sh.push(format!("cd {}", cwd.display()));

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

        let mut sim_cmd: Vec<String> = vec![
            accelsim_bin.to_string_lossy().to_string(),
            "-trace".to_string(),
            kernelslist.to_string_lossy().to_string(),
            "-config".to_string(),
            gpgpusim_config.to_string_lossy().to_string(),
        ];

        let trace_config = config.trace_config().as_deref().map(Path::canonicalize);
        match trace_config {
            Some(Ok(config)) if config.is_file() => {
                sim_cmd.extend(["-config".to_string(), config.to_string_lossy().to_string()]);
            }
            _ => {}
        }

        // extra simulatin arguments have highest precedence
        sim_cmd.extend(extra_sim_args);

        sim_sh.push(sim_cmd.join(" "));
        Ok(sim_sh.join("\n"))
    }

    pub fn new<A>(
        kernelslist: impl AsRef<Path>,
        cwd: &'a Path,
        config: &'a SimConfig,
        extra_sim_args: A,
        use_upstream: bool,
    ) -> eyre::Result<Self>
    where
        A: IntoIterator,
        <A as IntoIterator>::Item: Into<String>,
    {
        #[cfg(not(feature = "upstream"))]
        if use_upstream {
            eyre::bail!("accelsim-sim was not compiled with upstream accelsim");
        }
        log::debug!("upstream = {}", use_upstream);

        let accelsim_path = accelsim::locate(use_upstream)?;

        let profile = "release";
        let accelsim_bin = locate_accelsim_bin(&accelsim_path, profile)?;
        let accelsim_bin = accelsim_bin
            .canonicalize()
            .wrap_err_with(|| format!("{} does not exist", accelsim_bin.display()))?;

        log::debug!("using accelsim binary at {}", accelsim_bin.display());

        let sim_root = accelsim_path.join("gpu-simulator/");
        let setup_env_path = sim_root.join("setup_environment.sh");
        let setup_env_path = setup_env_path
            .canonicalize()
            .wrap_err_with(|| format!("{} does not exist", setup_env_path.display()))?;

        let kernelslist = kernelslist.as_ref();
        let kernelslist = kernelslist
            .canonicalize()
            .wrap_err_with(|| format!("{} does not exist", kernelslist.display()))?;

        let extra_sim_args: Vec<String> = extra_sim_args.into_iter().map(Into::into).collect();
        Ok(Self {
            accelsim_bin,
            kernelslist,
            cwd,
            profile,
            config,
            setup_env_path,
            extra_sim_args,
        })
    }
}

pub async fn simulate_trace<A>(
    traces_dir: impl AsRef<Path>,
    kernelslist: impl AsRef<Path>,
    config: &SimConfig,
    timeout: Option<Duration>,
    extra_sim_args: A,
    stream_output: bool,
    use_upstream: bool,
) -> eyre::Result<(std::process::Output, Duration)>
where
    A: IntoIterator,
    <A as IntoIterator>::Item: Into<String>,
{
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

    let tmp_sim_sh = SimulationScript::new(
        &kernelslist,
        &config_dir,
        config,
        extra_sim_args,
        use_upstream,
    )?;
    let tmp_sim_sh = tmp_sim_sh.render()?;
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

    let get_cmd_output = async {
        if stream_output {
            use futures::{AsyncBufReadExt, StreamExt};
            let mut stdout: Vec<u8> = Vec::new();
            let mut child = cmd.stdout(async_process::Stdio::piped()).spawn()?;

            let mut line_reader = futures::io::BufReader::new(child.stdout.take().unwrap()).lines();
            while let Some(line) = line_reader.next().await {
                let line = line?;
                println!("{line}");
                stdout.extend(line.into_bytes());
                stdout.write_all(b"\n")?;
            }
            Ok(std::process::Output {
                status: child.status().await?,
                stdout,
                stderr: Vec::new(),
            })
        } else {
            cmd.output().await
        }
    };

    let start = Instant::now();
    let result = match timeout {
        Some(timeout) => tokio::time::timeout(timeout, get_cmd_output).await,
        None => Ok(get_cmd_output.await),
    };
    let result = result??;
    let dur = start.elapsed();

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
    Ok((result, dur))
}
