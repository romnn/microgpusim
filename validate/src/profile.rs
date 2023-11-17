use super::materialized::{BenchmarkConfig, TargetBenchmarkConfig};
use crate::{
    open_writable,
    options::{self, Options},
    RunError,
};
use color_eyre::{eyre, Help, SectionExt};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use utils::fs::create_dirs;

pub const DAS6_FORWARD_PORT: u16 = 2201;
pub const DAS5_FORWARD_PORT: u16 = 2202;

#[derive(Debug, Clone, Copy, strum::Display, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[strum(serialize_all = "lowercase")]
pub enum Profiler {
    Nvprof,
    Nsight,
}

fn job_name(
    profiler: Profiler,
    gpu: &str,
    executable: impl AsRef<Path> + Send,
    args: &[String],
) -> String {
    [
        profiler.to_string(),
        gpu.replace(" ", "-"),
        executable
            .as_ref()
            .file_stem()
            .and_then(std::ffi::OsStr::to_str)
            .unwrap_or("")
            .to_string(),
    ]
    .iter()
    .chain(args)
    .map(String::as_str)
    .collect::<Vec<&str>>()
    .join("-")
}

async fn prepare_profiling<R>(
    remote: &R,
    profiler: Profiler,
    gpu: &str,
    executable: impl AsRef<Path> + Send,
    args: &[String],
) -> eyre::Result<(String, PathBuf)>
where
    R: ProfileDAS,
{
    let job_name = job_name(profiler, gpu, executable.as_ref(), &*args);

    let remote_profile_dir = remote
        .remote_scratch_dir()
        .join(format!("profile-{}", profiler))
        .join(&job_name);

    // empty results dir
    let delete_dir_cmd = format!("rm -rf {}", remote_profile_dir.display());
    let _ = remote.run_command(delete_dir_cmd).await?;

    // create results dir
    let create_dir_cmd = format!("mkdir -p {}", remote_profile_dir.display());
    let (exit_status, stdout, stderr) = remote.run_command(create_dir_cmd).await?;
    if !stdout.is_empty() {
        log::debug!("{}", stdout);
    }
    if !stderr.is_empty() {
        log::error!("{}", stderr);
    }
    assert_eq!(exit_status, 0);
    Ok((job_name, remote_profile_dir))
}

#[async_trait::async_trait]
trait ProfileDAS
where
    Self: remote::slurm::Client + remote::scp::Client + remote::Remote,
{
    fn remote_scratch_dir(&self) -> PathBuf {
        PathBuf::from("/var/scratch").join(self.username())
    }

    async fn read_remote_file(
        &self,
        remote_path: &Path,
        allow_empty: bool,
    ) -> eyre::Result<String> {
        use tokio::io::AsyncReadExt;
        // wait for file to become available
        self.wait_for_file(
            remote_path,
            std::time::Duration::from_secs(2),
            allow_empty,
            Some(20),
        )
        .await?;
        let (mut stream, stat) = self.download_file(remote_path).await?;
        assert!(allow_empty || stat.size() > 0);
        let mut content = String::new();
        stream.read_to_string(&mut content).await?;
        Ok(content)
    }

    async fn profile_nvprof<A>(
        &self,
        gpu: &str,
        executable: impl AsRef<Path> + Send + Sync,
        args: A,
        timeout: Option<std::time::Duration>,
    ) -> eyre::Result<profile::nvprof::Output>
    where
        A: Clone + IntoIterator + Send,
        <A as IntoIterator>::Item: AsRef<std::ffi::OsStr>;

    async fn profile_nsight<A>(
        &self,
        gpu: &str,
        executable: impl AsRef<Path> + Send + Sync,
        args: A,
        timeout: Option<std::time::Duration>,
    ) -> eyre::Result<profile::nsight::Output>
    where
        A: Clone + IntoIterator + Send,
        <A as IntoIterator>::Item: AsRef<std::ffi::OsStr>;
}

#[async_trait::async_trait]
impl<T> ProfileDAS for T
where
    T: remote::slurm::Client + remote::Remote + remote::scp::Client + Sync,
{
    async fn profile_nsight<A>(
        &self,
        gpu: &str,
        executable: impl AsRef<Path> + Send + Sync,
        args: A,
        timeout: Option<std::time::Duration>,
    ) -> eyre::Result<profile::nsight::Output>
    where
        A: Clone + IntoIterator + Send,
        <A as IntoIterator>::Item: AsRef<std::ffi::OsStr>,
    {
        use std::fmt::Write;
        let args: Vec<String> = args
            .into_iter()
            .map(|arg| arg.as_ref().to_string_lossy().to_string())
            .collect();

        let (job_name, remote_profile_dir) =
            prepare_profiling(self, Profiler::Nsight, gpu, executable.as_ref(), &*args).await?;
        let remote_job_path = remote_profile_dir.join("job.slurm");
        let remote_stdout_path = remote_profile_dir.join("stdout.log");
        let remote_stderr_path = remote_profile_dir.join("stderr.log");

        let nsight_args = profile::nsight::build_nsight_args(executable.as_ref(), &*args)?;

        // build slurm script
        let mut slurm_script = String::new();
        writeln!(slurm_script, "#!/bin/sh")?;
        writeln!(slurm_script, "#SBATCH --job-name={}", job_name)?;
        writeln!(
            slurm_script,
            "#SBATCH --output={}",
            remote_stdout_path.display()
        )?;
        writeln!(
            slurm_script,
            "#SBATCH --error={}",
            remote_stderr_path.display()
        )?;

        if let Some(timeout) = timeout {
            writeln!(
                slurm_script,
                "#SBATCH --time={}",
                remote::slurm::duration_to_slurm(&timeout)
            )?;
        }
        writeln!(slurm_script, "#SBATCH -N 1")?;
        writeln!(slurm_script, "#SBATCH -C {}", gpu)?;
        writeln!(slurm_script, "#SBATCH --gres=gpu:1")?;
        writeln!(slurm_script, "module load cuda11.1/toolkit")?;
        writeln!(slurm_script, "nv-nsight-cu-cli {}", nsight_args.join(" "))?;

        log::debug!("slurm script:\n{}", &slurm_script);

        // upload slurm script
        self.upload_data(&remote_job_path, slurm_script.as_bytes(), None)
            .await?;

        let job_id = self.submit_job(&remote_job_path).await?;
        log::info!("slurm: submitted job <{}> [ID={}]", &job_name, job_id);

        self.wait_for_job(job_id, std::time::Duration::from_secs(2), Some(2))
            .await?;

        let stderr = self
            .read_remote_file(&remote_stderr_path, true)
            .await
            .as_deref()
            .unwrap_or("")
            .trim()
            .to_string();
        if !stderr.is_empty() {
            log::error!("{}", stderr);
        }
        let raw_metrics_log = self.read_remote_file(&remote_stdout_path, false).await?;
        let metrics: Vec<profile::nsight::Metrics> =
            profile::nsight::parse_nsight_csv(&mut std::io::Cursor::new(&raw_metrics_log))
                .map_err(|source| {
                    profile::Error::Parse {
                        raw_log: raw_metrics_log.clone(),
                        source,
                    }
                    .into_eyre()
                })?;

        Ok(profile::nsight::Output {
            raw_metrics_log,
            metrics,
        })
    }

    async fn profile_nvprof<A>(
        &self,
        gpu: &str,
        executable: impl AsRef<Path> + Send + Sync,
        args: A,
        timeout: Option<std::time::Duration>,
    ) -> eyre::Result<profile::nvprof::Output>
    where
        A: Clone + IntoIterator + Send,
        <A as IntoIterator>::Item: AsRef<std::ffi::OsStr>,
    {
        use std::fmt::Write;
        let args: Vec<String> = args
            .into_iter()
            .map(|arg| arg.as_ref().to_string_lossy().to_string())
            .collect();

        let (job_name, remote_profile_dir) =
            prepare_profiling(self, Profiler::Nvprof, gpu, executable.as_ref(), &*args).await?;

        let remote_job_path = remote_profile_dir.join("job.slurm");
        let remote_stdout_path = remote_profile_dir.join("stdout.log");
        let remote_stderr_path = remote_profile_dir.join("stderr.log");
        let remote_commands_log_path = remote_profile_dir.join("commands.log");
        let remote_metrics_log_path = remote_profile_dir.join("metrics.log");

        let metrics_args = profile::nvprof::build_metrics_args(
            executable.as_ref(),
            &*args,
            remote_metrics_log_path.as_ref(),
        )?;

        let commands_args = profile::nvprof::build_command_args(
            executable.as_ref(),
            &*args,
            remote_commands_log_path.as_ref(),
        )?;

        // build slurm script
        let mut slurm_script = String::new();
        writeln!(slurm_script, "#!/bin/sh")?;
        writeln!(slurm_script, "#SBATCH --job-name={}", job_name)?;
        writeln!(
            slurm_script,
            "#SBATCH --output={}",
            remote_stdout_path.display()
        )?;
        writeln!(
            slurm_script,
            "#SBATCH --error={}",
            remote_stderr_path.display()
        )?;

        if let Some(timeout) = timeout {
            writeln!(
                slurm_script,
                "#SBATCH --time={}",
                remote::slurm::duration_to_slurm(&timeout)
            )?;
        }
        writeln!(slurm_script, "#SBATCH -N 1")?;
        writeln!(slurm_script, "#SBATCH -C {}", gpu)?;
        writeln!(slurm_script, "#SBATCH --gres=gpu:1")?;
        writeln!(slurm_script, "module load cuda11.1/toolkit")?;
        writeln!(slurm_script, "nvprof {}", commands_args.join(" "))?;
        writeln!(slurm_script, "nvprof {}", metrics_args.join(" "))?;

        log::debug!("slurm script:\n{}", &slurm_script);

        // upload slurm script
        self.upload_data(&remote_job_path, slurm_script.as_bytes(), None)
            .await?;

        let job_id = self.submit_job(&remote_job_path).await?;
        log::info!("slurm: submitted job <{}> [ID={}]", &job_name, job_id);

        self.wait_for_job(job_id, std::time::Duration::from_secs(2), Some(2))
            .await?;

        let stdout = self.read_remote_file(&remote_stdout_path, true).await;
        let stderr = self.read_remote_file(&remote_stderr_path, true).await;

        let stderr = stderr.as_deref().unwrap_or("").trim();
        if !stderr.is_empty() {
            log::error!("{}", stderr);
        }
        let stdout = stdout.as_deref().unwrap_or("").trim();
        if !stdout.is_empty() {
            log::debug!("{}", stdout);
        }

        let raw_commands_log = self
            .read_remote_file(&remote_commands_log_path, false)
            .await;
        let raw_metrics_log = self.read_remote_file(&remote_metrics_log_path, false).await;

        let raw_metrics_log = raw_metrics_log?;
        log::debug!("METRICS:  {}", &raw_metrics_log);
        let metrics: Vec<profile::nvprof::Metrics> =
            profile::nvprof::parse_nvprof_csv(&mut std::io::Cursor::new(&raw_metrics_log))
                .map_err(|source| {
                    profile::Error::Parse {
                        raw_log: raw_metrics_log.clone(),
                        source,
                    }
                    .into_eyre()
                })?;

        let raw_commands_log = raw_commands_log?;
        log::debug!("COMMANDS: {}", &raw_commands_log);
        let commands: Vec<profile::nvprof::Command> =
            profile::nvprof::parse_nvprof_csv(&mut std::io::Cursor::new(&raw_commands_log))
                .map_err(|source| {
                    profile::Error::Parse {
                        raw_log: raw_commands_log.clone(),
                        source,
                    }
                    .into_eyre()
                })?;

        Ok(profile::nvprof::Output {
            raw_metrics_log,
            raw_commands_log,
            metrics,
            commands,
        })
    }
}

async fn connect_das(profile_options: &options::Profile) -> eyre::Result<remote::SSHClient> {
    let use_das6 = profile_options.das == Some(6);
    let port = if use_das6 {
        DAS6_FORWARD_PORT
    } else {
        DAS5_FORWARD_PORT
    };
    let host = "localhost".to_string();
    let username = std::env::var(if use_das6 {
        "DAS6_USERNAME"
    } else {
        "DAS5_USERNAME"
    })
    .ok()
    .ok_or(eyre::eyre!("missing ssh username"))?;
    let password = std::env::var(if use_das6 {
        "DAS6_PASSWORD"
    } else {
        "DAS5_PASSWORD"
    })
    .ok()
    .ok_or(eyre::eyre!("missing ssh password"))?;

    let addr = std::net::ToSocketAddrs::to_socket_addrs(&(host.as_str(), port))
        .map_err(eyre::Report::from)?
        .next()
        .ok_or(eyre::eyre!("failed to resolve {}:{}", host, port))?;
    let das = remote::SSHClient::connect(addr, username, password).await?;
    log::info!("connected to {}", addr);
    Ok(das)
}

pub async fn profile(
    bench: &BenchmarkConfig,
    options: &Options,
    profile_options: &options::Profile,
    _bar: &indicatif::ProgressBar,
) -> Result<Duration, RunError> {
    let TargetBenchmarkConfig::Profile {
        ref profile_dir, ..
    } = bench.target_config
    else {
        unreachable!();
    };

    if let (Some(false), Some(false)) = (profile_options.use_nvprof, profile_options.use_nsight) {
        return Err(RunError::Failed(eyre::eyre!(
            "must use either nvprof or nsight"
        )));
    }

    if options.clean {
        utils::fs::remove_dir(profile_dir).map_err(eyre::Report::from)?;
    }

    create_dirs(profile_dir).map_err(eyre::Report::from)?;

    let remote = if let Some(ref gpu) = profile_options.gpu {
        Some((gpu, connect_das(profile_options).await?))
    } else {
        None
    };

    let start = Instant::now();
    for repetition in 0..bench.common.repetitions {
        // #[cfg(feature = "cuda")]
        // crate::cuda::flush_l2(None)?;

        let mut profiling_errors = Vec::new();
        if profile_options.use_nvprof.unwrap_or(true) {
            let metrics_log_file =
                profile_dir.join(format!("profile.nvprof.metrics.{repetition}.log"));
            let commands_log_file =
                profile_dir.join(format!("profile.nvprof.commands.{repetition}.log"));
            let metrics_file_json =
                profile_dir.join(format!("profile.nvprof.metrics.{repetition}.json"));
            let commands_file_json =
                profile_dir.join(format!("profile.nvprof.commands.{repetition}.json"));

            if !options.force
                && [
                    metrics_log_file.as_path(),
                    commands_log_file.as_path(),
                    metrics_file_json.as_path(),
                    commands_file_json.as_path(),
                ]
                .into_iter()
                .all(Path::is_file)
            {
                return Err(RunError::Skipped);
            }

            let output = if let Some((gpu, ref das)) = remote {
                let remote_repo = profile_options
                    .remote_repo
                    .clone()
                    .unwrap_or(das.remote_scratch_dir().join("gpucachesim"));
                let executable_path = remote_repo
                    .join("test-apps")
                    .join(&bench.rel_path)
                    .join(&bench.executable);

                let output = das
                    .profile_nvprof(
                        gpu,
                        &executable_path,
                        &bench.args,
                        Some(std::time::Duration::from_secs(60 * 60)),
                    )
                    .await;
                output
            } else {
                let options = profile::nvprof::Options {
                    nvprof_path: profile_options.nvprof_path.clone(),
                };
                let output = profile::nvprof::nvprof(&bench.executable_path, &bench.args, &options)
                    .await
                    .map_err(profile::Error::into_eyre);
                output
            };

            if let Ok(ref output) = output {
                open_writable(&metrics_log_file)?
                    .write_all(output.raw_metrics_log.as_bytes())
                    .map_err(eyre::Report::from)?;
                open_writable(&commands_log_file)?
                    .write_all(output.raw_commands_log.as_bytes())
                    .map_err(eyre::Report::from)?;

                serde_json::to_writer_pretty(open_writable(&metrics_file_json)?, &output.metrics)
                    .map_err(eyre::Report::from)?;
                serde_json::to_writer_pretty(open_writable(&commands_file_json)?, &output.commands)
                    .map_err(eyre::Report::from)?;
            }
            profiling_errors.push(output.err());
        }

        if profile_options.use_nsight.unwrap_or(true) {
            let metrics_log_file =
                profile_dir.join(format!("profile.nsight.metrics.{repetition}.log"));
            let metrics_file_json =
                profile_dir.join(format!("profile.nsight.metrics.{repetition}.json"));

            if !options.force
                && [metrics_log_file.as_path(), metrics_file_json.as_path()]
                    .into_iter()
                    .all(Path::is_file)
            {
                return Err(RunError::Skipped);
            }

            let output = if let Some((gpu, ref das)) = remote {
                let remote_repo = profile_options
                    .remote_repo
                    .clone()
                    .unwrap_or(das.remote_scratch_dir().join("gpucachesim"));
                let executable_path = remote_repo
                    .join("test-apps")
                    .join(&bench.rel_path)
                    .join(&bench.executable);

                let output = das
                    .profile_nsight(
                        gpu,
                        &executable_path,
                        &bench.args,
                        Some(std::time::Duration::from_secs(60 * 60)),
                    )
                    .await;
                output
            } else {
                let options = profile::nsight::Options {
                    nsight_path: profile_options.nsight_path.clone(),
                };
                let output = profile::nsight::nsight(&bench.executable_path, &bench.args, &options)
                    .await
                    .map_err(profile::Error::into_eyre);
                output
            };

            if let Ok(ref output) = output {
                open_writable(&metrics_log_file)?
                    .write_all(output.raw_metrics_log.as_bytes())
                    .map_err(eyre::Report::from)?;
                serde_json::to_writer_pretty(open_writable(&metrics_file_json)?, &output.metrics)
                    .map_err(eyre::Report::from)?;
            }
            profiling_errors.push(output.err());
        }

        if profiling_errors.iter().all(Option::is_some) {
            return Err(RunError::Failed(
                eyre::eyre!(
                    "could not profile {} using any of {} profilers",
                    bench.name,
                    profiling_errors.len()
                )
                .with_section(|| {
                    profiling_errors
                        .iter()
                        .filter_map(Option::as_ref)
                        .enumerate()
                        .map(|(i, err)| format!("({}) {}", i, err))
                        .collect::<Vec<_>>()
                        .join(",")
                        .header("errors")
                }),
            ));
        }
    }
    Ok(start.elapsed())
}
