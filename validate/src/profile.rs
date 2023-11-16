use super::materialized::{BenchmarkConfig, TargetBenchmarkConfig};
use crate::{
    open_writable,
    options::{self, Options},
    RunError,
};
use color_eyre::eyre;
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
            std::time::Duration::from_secs(5),
            allow_empty,
            Some(10),
        )
        .await?;

        // read file
        let (mut stream, _) = self.download_file(remote_path).await?;
        let mut content = String::new();
        stream.read_to_string(&mut content).await?;
        Ok(content)
    }

    async fn profile_nvprof<A>(
        &self,
        gpu: &str,
        executable: impl AsRef<Path> + Send,
        args: A,
        timeout: Option<std::time::Duration>,
    ) -> eyre::Result<()>
    where
        A: Clone + IntoIterator + Send,
        <A as IntoIterator>::Item: AsRef<std::ffi::OsStr>;
}

#[async_trait::async_trait]
impl<T> ProfileDAS for T
where
    T: remote::slurm::Client + remote::Remote + remote::scp::Client + Sync,
{
    async fn profile_nvprof<A>(
        &self,
        gpu: &str,
        executable: impl AsRef<Path> + Send,
        args: A,
        timeout: Option<std::time::Duration>,
    ) -> eyre::Result<()>
    where
        A: Clone + IntoIterator + Send,
        <A as IntoIterator>::Item: AsRef<std::ffi::OsStr>,
    {
        let args: Vec<String> = args
            .into_iter()
            .map(|arg| arg.as_ref().to_string_lossy().to_string())
            .collect();

        let job_name = job_name(Profiler::Nvprof, gpu, executable.as_ref(), &*args);

        let remote_profile_dir = self
            .remote_scratch_dir()
            .join("profile-nvprof")
            .join(&job_name);

        let remote_job_path = remote_profile_dir.join("job.slurm");
        let remote_stdout_path = remote_profile_dir.join("stdout.log");
        let remote_stderr_path = remote_profile_dir.join("stderr.log");
        let remote_commands_log_path = remote_profile_dir.join("commands.log");
        let remote_metrics_log_path = remote_profile_dir.join("metrics.log");
        dbg!(&remote_job_path);
        dbg!(&remote_stdout_path);
        dbg!(&remote_stderr_path);
        dbg!(&remote_commands_log_path);
        dbg!(&remote_metrics_log_path);

        let metrics_args = profile::nvprof::build_metrics_args(
            executable.as_ref(),
            &*args,
            remote_metrics_log_path.as_ref(),
        )?;
        // dbg!(&metrics_args);

        let commands_args = profile::nvprof::build_command_args(
            executable.as_ref(),
            &*args,
            remote_commands_log_path.as_ref(),
        )?;
        // dbg!(&commands_args);

        // load cuda toolkit
        let load_module_cmd = "module load cuda11.1/toolkit";
        let (exit_status, stdout, stderr) = self.run_command(load_module_cmd).await?;
        log::debug!("{}", stdout);
        log::error!("{}", stderr);
        assert_eq!(exit_status, 0);

        // create results dir
        let create_dir_cmd = format!("mkdir -p {}", remote_profile_dir.display());
        let (exit_status, stdout, stderr) = self.run_command(create_dir_cmd).await?;
        log::debug!("{}", stdout);
        log::error!("{}", stderr);
        assert_eq!(exit_status, 0);

        // build slurm script
        use std::fmt::Write;
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
        // for k, v in env.items():
        //     slurm_script += "export {}={}\n".format(k, v)
        writeln!(slurm_script, "nvprof {}", commands_args.join(" "))?;

        log::debug!("slurm script:\n{}", &slurm_script);

        // upload slurm script
        self.upload_data(&remote_job_path, slurm_script.as_bytes(), None)
            .await?;

        let job_id = self.submit_job(&remote_job_path).await?;
        log::info!("slurm: submitted job <{}> [ID={}]", &job_name, job_id);

        self.wait_for_job(job_id, std::time::Duration::from_secs(6), Some(2))
            .await?;

        let (stdout, stderr) = futures::join!(
            self.read_remote_file(&remote_stdout_path, true),
            self.read_remote_file(&remote_stderr_path, true),
        );
        log::debug!("{}", stdout.as_deref().unwrap_or(""));
        log::error!("{}", stderr.as_deref().unwrap_or(""));

        let (commands_log, metrics_log) = futures::join!(
            self.read_remote_file(&remote_commands_log_path, false),
            self.read_remote_file(&remote_metrics_log_path, false),
        );

        let metrics_log = metrics_log?;
        let metrics: Vec<profile::nvprof::Metrics> = profile::nvprof::parse_nvprof_csv(
            &mut std::io::Cursor::new(&metrics_log),
        )
        .map_err(|source| {
            profile::Error::Parse {
                raw_log: metrics_log,
                source,
            }
            .into_eyre()
        })?;

        let commands_log = commands_log?;
        let commands: Vec<profile::nvprof::Command> = profile::nvprof::parse_nvprof_csv(
            &mut std::io::Cursor::new(&commands_log),
        )
        .map_err(|source| {
            profile::Error::Parse {
                raw_log: commands_log,
                source,
            }
            .into_eyre()
        })?;

        Ok(())
    }
}

async fn connect_das(profile_options: &options::Profile) -> eyre::Result<remote::SSHClient> {
    let port = if profile_options.das == Some(6) {
        DAS6_FORWARD_PORT
    } else {
        DAS5_FORWARD_PORT
    };
    let host = "localhost".to_string();
    let username = std::env::var("DAS6_USERNAME")
        .ok()
        .ok_or(eyre::eyre!("missing ssh username"))?;
    let password = std::env::var("DAS6_PASSWORD")
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

    let start = Instant::now();
    for repetition in 0..bench.common.repetitions {
        // #[cfg(feature = "cuda")]
        // crate::cuda::flush_l2(None)?;

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

            let output = if let Some(ref gpu) = profile_options.gpu {
                let das = connect_das(profile_options).await?;
                let remote_repo = profile_options
                    .remote_repo
                    .clone()
                    .unwrap_or(das.remote_scratch_dir().join("gpucachesim"));
                let executable_path = remote_repo
                    .join("test-apps")
                    .join(&bench.rel_path)
                    .join(&bench.executable);

                das.profile_nvprof(
                    gpu,
                    &executable_path,
                    &bench.args,
                    Some(std::time::Duration::from_secs(60 * 60)),
                )
                .await?;

                return Err(RunError::Skipped);
                // profile::nvprof::Output {
                //     raw_metrics_log,
                //     raw_commands_log,
                //     metrics,
                //     commands,
                // }
            } else {
                let options = profile::nvprof::Options {
                    nvprof_path: profile_options.nvprof_path.clone(),
                };
                let output = profile::nvprof::nvprof(&bench.executable_path, &bench.args, &options)
                    .await
                    .map_err(profile::Error::into_eyre)?;
                output
            };

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

            let output = if let Some(ref gpu) = profile_options.gpu {
                let das = connect_das(profile_options).await?;
                // das.profile_nsight(gpu, &bench.executable_path, &bench.args)
                // .await?;
                todo!()
            } else {
                let options = profile::nsight::Options {
                    nsight_path: profile_options.nsight_path.clone(),
                };
                let output = profile::nsight::nsight(&bench.executable_path, &bench.args, &options)
                    .await
                    .map_err(profile::Error::into_eyre)?;
                output
            };

            open_writable(&metrics_log_file)?
                .write_all(output.raw_metrics_log.as_bytes())
                .map_err(eyre::Report::from)?;
            serde_json::to_writer_pretty(open_writable(&metrics_file_json)?, &output.metrics)
                .map_err(eyre::Report::from)?;
        }
    }
    Ok(start.elapsed())
}
