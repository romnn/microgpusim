use super::materialized::{BenchmarkConfig, TargetBenchmarkConfig};
use crate::das::DAS;
use crate::{
    open_writable,
    options::{self, Options},
    RunError,
};
use color_eyre::{eyre, Help};
use remote::{scp::Client as ScpClient, slurm::Client as SlurmClient, Remote};
use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;
use utils::fs::{Bytes, PathExt};

fn job_name(gpu: &str, executable: impl AsRef<Path> + Send, args: &[String]) -> String {
    [
        "trace".to_string(),
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

pub async fn upload_container_image<R>(
    remote: &R,
    local_container_image: &Path,
    remote_container_image: &Path,
    hasher: Option<&str>,
) -> eyre::Result<()>
where
    R: remote::Remote + remote::scp::Client,
{
    let hasher = hasher.unwrap_or("md5sum");

    let local_checksum_future = || async {
        // compute local file checksum
        let mut cmd = async_process::Command::new(hasher);
        cmd.arg(local_container_image);
        let result = cmd.output().await.ok()?;
        if !result.status.success() {
            None
        } else {
            let stdout = utils::decode_utf8!(result.stdout);
            let (hash, _) = stdout.split_once(" ")?;
            Some(hash.trim().to_lowercase())
        }
    };

    let remote_checksum_future = || async {
        // compute local file checksum
        let cmd = format!("{} {}", hasher, remote_container_image.display());
        let (exit_status, stdout, _) = remote.run_command(&cmd).await.ok()?;
        if exit_status == 0 {
            let (hash, _) = stdout.split_once(" ")?;
            Some(hash.trim().to_lowercase())
        } else {
            None
        }
    };

    let (local_checksum, remote_checksum) =
        futures::join!(local_checksum_future(), remote_checksum_future());
    log::info!(
        "local hash={:?} remote hash={:?}",
        local_checksum,
        remote_checksum
    );

    if local_checksum == remote_checksum {
        log::info!(
            "skip uploading container image {}: checksums match",
            local_container_image.display(),
        );
    } else {
        // upload the container tar
        let start = std::time::Instant::now();

        remote.remove_file(&remote_container_image).await.ok();
        tokio::time::sleep(Duration::from_secs(10)).await;

        let file = tokio::fs::OpenOptions::new()
            .read(true)
            .open(&local_container_image)
            .await
            .unwrap();
        let size = file.metadata().await.unwrap().len();
        log::info!(
            "uploading container image {} ({}) to {}",
            local_container_image.display(),
            Bytes(size as usize),
            remote_container_image.display(),
        );
        let mut reader = tokio::io::BufReader::new(file);
        remote
            .upload_streamed(&remote_container_image, &mut reader, size, None)
            .await
            .unwrap();
        log::info!(
            "uploaded container image {} ({}) to {} in {:?} ({:3.2} MiB/sec)",
            local_container_image.display(),
            Bytes(size as usize),
            remote_container_image.display(),
            start.elapsed(),
            (size / (1024 * 1024)) as f64 / start.elapsed().as_secs_f64(),
        );

        let remote_checksum = remote_checksum_future().await;
        assert_eq!(
            local_checksum, remote_checksum,
            "checksums do not match after upload"
        );
    }
    Ok(())
}

pub async fn trace(
    bench: &BenchmarkConfig,
    options: &Options,
    trace_options: &options::Trace,
    _bar: &indicatif::ProgressBar,
) -> Result<Duration, RunError> {
    let TargetBenchmarkConfig::Trace {
        ref traces_dir,
        save_json,
        full_trace,
        ref skip_kernel_prefixes,
        ..
    } = bench.target_config
    else {
        unreachable!();
    };

    if options.clean {
        utils::fs::remove_dir(traces_dir).map_err(eyre::Report::from)?;
    }

    utils::fs::create_dirs(traces_dir).map_err(eyre::Report::from)?;

    let remote = if let (Some(das), Some(gpu)) = (&trace_options.das, &trace_options.gpu) {
        Some((gpu, das.connect().await?))
    } else {
        None
    };

    if !options.force && traces_dir.join("commands.json").is_file() {
        return Err(RunError::Skipped);
    }

    #[cfg(debug_assertions)]
    let validate = true;
    #[cfg(not(debug_assertions))]
    let validate = false;

    let options = invoke_trace::Options {
        traces_dir: traces_dir.clone(),
        tracer_so: None, // auto detect
        skip_kernel_prefixes: skip_kernel_prefixes.clone(),
        save_json,
        validate,
        full_trace,
    };
    // todo: use the invoke trace binary on the remote
    let mut dur = Duration::ZERO;
    let output = if let Some((gpu, ref remote)) = remote {
        let remote_repo = remote.remote_scratch_dir().join("gpucachesim");
        dbg!(&remote_repo);

        let job_name = job_name(gpu, &bench.executable_path, &*bench.args);
        dbg!(&job_name);

        let remote_job_dir = remote.remote_scratch_dir().join("trace").join(&job_name);
        dbg!(&remote_job_dir);

        // empty trace dir
        // let delete_dir_cmd = format!("rm -rf {}", remote_job_dir.display());
        // let _ = das.run_command(delete_dir_cmd).await?;
        //
        // // create trace dir
        // let create_dir_cmd = format!("mkdir -p {}", remote_job_dir.display());
        // let (exit_status, stdout, stderr) = das.run_command(create_dir_cmd).await?;
        // if !stdout.is_empty() {
        //     log::debug!("{}", stdout);
        // }
        // if !stderr.is_empty() {
        //     log::error!("{}", stderr);
        // }
        // assert_eq!(exit_status, 0);
        remote.remove_dir(&remote_job_dir).await.ok();
        remote.create_dir_all(&remote_job_dir).await?;

        // let container_mount_dir = PathBuf::from("/mnt");
        let remote_traces_dir = remote_job_dir.join("traces");
        remote.create_dir_all(&remote_traces_dir).await?;

        // ./trace [OPTIONS] -- <executable> [args]
        // let invoke_trace = remote_repo.join("target/release/trace");
        let mut trace_cmd: Vec<String> = vec![
            // invoke_trace.to_string_lossy().to_string(),
            "--traces-dir".to_string(),
            // options.traces_dir.to_string_lossy().to_string(),
            // container_mount_dir.to_string_lossy().to_string(),
            remote_traces_dir.to_string_lossy().to_string(),
            // tmp_remote_traces_dir.to_string_lossy().to_string(),
        ];
        // if let Some(tracer_so) = options.tracer_so {
        //     cmd.extend([
        //         "--tracer".to_string(),
        //         tracer_so.to_string_lossy().to_string(),
        //     ])
        // }
        for prefix in &options.skip_kernel_prefixes {
            trace_cmd.extend(["--skip-kernel-prefixes".to_string(), prefix.to_string()]);
        }

        if options.save_json {
            trace_cmd.extend(["--save-json".to_string()]);
        }

        if options.validate {
            trace_cmd.extend(["--validate".to_string()]);
        }

        if options.full_trace {
            trace_cmd.extend(["--full-trace".to_string()]);
        }

        let remote_executable_path = remote_repo
            .join("test-apps")
            .join(&bench.rel_path)
            .join(&bench.executable);

        trace_cmd.extend([
            "--".to_string(),
            remote_executable_path.to_string_lossy().to_string(),
        ]);
        trace_cmd.extend(bench.args.clone());
        dbg!(&trace_cmd);

        let remote_job_path = remote_job_dir.join("job.slurm");
        let remote_stdout_path = remote_job_dir.join("stdout.log");
        let remote_stderr_path = remote_job_dir.join("stderr.log");
        dbg!(&remote_job_path);

        // let remote_container_tar = remote.remote_scratch_dir().join("trace.tar.gz");
        let remote_container_image = remote.remote_scratch_dir().join("trace.sif");
        if let Some(ref local_container_image) = trace_options.container_image {
            upload_container_image(
                remote,
                &local_container_image,
                &remote_container_image,
                None,
            )
            .await?;
            // build singularity container
            // let (exit_status, stdout, stderr) = remote
            //     .run_command(format!(
            //         "singularity build --sandbox trace docker-archive://{}",
            //         remote_container_tar.to_string_lossy()
            //     ))
            //     .await
            //     .unwrap();
            // if !stdout.is_empty() {
            //     log::debug!("{}", stdout);
            // }
            // if !stderr.is_empty() {
            //     log::error!("{}", stderr);
            // }
            // if exit_status != 0 {
            //     return Err(RunError::Failed(eyre::eyre!(
            //         "failed to build singularity container from {} with code {}",
            //         remote_container_tar.display(),
            //         exit_status
            //     )));
            // }
        }
        // writeln!(
        //     slurm_script,
        //     "singularity build --sandbox trace docker-archive://{}",
        //     remote_container_tar.to_string_lossy()
        // )
        // .unwrap();

        // build slurm script
        let mut slurm_script = String::new();
        writeln!(slurm_script, "#!/bin/sh").unwrap();
        writeln!(slurm_script, "#SBATCH --job-name={}", job_name).unwrap();
        writeln!(
            slurm_script,
            "#SBATCH --output={}",
            remote_stdout_path.display()
        )
        .unwrap();
        writeln!(
            slurm_script,
            "#SBATCH --error={}",
            remote_stderr_path.display()
        )
        .unwrap();

        let timeout = Some(Duration::from_secs(20 * 60));
        if let Some(timeout) = timeout {
            writeln!(
                slurm_script,
                "#SBATCH --time={}",
                remote::slurm::duration_to_slurm(&timeout)
            )
            .unwrap();
        }
        writeln!(slurm_script, "#SBATCH -N 1").unwrap();
        writeln!(slurm_script, "#SBATCH -C {}", gpu).unwrap();
        writeln!(slurm_script, "#SBATCH --gres=gpu:1").unwrap();
        writeln!(slurm_script, "module load cuda11.1/toolkit").unwrap();
        // writeln!(slurm_script, "module load cuda11.1/toolkit").unwrap();
        // writeln!(slurm_script, "module load cuda12.3/toolkit").unwrap();
        writeln!(slurm_script, "echo $LD_LIBRARY_PATH").unwrap();
        // writeln!(slurm_script, "ls -lia /.singularity.d/libs").unwrap();
        writeln!(slurm_script, "nvidia-smi").unwrap();

        // try normal execution
        writeln!(
            slurm_script,
            "{} {}",
            remote_executable_path.to_string_lossy().to_string(),
            bench.args.join(" ")
        )
        .unwrap();

        // try tracing with container
        write!(
            slurm_script,
            // "singularity exec --bind {}:{} --nv {} invoke-trace {}",
            "singularity exec --env RUST_LOG=debug --cleanenv"
        )
        .unwrap();
        writeln!(
            slurm_script,
            " --bind {}:{} --bind /cm:/cm --nv {} invoke-trace {}",
            // remote_traces_dir.display(),
            remote.remote_scratch_dir().display(),
            remote.remote_scratch_dir().display(),
            // container_mount_dir.display(),
            remote_container_image.display(),
            // format!(
            //     // "echo $LD_LIBRARY_PATH && ls -lia /.singularity.d/libs && {} {}",
            //     "echo $LD_LIBRARY_PATH && {} {}",
            //     remote_executable_path.to_string_lossy().to_string(),
            //     bench.args.join(" ")
            // ),
            trace_cmd.join(" "),
        )
        .unwrap();

        log::info!("slurm script:\n{}", &slurm_script);

        // if false {
        // upload slurm script
        remote
            .upload_data(&remote_job_path, slurm_script.as_bytes(), None)
            .await?;

        let job_id = remote.submit_job(&remote_job_path).await?;
        log::info!("slurm: submitted job <{}> [ID={}]", &job_name, job_id);

        remote
            .wait_for_job(job_id, Duration::from_secs(2), Some(2))
            .await?;

        let stdout = remote
            .read_remote_file(&remote_stdout_path, true)
            .await
            .as_deref()
            .unwrap_or("")
            .trim()
            .to_string();
        log::info!("{}", stdout);

        let stderr = remote
            .read_remote_file(&remote_stderr_path, true)
            .await
            .as_deref()
            .unwrap_or("")
            .trim()
            .to_string();
        if !stderr.is_empty() {
            log::error!("{}", stderr);
        }

        let generated_trace_files = remote.read_dir(&remote_traces_dir).await?;
        log::info!(
            "generated {} trace files: {:?}",
            generated_trace_files.len(),
            generated_trace_files
        );

        // download remote traces dir to local traces dir
        log::info!(
            "downloading {} to {}",
            &remote_traces_dir.display(),
            &options.traces_dir.display()
        );
        remote
            .download_directory_recursive(&remote_traces_dir, &options.traces_dir)
            .await?;
        // options.traces_dir
        // let (mut stream, stat) = das.download_file(tmp_remote_traces_dir).await?;
        // }

        // source ./accelsim/accel-sim-framework-dev/gpu-simulator/setup_environment.sh
        // make -j1 -B -C ./accelsim/accel-sim-framework-dev/gpu-simulator/ clean
    } else {
        dur = invoke_trace::trace(&bench.executable_path, &bench.args, &options)
        .await
        .map_err(|err| match err {
            invoke_trace::Error::Command(err) => {
                // let stderr = utils::decode_utf8!(&err.output.stderr);
                let stdout = utils::decode_utf8!(&err.output.stdout);
                if stdout.contains("not found on PATH") {
                    eyre::Report::from(err).with_suggestion(|| {
                        "Are you running as sudo? Tracing (unlike profiling) does not require running as sudo and can lead to problems."
                    })
                } else {
                    err.into_eyre()
                }
            }
            err @ invoke_trace::Error::MissingExecutable(_) => eyre::Report::from(err)
                .suggestion("did you build the benchmarks first using `cargo validate build`?"),
            err => err.into_eyre(),
        })?;

        let trace_dur_file = traces_dir.join("trace_time.json");
        serde_json::to_writer_pretty(open_writable(trace_dur_file)?, &dur.as_millis())
            .map_err(eyre::Report::from)?;
    };
    Ok(dur)
}
