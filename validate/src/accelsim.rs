use super::materialized::{self, BenchmarkConfig, TargetBenchmarkConfig};
use crate::das::DAS;
use crate::{
    open_writable,
    options::{self, Options},
    RunError,
};
use accelsim::tracegen;
use color_eyre::{eyre, Help};
use remote::{scp::Client as ScpClient, slurm::Client as SlurmClient, Remote};
use std::fmt::Write;
use std::path::Path;
use std::time::Duration;

#[must_use]
pub fn is_debug() -> bool {
    accelsim_sim::is_debug()
}

fn job_name(gpu: &str, executable: impl AsRef<Path> + Send, args: &[String]) -> String {
    [
        "accelsim-trace".to_string(),
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

fn convert_traces_to_json(
    trace_dir: &Path,
    kernelslist: &Path,
    mem_only: bool,
) -> eyre::Result<()> {
    let mut command_traces =
        tracegen::reader::read_command_traces(trace_dir, kernelslist, mem_only)?;
    for (cmd, traces) in &mut command_traces {
        if let Some(trace_model::Command::KernelLaunch(kernel)) = cmd {
            let json_kernel_trace_name = format!("kernel-{}.json", kernel.id);
            let json_kernel_trace_path = trace_dir.join(&json_kernel_trace_name);
            let mut writer = utils::fs::open_writable(json_kernel_trace_path)?;

            serde_json::to_writer_pretty(&mut writer, &traces)?;

            // update the kernel trace path
            kernel.trace_file = json_kernel_trace_name;
        }
    }

    let commands: Vec<_> = command_traces.iter().map(|(cmd, _)| cmd).collect();

    let json_kernelslist = kernelslist.with_extension("json");
    serde_json::to_writer_pretty(utils::fs::open_writable(json_kernelslist)?, &commands)?;

    Ok(())
}

pub async fn trace(
    bench: &BenchmarkConfig,
    options: &Options,
    trace_options: &options::AccelsimTrace,
    _bar: &indicatif::ProgressBar,
) -> Result<Duration, RunError> {
    let TargetBenchmarkConfig::AccelsimTrace { ref traces_dir, .. } = bench.target_config else {
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

    let kernelslist = traces_dir.join("kernelslist.g");
    if !options.force && kernelslist.is_file() {
        return Err(RunError::Skipped);
    }

    let options = accelsim_trace::Options {
        traces_dir: traces_dir.clone(),
        nvbit_tracer_tool: None, // auto detect
        ..accelsim_trace::Options::default()
    };
    let dur = if let Some((gpu, ref remote)) = remote {
        let remote_repo = remote.remote_scratch_dir().join("gpucachesim");
        dbg!(&remote_repo);

        let job_name = job_name(gpu, &bench.executable_path, &*bench.args);
        dbg!(&job_name);

        let remote_job_dir = remote.remote_scratch_dir().join("trace").join(&job_name);
        dbg!(&remote_job_dir);

        remote.remove_dir(&remote_job_dir).await.ok();
        remote.create_dir_all(&remote_job_dir).await?;

        let remote_traces_dir = remote_job_dir.join("traces");
        remote.create_dir_all(&remote_traces_dir).await?;

        let mut trace_cmd: Vec<String> = vec![
            "--traces-dir".to_string(),
            remote_traces_dir.to_string_lossy().to_string(),
        ];
        // if let Some(tracer_tool) = options.tracer_tool {
        //     cmd.extend([
        //         "--tracer-tool".to_string(),
        //         tracer_tool.to_string_lossy().to_string(),
        //     ])
        // }
        if let Some(number) = options.kernel_number {
            trace_cmd.extend(["--kernel-number".to_string(), number.to_string()]);
        }

        if let Some(limit) = options.terminate_upon_limit {
            trace_cmd.extend(["--terminate-upon-limit".to_string(), limit.to_string()]);
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

        let remote_container_image = remote.remote_scratch_dir().join("accelsim.sif");
        if let Some(ref local_container_image) = trace_options.container_image {
            super::trace::upload_container_image(
                remote,
                &local_container_image,
                &remote_container_image,
                None,
            )
            .await?;
        }

        // build slurm script
        let mut slurm_script = String::new();
        writeln!(&mut slurm_script, "#!/bin/sh").unwrap();
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
        writeln!(slurm_script, "echo $LD_LIBRARY_PATH").unwrap();
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
            "singularity exec --env RUST_LOG=debug --cleanenv"
        )
        .unwrap();
        writeln!(
            slurm_script,
            " --bind {}:{} --bind /cm:/cm --nv {} accelsim-trace {}",
            remote.remote_scratch_dir().display(),
            remote.remote_scratch_dir().display(),
            remote_container_image.display(),
            trace_cmd.join(" "),
        )
        .unwrap();

        log::info!("slurm script:\n{}", &slurm_script);

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

        Duration::ZERO
    } else {
        let dur = accelsim_trace::trace(&bench.executable_path, &bench.args, &options).await?;

        let trace_dur_file = traces_dir.join("trace_time.json");
        serde_json::to_writer_pretty(open_writable(trace_dur_file)?, &dur.as_millis())
            .map_err(eyre::Report::from)?;
        dur
    };

    // convert accelsim traces to JSON for us to easily inspect
    if trace_options.save_json.unwrap_or(false) {
        let mem_only = false;
        convert_traces_to_json(traces_dir, &kernelslist, mem_only)?;
    }
    Ok(dur)
}

impl From<materialized::config::AccelsimSimConfigFiles> for accelsim::SimConfig {
    fn from(val: materialized::config::AccelsimSimConfigFiles) -> Self {
        accelsim::SimConfig {
            config: Some(val.config),
            config_dir: Some(val.config_dir),
            trace_config: Some(val.trace_config),
            inter_config: Some(val.inter_config),
        }
    }
}

pub async fn simulate_bench_config(
    bench: &BenchmarkConfig,
) -> Result<(async_process::Output, Duration), RunError> {
    let TargetBenchmarkConfig::AccelsimSimulate {
        ref traces_dir,
        ref configs,
        ..
    } = bench.target_config
    else {
        unreachable!();
    };

    // todo: allow setting this in test-apps.yml ?
    let kernelslist = traces_dir.join("kernelslist.g");
    if !kernelslist.is_file() {
        return Err(RunError::Failed(
            eyre::eyre!("missing {}", kernelslist.display()).with_suggestion(|| {
                let trace_cmd = format!(
                    "cargo validate -b {}@{} accelsim-trace",
                    bench.name,
                    bench.input_idx + 1
                );
                format!("generate traces first using: `{trace_cmd}`")
            }),
        ));
    }

    let common = &bench.common;

    let timeout = common.timeout.map(Into::into);

    let config: accelsim::SimConfig = configs.clone().into();

    let extra_sim_args: &[String] = &[];
    let stream_output = false;
    let use_upstream = true;
    let (log, dur) = accelsim_sim::simulate_trace(
        &traces_dir,
        &kernelslist,
        &config,
        timeout,
        extra_sim_args,
        stream_output,
        use_upstream,
    )
    .await?;
    Ok((log, dur))
}

pub async fn simulate(
    bench: &BenchmarkConfig,
    options: &Options,
    _sim_options: &options::AccelsimSim,
    _bar: &indicatif::ProgressBar,
) -> Result<Duration, RunError> {
    let TargetBenchmarkConfig::AccelsimSimulate { ref stats_dir, .. } = bench.target_config else {
        unreachable!();
    };
    if options.clean {
        utils::fs::remove_dir(stats_dir).map_err(eyre::Report::from)?;
    }

    utils::fs::create_dirs(stats_dir).map_err(eyre::Report::from)?;

    if !options.force && crate::stats::already_exist(&bench.common, stats_dir) {
        return Err(RunError::Skipped);
    }

    let mut total_dur = Duration::ZERO;
    for repetition in 0..bench.common.repetitions {
        let (output, dur) = simulate_bench_config(bench).await?;
        total_dur += dur;
        process_stats(output.stdout, &dur, stats_dir, repetition)?;
    }
    Ok(total_dur)
}

pub fn process_stats(
    log: impl AsRef<Vec<u8>>,
    dur: &Duration,
    stats_dir: &Path,
    repetition: usize,
) -> Result<(), RunError> {
    use std::io::Write as _;

    // parse stats
    let parse_options = accelsim::parser::Options::default();
    let log_reader = std::io::Cursor::new(log.as_ref());
    let stats = accelsim::Stats {
        is_release_build: !accelsim_sim::is_debug(),
        ..accelsim::parser::parse_stats(log_reader, &parse_options)?
    };

    utils::fs::create_dirs(stats_dir).map_err(eyre::Report::from)?;

    let mut log_file = open_writable(stats_dir.join(format!("log.{repetition}.txt")))?;
    log_file
        .write_all(log.as_ref())
        .map_err(eyre::Report::from)?;

    super::stats::write_csv_rows(
        open_writable(stats_dir.join(format!("raw.stats.{repetition}.csv")))?,
        stats.iter().collect::<Vec<_>>(),
    )?;

    let mut per_kernel_stats: stats::PerKernel = stats.try_into()?;

    let num_kernels = per_kernel_stats.num_kernels();
    let per_kernel_dur = dur.as_millis() / num_kernels as u128;
    for kernel_stats in per_kernel_stats.iter_mut() {
        kernel_stats.sim.elapsed_millis = per_kernel_dur;
    }

    // todo: how to handle this?
    // converted_stats.sim.elapsed_millis = dur.as_millis();

    let full = false;
    crate::stats::write_stats_as_csv(stats_dir, &per_kernel_stats.kernel_stats, repetition, full)?;
    Ok(())
}
