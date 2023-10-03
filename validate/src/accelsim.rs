use super::materialized::{self, BenchmarkConfig, TargetBenchmarkConfig};
use crate::{
    open_writable,
    options::{self, Options},
    RunError,
};
use accelsim::tracegen;
use color_eyre::{eyre, Help};
use std::io::Write;
use std::path::Path;
use std::time::Duration;

#[must_use]
pub fn is_debug() -> bool {
    accelsim_sim::is_debug()
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
    _trace_opts: &options::AccelsimTrace,
    _bar: &indicatif::ProgressBar,
) -> Result<Duration, RunError> {
    let TargetBenchmarkConfig::AccelsimTrace { ref traces_dir, .. } = bench.target_config else {
        unreachable!();
    };
    utils::fs::create_dirs(traces_dir).map_err(eyre::Report::from)?;

    let kernelslist = traces_dir.join("kernelslist.g");
    if !options.force && kernelslist.is_file() {
        return Err(RunError::Skipped);
    }

    let options = accelsim_trace::Options {
        traces_dir: traces_dir.clone(),
        nvbit_tracer_tool: None, // auto detect
        ..accelsim_trace::Options::default()
    };
    let dur = accelsim_trace::trace(&bench.executable, &bench.args, &options).await?;

    let trace_dur_file = traces_dir.join("trace_time.json");
    serde_json::to_writer_pretty(open_writable(trace_dur_file)?, &dur.as_millis())
        .map_err(eyre::Report::from)?;

    // convert accelsim traces to JSON for us to easily inspect
    let mem_only = false;
    convert_traces_to_json(traces_dir, &kernelslist, mem_only)?;
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
    // parse stats
    let parse_options = accelsim::parser::Options::default();
    let log_reader = std::io::Cursor::new(log.as_ref());
    let stats = accelsim::Stats {
        is_release_build: !accelsim_sim::is_debug(),
        ..accelsim::parser::parse_stats(log_reader, &parse_options)?
    };

    utils::fs::create_dirs(stats_dir).map_err(eyre::Report::from)?;

    open_writable(stats_dir.join(format!("log.{repetition}.txt")))?
        .write_all(log.as_ref())
        .map_err(eyre::Report::from)?;

    super::stats::write_csv_rows(
        open_writable(stats_dir.join(format!("raw.stats.{repetition}.csv")))?,
        stats.iter().collect::<Vec<_>>(),
    )?;

    let mut per_kernel_stats: stats::PerKernel = stats.try_into()?;

    let num_kernels = per_kernel_stats.as_ref().len();
    let per_kernel_dur = dur.as_millis() / num_kernels as u128;
    for kernel_stats in per_kernel_stats.as_mut().iter_mut() {
        kernel_stats.sim.elapsed_millis = per_kernel_dur;
    }

    // todo: how to handle this?
    // converted_stats.sim.elapsed_millis = dur.as_millis();

    crate::stats::write_stats_as_csv(stats_dir, per_kernel_stats.as_ref(), repetition)?;
    Ok(())
}
