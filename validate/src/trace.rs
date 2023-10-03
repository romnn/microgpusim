use super::materialized::{BenchmarkConfig, TargetBenchmarkConfig};
use crate::{
    open_writable,
    options::{self, Options},
    RunError,
};
use color_eyre::{eyre, Help};
use std::time::Duration;
use utils::fs::create_dirs;

pub async fn trace(
    bench: &BenchmarkConfig,
    options: &Options,
    _trace_options: &options::Trace,
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

    create_dirs(traces_dir).map_err(eyre::Report::from)?;

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
    let dur = invoke_trace::trace(&bench.executable, &bench.args, &options)
        .await
        .map_err(|err| match err {
            invoke_trace::Error::Command(utils::CommandError { ref output, .. }) => {
                let stdout = utils::decode_utf8!(&output.stdout);
                if stdout.contains("not found on PATH") {
                    eyre::Report::from(err).with_suggestion(|| {
                        "Are you running as sudo? Tracing does not require running as sudo and can lead to problems."
                    })
                } else {
                    eyre::Report::from(err)
                }
            }
            err @ invoke_trace::Error::MissingExecutable(_) => eyre::Report::from(err)
                .suggestion("did you build the benchmarks first using `cargo validate build`?"),
            err => err.into_eyre(),
        })?;

    let trace_dur_file = traces_dir.join("trace_time.json");
    serde_json::to_writer_pretty(open_writable(trace_dur_file)?, &dur.as_millis())
        .map_err(eyre::Report::from)?;
    Ok(dur)
}
