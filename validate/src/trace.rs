use crate::{
    open_writable,
    options::{self, Options},
    RunError,
};
use color_eyre::{eyre, Help};
use utils::fs::create_dirs;
use validate::materialize::BenchmarkConfig;

pub async fn trace(
    bench: &BenchmarkConfig,
    options: &Options,
    _trace_opts: &options::Trace,
) -> Result<(), RunError> {
    let traces_dir = &bench.trace.traces_dir;
    create_dirs(traces_dir).map_err(eyre::Report::from)?;

    if !options.force && traces_dir.join("commands.json").is_file() {
        return Err(RunError::Skipped);
    }

    let options = invoke_trace::Options {
        traces_dir: traces_dir.clone(),
        tracer_so: None, // auto detect
        save_json: bench.trace.save_json,
        #[cfg(debug_assertions)]
        validate: true,
        #[cfg(not(debug_assertions))]
        validate: false,
        full_trace: bench.trace.full_trace,
    };
    let dur = invoke_trace::trace(&bench.executable, &bench.args, &options)
        .await
        .map_err(|err| match err {
            err @ invoke_trace::Error::MissingExecutable(_) => eyre::Report::from(err)
                .suggestion("did you build the benchmarks first using `cargo validate build`?"),
            err => err.into_eyre(),
        })?;

    let trace_dur_file = traces_dir.join("trace_time.json");
    serde_json::to_writer_pretty(open_writable(trace_dur_file)?, &dur.as_millis())
        .map_err(eyre::Report::from)?;
    Ok(())
}
