use super::materialize::BenchmarkConfig;
use crate::{
    open_writable,
    options::{self, Options},
    RunError,
};
use color_eyre::eyre;
use std::io::Write;
use std::path::Path;
use utils::fs::create_dirs;

pub async fn profile(
    bench: &BenchmarkConfig,
    options: &Options,
    _trace_opts: &options::Profile,
) -> Result<(), RunError> {
    let profile_dir = &bench.profile.profile_dir;
    create_dirs(profile_dir).map_err(eyre::Report::from)?;

    let metrics_log_file = profile_dir.join("profile.nvprof.metrics.log");
    let commands_log_file = profile_dir.join("profile.nvprof.commands.log");
    let metrics_file_json = profile_dir.join("profile.metrics.json");
    let commands_file_json = profile_dir.join("profile.commands.json");

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

    let options = profile::nvprof::Options {};
    let output = profile::nvprof::nvprof(&bench.executable, &bench.args, &options)
        .await
        .map_err(profile::Error::into_eyre)?;

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

    Ok(())
}
