use crate::{
    open_writable,
    options::{self, Options},
    RunError,
};
use color_eyre::eyre;
use std::io::Write;
use utils::fs::create_dirs;
use validate::materialize::BenchmarkConfig;

pub async fn profile(
    bench: &BenchmarkConfig,
    options: &Options,
    _trace_opts: &options::Profile,
) -> Result<(), RunError> {
    let profile_dir = &bench.profile.profile_dir;
    // dbg!(&profile_dir);
    create_dirs(profile_dir).map_err(eyre::Report::from)?;

    let log_file = profile_dir.join("profile.log");
    let metrics_file = profile_dir.join("profile.metrics.csv");

    if !options.force && log_file.is_file() && metrics_file.is_file() {
        return Err(RunError::Skipped);
    }

    let options = profile::nvprof::Options {};
    let results = profile::nvprof::nvprof(&bench.executable, &bench.args, &options)
        .await
        .map_err(|err| match err {
            profile::Error::Command(err) => err.into_eyre(),
            err => err.into(),
        })?;

    serde_json::to_writer_pretty(open_writable(&metrics_file)?, &results.metrics)
        .map_err(eyre::Report::from)?;
    open_writable(&log_file)?
        .write_all(results.raw.as_bytes())
        .map_err(eyre::Report::from)?;
    Ok(())
}
