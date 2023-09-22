use super::materialized::{BenchmarkConfig, TargetBenchmarkConfig};
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
    _profile_options: &options::Profile,
    _bar: &indicatif::ProgressBar,
) -> Result<(), RunError> {
    let TargetBenchmarkConfig::Profile {
        ref profile_dir, ..
    } = bench.target_config
    else {
        unreachable!();
    };

    if options.clean {
        utils::fs::remove_dir(profile_dir).map_err(eyre::Report::from)?;
    }

    create_dirs(profile_dir).map_err(eyre::Report::from)?;

    for repetition in 0..bench.common.repetitions {
        let metrics_log_file = profile_dir.join(format!("profile.nvprof.metrics.{repetition}.log"));
        let commands_log_file =
            profile_dir.join(format!("profile.nvprof.commands.{repetition}.log"));
        let metrics_file_json = profile_dir.join(format!("profile.metrics.{repetition}.json"));
        let commands_file_json = profile_dir.join(format!("profile.commands.{repetition}.json"));

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

        #[cfg(feature = "cuda")]
        crate::cuda::flush_l2(None)?;

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
    }
    Ok(())
}
