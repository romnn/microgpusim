use async_process::{Command, Output};
use std::os::unix::fs::DirBuilderExt;
use std::path::{Path, PathBuf};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("missing libtrace.so shared library")]
    MissingSharedLib,

    #[error("executable {0:?} not found")]
    MissingExecutable(PathBuf),

    #[error(transparent)]
    Command(#[from] utils::CommandError),
}

#[derive(Debug, Clone)]
pub struct Options {
    pub traces_dir: PathBuf,
    pub save_json: Option<bool>,
    pub tracer_so: Option<PathBuf>,
}

pub fn find_trace_so() -> Option<PathBuf> {
    // assume running in target/debug or target/release dir
    let current_exe = std::env::current_exe().ok()?;
    let target_dir = current_exe.parent()?;
    let tracer_so = target_dir.join("libtrace.so");
    Some(tracer_so)
}

/// Trace a test application.
///
/// # Errors
/// If tracing the application fails.
pub async fn trace<A>(
    executable: impl AsRef<Path>,
    args: A,
    options: &Options,
) -> Result<Output, Error>
where
    A: IntoIterator,
    <A as IntoIterator>::Item: AsRef<std::ffi::OsStr>,
{
    let tracer_so = options
        .tracer_so
        .clone()
        .or_else(|| find_trace_so())
        .ok_or(Error::MissingSharedLib)?;
    if !tracer_so.is_file() {
        return Err(Error::MissingSharedLib);
    }

    let traces_dir = &options.traces_dir;

    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o777)
        .create(traces_dir)
        .ok();

    let executable = executable.as_ref().to_path_buf();
    let executable = executable.canonicalize().map_err(|_| Error::MissingExecutable(executable))?;

    let mut cmd = Command::new(executable);
    cmd.args(args);
    cmd.env("TRACES_DIR", traces_dir.to_string_lossy().to_string());
    cmd.env(
        "SAVE_JSON",
        if options.save_json.unwrap_or(false) {
            "yes"
        } else {
            "no"
        },
    );

    cmd.env(
        "LD_PRELOAD",
        &tracer_so.canonicalize()?.to_string_lossy().to_string(),
    );

    let result = cmd.output().await?;
    if !result.status.success() {
        Err(Error::Command(utils::CommandError::new(cmd, result)))
    } else {
        Ok(result)
    }
}
