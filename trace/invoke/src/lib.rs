use async_process::Command;
use std::path::{Path, PathBuf};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Fs(#[from] utils::fs::Error),

    #[error("missing libtrace shared library at {0:?}")]
    MissingSharedLib(PathBuf),

    #[error("executable {0:?} not found")]
    MissingExecutable(PathBuf),

    #[error("{0:?} is not an executable")]
    BadExecutable(PathBuf),

    #[error(transparent)]
    Command(#[from] utils::CommandError),

    #[error(transparent)]
    Join(#[from] tokio::task::JoinError),
}

impl Error {
    pub fn into_eyre(self) -> color_eyre::Report {
        use color_eyre::{eyre, Help};
        match self {
            Error::Command(err) => err.into_eyre(),
            Error::MissingSharedLib(path) => {
                eyre::Report::from(Error::MissingSharedLib(path.clone())).with_suggestion(|| {
                    let is_release = path.components().any(|c| c.as_os_str() == "release");
                    let cmd = if is_release {
                        "cargo build --release -p trace"
                    } else {
                        "cargo build -p trace"
                    };
                    format!("did you build the tracer first using `{cmd}`?")
                })
            }
            err => err.into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Options {
    pub traces_dir: PathBuf,
    pub save_json: bool,
    pub full_trace: bool,
    pub skip_kernel_prefixes: Vec<String>,
    pub validate: bool,
    pub tracer_so: Option<PathBuf>,
}

#[must_use]
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
) -> Result<std::time::Duration, Error>
where
    A: IntoIterator,
    <A as IntoIterator>::Item: AsRef<std::ffi::OsStr>,
{
    use std::io::Write;

    let tracer_so = options
        .tracer_so
        .clone()
        .or_else(find_trace_so)
        .ok_or(Error::MissingSharedLib(PathBuf::from("libtrace.so")))?;
    let tracer_so = tracer_so
        .canonicalize()
        .map_err(|_| Error::MissingSharedLib(tracer_so))?;

    let traces_dir = &options.traces_dir;

    utils::fs::create_dirs(traces_dir)?;

    let executable = executable
        .as_ref()
        .canonicalize()
        .map_err(|_| Error::MissingExecutable(executable.as_ref().into()))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let metadata = executable.metadata()?;
        let is_executable = metadata.permissions().mode() & 0o111 != 0;
        if !is_executable || metadata.is_dir() {
            return Err(Error::BadExecutable(executable.clone()));
        }
    }

    let mut cmd = Command::new(executable);
    // configure application
    cmd.args(args);

    log::debug!("traces dir = {}", traces_dir.display());
    log::debug!("tracer ld preload = {}", tracer_so.display());
    log::debug!("full trace = {}", options.full_trace);
    log::debug!("skip kernel prefixes = {:?}", options.skip_kernel_prefixes);

    // configure tracer
    cmd.env("TRACES_DIR", traces_dir.to_string_lossy().to_string());
    cmd.env("SAVE_JSON", if options.save_json { "yes" } else { "no" });
    cmd.env("FULL_TRACE", if options.full_trace { "yes" } else { "no" });
    cmd.env(
        "SKIP_KERNEL_PREFIXES",
        options.skip_kernel_prefixes.join(","),
    );
    cmd.env("VALIDATE", if options.validate { "yes" } else { "no" });
    cmd.env("LD_PRELOAD", &tracer_so.to_string_lossy().to_string());

    let start = std::time::Instant::now();
    let result = cmd.output().await?;
    let dur = start.elapsed();

    // stdout just contains nvbit banner and application outputs
    // println!("stderr: {}", utils::decode_utf8!(result.stderr));

    std::io::stdout().write_all(&result.stderr)?;
    std::io::stdout().flush()?;

    if result.status.success() {
        Ok(dur)
    } else {
        Err(Error::Command(utils::CommandError::new(&cmd, result)))
    }
}
