use async_process::Command;
use std::path::{Path, PathBuf};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Fs(#[from] utils::fs::Error),

    #[error("missing libtrace.so shared library")]
    MissingSharedLib,

    #[error("executable {0:?} not found")]
    MissingExecutable(PathBuf),

    #[error(transparent)]
    Command(#[from] utils::CommandError),

    #[error(transparent)]
    Join(#[from] tokio::task::JoinError),
}

#[derive(Debug, Clone)]
pub struct Options {
    pub traces_dir: PathBuf,
    pub save_json: bool,
    pub full_trace: bool,
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
pub async fn trace<A>(executable: impl AsRef<Path>, args: A, options: &Options) -> Result<(), Error>
where
    A: IntoIterator,
    <A as IntoIterator>::Item: AsRef<std::ffi::OsStr>,
{
    let tracer_so = options
        .tracer_so
        .clone()
        .or_else(find_trace_so)
        .ok_or(Error::MissingSharedLib)?;
    let tracer_so = tracer_so
        .canonicalize()
        .map_err(|_| Error::MissingSharedLib)?;

    let traces_dir = &options.traces_dir;

    utils::fs::create_dirs(traces_dir)?;

    let executable = executable
        .as_ref()
        .canonicalize()
        .map_err(|_| Error::MissingExecutable(executable.as_ref().into()))?;

    let mut cmd = Command::new(executable);
    // configure application
    cmd.args(args);

    // let mut cmd = duct::cmd(executable, args.into_iter());
    // let cmd = cmd.env("TRACES_DIR", traces_dir.to_string_lossy().to_string());
    // let cmd = cmd.env("SAVE_JSON", if options.save_json { "yes" } else { "no" });
    // let cmd = cmd.env("FULL_TRACE", if options.full_trace { "yes" } else { "no" });
    // let cmd = cmd.env("RUST_LOG", "trace");
    // let cmd = cmd.env("LD_PRELOAD", &tracer_so.to_string_lossy().to_string());

    // configure tracer
    cmd.env("TRACES_DIR", traces_dir.to_string_lossy().to_string());
    cmd.env("SAVE_JSON", if options.save_json { "yes" } else { "no" });
    cmd.env("FULL_TRACE", if options.full_trace { "yes" } else { "no" });
    cmd.env("RUST_LOG", "debug");
    cmd.env("LD_PRELOAD", &tracer_so.to_string_lossy().to_string());

    // dbg!(&cmd);
    // let cmd_string = format!("{:?}", &cmd);

    // let result = cmd.run()?;
    // let result = handle.output().wait()?;

    // let result = tokio::task::spawn_blocking::<_, std::io::Result<Output>>(move || {

    let result = cmd.output().await?;
    // dbg!(&utils::decode_utf8!(result.stderr));
    // dbg!(&utils::decode_utf8!(result.stdout));

    // Ok(result)
    // })
    // .await??;

    // let mut child = cmd.spawn()?;
    // let mut stderr_reader = child.stderr.take().unwrap();
    // let mut stdout_reader = child.stdout.take().unwrap();
    //
    // tokio::task::spawn(async move {
    //     use std::time::Duration;
    //     let mut buffer: Vec<u8> = Vec::new();
    //     let copy_fut = futures::io::copy(&mut stderr_reader, &mut buffer);
    //     let _ = tokio::time::timeout(Duration::from_secs(20), copy_fut).await;
    //     let stderr = String::from_utf8_lossy(&buffer).to_string();
    //     println!("stdout: {}", &stderr);
    // });
    // // tokio::task::spawn(async move {
    // //     use std::time::Duration;
    // //     let mut buffer: Vec<u8> = Vec::new();
    // //     let copy_fut = futures::io::copy(&mut stdout_reader, &mut buffer);
    // //     let _ = tokio::time::timeout(Duration::from_secs(20), copy_fut).await;
    // //     let stdout = String::from_utf8_lossy(&buffer).to_string();
    // //     println!("stdout: {}", &stdout);
    // // });
    //
    // let result = child.output().await?;
    //
    if result.status.success() {
        Ok(())
    } else {
        Err(Error::Command(utils::CommandError::new(&cmd, result)))
    }
}
