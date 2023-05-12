use async_process::{Command, Output};
use std::path::Path;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("missing libtrace.so shared library")]
    MissingSharedLib,

    #[error("command failed {0:?}")]
    Command(Output),
    // #[error(transparent)]
    // Command(#[from] CommandError),
}

/// Trace a test application.
///
/// # Errors
/// When test app cannot be traced.
pub async fn trace<P, A, D>(executable: P, args: A, trace_dir: D) -> Result<(), Error>
where
    P: AsRef<Path>,
    A: IntoIterator,
    <A as IntoIterator>::Item: AsRef<std::ffi::OsStr>,
    D: AsRef<Path>,
{
    let current_exe = std::env::current_exe()?;
    let target_dir = current_exe.parent().ok_or(Error::MissingSharedLib)?;
    let tracer_so = target_dir.join("libtrace.so");
    if !tracer_so.is_file() {
        return Err(Error::MissingSharedLib);
    }

    let mut cmd = Command::new(executable.as_ref());
    cmd.args(args);
    cmd.env(
        "TRACES_DIR",
        &trace_dir
            .as_ref()
            .canonicalize()?
            .to_string_lossy()
            .to_string(),
    );
    cmd.env(
        "LD_PRELOAD",
        &tracer_so.canonicalize()?.to_string_lossy().to_string(),
    );

    dbg!(&tracer_so);
    dbg!(&cmd);

    let result = cmd.output().await?;
    if !result.status.success() {
        return Err(Error::Command(result));
    }
    println!("{}", String::from_utf8_lossy(&result.stdout));
    println!("{}", String::from_utf8_lossy(&result.stderr));
    Ok(())
}

#[cfg(test)]
mod tests {}
