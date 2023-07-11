#![allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
pub mod build;
pub mod fs;

use color_eyre::{eyre, Section, SectionExt};
use std::path::{Path, PathBuf};

pub const GID_NOBODY: libc::gid_t = 65534;
pub const UID_NOBODY: libc::uid_t = 65534;

pub fn chown(
    path: impl AsRef<Path>,
    user_id: libc::uid_t,
    group_id: libc::gid_t,
    follow_symlinks: bool,
) -> std::io::Result<()> {
    let s = path.as_ref().as_os_str().to_str().unwrap();
    let s = std::ffi::CString::new(s.as_bytes()).unwrap();
    let ret = unsafe {
        if follow_symlinks {
            libc::chown(s.as_ptr(), user_id, group_id)
        } else {
            libc::lchown(s.as_ptr(), user_id, group_id)
        }
    };
    if ret == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

pub fn rchown(
    root: impl AsRef<Path>,
    user_id: libc::uid_t,
    group_id: libc::gid_t,
    follow_symlinks: bool,
) -> std::io::Result<()> {
    for entry in std::fs::read_dir(root.as_ref())? {
        let entry = entry?;
        let path = entry.path();
        chown(path, user_id, group_id, follow_symlinks)?;
    }
    Ok(())
}

pub fn multi_glob<I, S>(patterns: I) -> impl Iterator<Item = Result<PathBuf, glob::GlobError>>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let globs = patterns.into_iter().map(|p| glob::glob(p.as_ref()));
    globs.flatten().flatten()
}

pub fn partition_results<O, E, OC, EC>(results: impl IntoIterator<Item = Result<O, E>>) -> (OC, EC)
where
    O: std::fmt::Debug,
    E: std::fmt::Debug,
    OC: std::iter::FromIterator<O>,
    EC: std::iter::FromIterator<E>,
{
    let (succeeded, failed): (Vec<_>, Vec<_>) = results.into_iter().partition(Result::is_ok);
    let succeeded: OC = succeeded.into_iter().map(Result::unwrap).collect();
    let failed: EC = failed.into_iter().map(Result::unwrap_err).collect();
    (succeeded, failed)
}

#[must_use]
pub fn find_cuda() -> Option<PathBuf> {
    let cuda_candidates = cuda_candidates();
    cuda_candidates.first().cloned()
}

#[must_use]
pub fn cuda_candidates() -> Vec<PathBuf> {
    let mut candidates = vec![
        std::env::var("CUDAHOME").ok().map(PathBuf::from),
        std::env::var("CUDA_HOME").ok().map(PathBuf::from),
        std::env::var("CUDA_LIBRARY_PATH").ok().map(PathBuf::from),
        Some(PathBuf::from("/opt/cuda")),
        Some(PathBuf::from("/usr/local/cuda")),
    ];
    candidates.extend(
        // specific cuda versions
        glob::glob("/usr/local/cuda-*")
            .expect("glob cuda")
            .map(Result::ok),
    );

    let mut valid_paths = vec![];
    for base in candidates.iter().flatten() {
        if base.is_dir() {
            valid_paths.push(base.clone());
        }
        let lib = base.join("lib64");
        if lib.is_dir() {
            valid_paths.extend([lib.clone(), lib.join("stubs")]);
        }
        let base = base.join("targets/x86_64-linux");
        if base.join("include/cuda.h").is_file() {
            valid_paths.extend([base.join("lib"), base.join("lib/stubs")]);
        }
    }
    valid_paths
}

/// Normalize paths
///
/// Unlike `std::fs::Path::canonicalize`, this function does not access the file system.
/// Hence, this function can be used for paths that do not (yet) exist.
///
/// # Source:
/// [cargo](https://github.com/rust-lang/cargo/blob/fede83ccf973457de319ba6fa0e36ead454d2e20/src/cargo/util/paths.rs#L61)
#[must_use]
pub fn normalize_path(path: impl AsRef<Path>) -> PathBuf {
    use std::path::Component;
    let mut components = path.as_ref().components().peekable();
    let mut ret = if let Some(c @ Component::Prefix(..)) = components.peek().copied() {
        components.next();
        PathBuf::from(c.as_os_str())
    } else {
        PathBuf::new()
    };

    for component in components {
        match component {
            Component::Prefix(..) => unreachable!(),
            Component::RootDir => {
                ret.push(component.as_os_str());
            }
            Component::CurDir => {}
            Component::ParentDir => {
                ret.pop();
            }
            Component::Normal(c) => {
                ret.push(c);
            }
        }
    }
    ret
}

#[derive(thiserror::Error, Debug)]
pub struct CommandError {
    pub command: String,
    pub output: async_process::Output,
}

impl CommandError {
    #[must_use]
    pub fn new(cmd: &async_process::Command, output: async_process::Output) -> Self {
        Self {
            command: format!("{cmd:?}"),
            output,
        }
    }
}

impl std::fmt::Display for CommandError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "command \"{}\" failed with exit code {:?}",
            self.command,
            self.output.status.code()
        )
    }
}

macro_rules! decode_utf8 {
    ($x:expr) => {
        String::from_utf8_lossy(&*$x).to_string()
    };
}

impl CommandError {
    pub fn into_eyre(self) -> eyre::Report {
        eyre::eyre!(
            "command failed with exit code {:?}",
            self.output.status.code()
        )
        .with_section(|| self.command.header("command:"))
        .with_section(|| decode_utf8!(&self.output.stderr).header("stderr:"))
        .with_section(|| decode_utf8!(&self.output.stdout).header("stdout:"))
    }
}

pub const SUCCESS_CODE: i32 = 0;
pub const BAD_USAGE_CODE: i32 = 2;

/// Exit with exit code.
pub fn safe_exit(code: i32) -> ! {
    use std::io::Write;

    let _ = std::io::stdout().lock().flush();
    let _ = std::io::stderr().lock().flush();

    std::process::exit(code)
}
