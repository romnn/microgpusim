#![allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
pub mod build;

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
    globs.flat_map(|x| x).flat_map(|x| x)
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
    cuda_candidates.iter().cloned().next()
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
