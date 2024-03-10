#![allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
pub mod build;
pub mod fs;

use color_eyre::{eyre, Section, SectionExt};
use std::path::{Path, PathBuf};

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

#[macro_export]
macro_rules! decode_utf8 {
    ($x:expr) => {
        String::from_utf8_lossy(&*$x).to_string()
    };
}

#[macro_export]
macro_rules! box_slice {
    () => (
        std::vec::Vec::new().into_boxed_slice()
    );
    ($elem:expr; $n:expr) => (
        std::vec::from_elem($elem, $n).into_boxed_slice()
    );
    ($($x:expr),+ $(,)?) => (
        std::vec![$($x),+].into_boxed_slice()
    );
}

#[derive(thiserror::Error, Debug)]
pub struct CommandError {
    pub command: String,
    pub output: async_process::Output,
}

impl CommandError {
    #[must_use]
    pub fn new(cmd: &impl std::fmt::Debug, output: async_process::Output) -> Self {
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
            "command {:?} failed with exit code {:?}",
            self.command,
            self.output.status.code()
        )
    }
}

impl CommandError {
    pub fn into_eyre(self) -> eyre::Report {
        let command_section = self.command.clone().header("command:");
        let stdout_section = decode_utf8!(&self.output.stdout).header("stdout:");
        let stderr_section = decode_utf8!(&self.output.stderr).header("stderr:");
        eyre::Report::from(self)
            .with_section(|| command_section)
            .with_section(|| stdout_section)
            .with_section(|| stderr_section)
    }
}

pub const SUCCESS_CODE: i32 = 0;
pub const BAD_USAGE_CODE: i32 = 2;

/// Exit with exit code.
// #[inline]
pub fn safe_exit(code: i32) -> ! {
    use std::io::Write;

    let _ = std::io::stdout().lock().flush();
    let _ = std::io::stderr().lock().flush();

    std::process::exit(code)
}

#[derive(thiserror::Error, Debug)]
pub enum TraceDirError {
    #[error("executable {0:?} has no file stem")]
    MissingFileStem(PathBuf),
}

// #[inline]
pub fn debug_trace_dir(exec: impl AsRef<Path>, args: &[String]) -> Result<PathBuf, TraceDirError> {
    let manifest = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
    let results = manifest.join("../debug_results");
    let name = exec
        .as_ref()
        .file_stem()
        .ok_or_else(|| TraceDirError::MissingFileStem(exec.as_ref().to_path_buf()))?;
    let config = format!("{}-{}", &*name.to_string_lossy(), &args.join("-"));
    let traces_dir = self::fs::normalize_path(results.join(name).join(config));
    Ok(traces_dir)
}

pub fn visible_characters(text: &str) -> usize {
    use unicode_segmentation::UnicodeSegmentation;
    let stripped = strip_ansi_escapes::strip(text);
    let Ok(stripped) = std::str::from_utf8(&stripped) else {
        return stripped.len();
    };
    stripped.graphemes(true).count()
}

pub fn next_multiple(value: u64, multiple_of: u64) -> u64 {
    (value as f64 / multiple_of as f64).ceil() as u64 * multiple_of
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_next_multiple() {
        assert_eq!(super::next_multiple(1, 512), 512);
        assert_eq!(super::next_multiple(512, 512), 512);
        assert_eq!(super::next_multiple(513, 512), 1024);
        assert_eq!(super::next_multiple(0, 512), 0);
        assert_eq!(super::next_multiple(1024, 512), 1024);
        assert_eq!(super::next_multiple(1023, 512), 1024);
    }
}
