use color_eyre::eyre;
use std::fs;
use std::path::{Path, PathBuf};

pub fn remove_dir(path: impl AsRef<Path>) -> eyre::Result<()> {
    if path.as_ref().exists() {
        Ok(fs::remove_dir_all(path.as_ref())?)
    } else {
        Ok(())
    }
}

pub fn remove_file(path: impl AsRef<Path>) -> eyre::Result<()> {
    if path.as_ref().exists() {
        Ok(fs::remove_file(path.as_ref())?)
    } else {
        Ok(())
    }
}

pub fn clean_files(pattern: &str) -> eyre::Result<()> {
    let files: Result<Vec<PathBuf>, _> = glob::glob(pattern)?.collect();
    files?.iter().try_for_each(remove_file)
}

pub fn multi_glob<I, S>(patterns: I) -> impl Iterator<Item = Result<PathBuf, glob::GlobError>>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let globs = patterns.into_iter().map(|p| glob::glob(p.as_ref()));
    globs.flat_map(|x| x).flat_map(|x| x)
}
