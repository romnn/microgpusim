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
