use color_eyre::eyre::{self, WrapErr};
use std::path::PathBuf;

#[must_use]
pub fn is_debug() -> bool {
    #[cfg(debug_assertions)]
    return true;
    #[cfg(not(debug_assertions))]
    return false;
    // std::option_env!("PROFILE").is_some_and(|profile| profile.to_lowercase() == "debug")
    // std::env!("PROFILE").to_lowercase() == "debug"
}

#[must_use]
pub fn is_force() -> bool {
    ["BUILD", "FORCE"]
        .into_iter()
        .filter_map(|name| std::env::var(name).ok())
        .map(|var| var.to_lowercase())
        .any(|var| var == "yes")
}

pub fn manifest_path() -> eyre::Result<PathBuf> {
    // let path = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
    let path = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
    let path = path
        .canonicalize()
        .wrap_err_with(|| format!("{} does not exist", path.display()))?;
    Ok(path)
}

pub fn output_path() -> eyre::Result<PathBuf> {
    let path = PathBuf::from(std::env::var("OUT_DIR")?);
    let path = path
        .canonicalize()
        .wrap_err_with(|| format!("{} does not exist", path.display()))?;
    Ok(path)
}
