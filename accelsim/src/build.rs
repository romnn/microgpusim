use color_eyre::eyre::{self, WrapErr};
use std::path::PathBuf;

#[must_use]
pub fn is_debug() -> bool {
    match std::env::var("PROFILE").unwrap().as_str() {
        "release" | "bench" => false,
        "debug" => true,
        other => panic!("unknown profile {:?}", other),
    }
    // #[cfg(debug_assertions)]
    // return true;
    // #[cfg(not(debug_assertions))]
    // return false;
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
