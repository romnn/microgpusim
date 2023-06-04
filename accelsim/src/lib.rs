#![allow(clippy::missing_panics_doc, clippy::missing_errors_doc)]
// #![allow(warnings)]

pub mod cache;
pub mod options;
pub mod parser;
pub mod read;

use color_eyre::eyre;
pub use options::{Options, SimConfig};
use std::path::PathBuf;

pub fn manifest_path() -> Result<PathBuf, std::io::Error> {
    PathBuf::from(std::env!("CARGO_MANIFEST_DIR")).canonicalize()
}

pub fn locate() -> Result<PathBuf, std::io::Error> {
    let use_remote = std::option_env!("USE_REMOTE_ACCELSIM")
        .map(|use_remote| use_remote.to_lowercase() == "yes")
        .unwrap_or(false);
    let accelsim_path = if use_remote {
        PathBuf::from(std::env!("OUT_DIR"))
            .canonicalize()?
            .join("accelsim")
    } else {
        manifest_path()?.join("accel-sim-framework-dev")
    };
    Ok(accelsim_path)
}

pub fn locate_nvbit_tracer() -> eyre::Result<PathBuf> {
    let accelsim_path = locate()?;
    let default_tracer_root = accelsim_path.join("util/tracer_nvbit/");
    let tracer_root = if let Ok(path) = std::env::var("NVBIT_TRACER_ROOT") {
        PathBuf::from(path)
    } else {
        println!(
            "NVBIT_TRACER_ROOT environment variable is not set, trying {}",
            default_tracer_root.display()
        );
        default_tracer_root
    };
    Ok(tracer_root)
}
