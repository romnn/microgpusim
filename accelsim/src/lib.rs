#![allow(clippy::missing_panics_doc, clippy::missing_errors_doc)]

pub mod build;
pub mod cache;
pub mod git;
pub mod options;
pub mod parser;
pub mod read;
pub mod stats;
pub mod tracegen;

pub use self::stats::Stats;
pub use options::{Options, SimConfig};

use color_eyre::eyre;
use std::path::{Path, PathBuf};

pub fn locate(use_upstream: bool) -> eyre::Result<PathBuf> {
    let accelsim_path = build::manifest_path()?.join(if use_upstream {
        "upstream/accel-sim-framework"
    } else {
        "accel-sim-framework-dev"
    });

    if use_upstream && !accelsim_path.is_dir() {
        // clone repository
        let repo = git::Repository {
            url: "https://github.com/accel-sim/accel-sim-framework.git".to_string(),
            path: accelsim_path.clone(),
            branch: Some("dev".to_string()),
        };
        repo.shallow_clone()?;
    }
    Ok(accelsim_path)
}

pub fn locate_nvbit_tracer(use_upstream: bool) -> eyre::Result<PathBuf> {
    let accelsim_path = locate(use_upstream)?;
    let default_tracer_root = accelsim_path.join("util/tracer_nvbit/");
    let tracer_root = if let Ok(path) = std::env::var("NVBIT_TRACER_ROOT") {
        PathBuf::from(path)
    } else {
        // log::warning!(
        //     "NVBIT_TRACER_ROOT environment variable is not set, trying {}",
        //     default_tracer_root.display()
        // );
        default_tracer_root
    };
    Ok(tracer_root)
}

#[must_use]
pub fn executable(accel_path: &Path, profile: &str) -> PathBuf {
    accel_path
        .join("gpu-simulator/bin")
        .join(profile)
        .join("accel-sim.out")
}
