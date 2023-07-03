#![allow(clippy::missing_panics_doc, clippy::missing_errors_doc)]

pub mod build;
pub mod cache;
pub mod git;
pub mod options;
pub mod parser;
pub mod read;

use color_eyre::{eyre, Section, SectionExt};
pub use options::{Options, SimConfig};
use std::path::{Path, PathBuf};

#[must_use]
pub fn use_upstream() -> bool {
    let use_upstream = std::env::var("USE_UPSTREAM_ACCELSIM");
    use_upstream.is_ok_and(|v| v.to_lowercase() == "yes")
}

#[must_use]
pub fn find_cuda() -> eyre::Result<PathBuf> {
    let cuda_candidates = cuda_candidates();
    cuda_candidates.iter().cloned().next().ok_or(
        eyre::eyre!("CUDA install path not found")
            .with_section(|| format!("{:?}", &cuda_candidates).header("candidates:")),
    )
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

pub fn locate() -> eyre::Result<(bool, PathBuf)> {
    let use_upstream = use_upstream();
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
    Ok((use_upstream, accelsim_path))
}

pub fn locate_nvbit_tracer() -> eyre::Result<PathBuf> {
    let (_, accelsim_path) = locate()?;
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

#[must_use]
pub fn profile() -> &'static str {
    if build::is_debug() {
        "debug"
    } else {
        "release"
    }
}

#[must_use]
pub fn executable(accel_path: &Path) -> PathBuf {
    accel_path
        .join("gpu-simulator/bin")
        .join(profile())
        .join("accel-sim.out")
}
