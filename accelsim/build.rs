#![allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]

use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

#[inline]
#[must_use]
pub fn output_path() -> PathBuf {
    PathBuf::from(std::env::var("OUT_DIR").unwrap())
        .canonicalize()
        .unwrap()
}

#[inline]
#[must_use]
pub fn manifest_path() -> PathBuf {
    PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .canonicalize()
        .unwrap()
}

#[inline]
fn find_cuda() -> Vec<PathBuf> {
    let mut candidates = vec![
        std::env::var("CUDA_LIBRARY_PATH").ok().map(PathBuf::from),
        Some(PathBuf::from("/opt/cuda")),
        Some(PathBuf::from("/usr/local/cuda")),
    ];
    candidates.extend(
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

pub struct GitRepository {
    pub url: String,
    pub path: PathBuf,
    pub branch: Option<String>,
}

impl GitRepository {
    pub fn shallow_clone(&self) -> Result<(), std::io::Error> {
        use std::io::{Error, ErrorKind};

        let _ = std::fs::remove_dir_all(&self.path);
        let mut cmd = Command::new("git");
        cmd.args(["clone", "--depth=1"]);
        if let Some(branch) = &self.branch {
            cmd.args(["-b", branch]);
        }

        cmd.args([&self.url, &self.path.to_string_lossy().to_string()]);
        println!(
            "cargo:warning=cloning {} to {}",
            &self.url,
            &self.path.display()
        );

        if cmd.status()?.success() {
            Ok(())
        } else {
            Err(Error::new(ErrorKind::Other, "fetch failed"))
        }
    }
}

fn main() {
    let use_remote = std::env::var("USE_REMOTE_ACCELSIM")
        .map(|use_remote| use_remote.to_lowercase() == "yes")
        .unwrap_or(false);
    let mut accelsim_path = manifest_path().join("accel-sim-framework-dev");

    if use_remote {
        accelsim_path = output_path().join("accelsim");
        let repo = GitRepository {
            url: "https://github.com/accel-sim/accel-sim-framework.git".to_string(),
            path: accelsim_path.clone(),
            branch: Some("release".to_string()),
        };
        repo.shallow_clone().unwrap();
    }

    let tmp_run_sh_path = output_path().join("run.tmp.sh");
    let tmp_run_sh = format!(
        "set -e
source {}
make -j -C {}",
        &accelsim_path
            .join("gpu-simulator/setup_environment.sh")
            .canonicalize()
            .unwrap()
            .to_string_lossy(),
        &accelsim_path
            .join("gpu-simulator/")
            .canonicalize()
            .unwrap()
            .to_string_lossy(),
    );
    dbg!(&tmp_run_sh);

    let mut tmp_run_sh_file = std::fs::OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(&tmp_run_sh_path)
        .unwrap();
    tmp_run_sh_file.write_all(tmp_run_sh.as_bytes()).unwrap();

    let mut cmd = Command::new("bash");
    cmd.arg(&*tmp_run_sh_path.canonicalize().unwrap().to_string_lossy());
    let cuda_paths = find_cuda();
    dbg!(&cuda_paths);
    cmd.env(
        "CUDA_INSTALL_PATH",
        cuda_paths.first().expect("CUDA install path not found"),
    );
    dbg!(&cmd);

    let result = cmd.output().unwrap();
    println!("{}", String::from_utf8_lossy(&result.stdout));
    println!("{}", String::from_utf8_lossy(&result.stderr));

    assert!(result.status.success(), "{:?}", result.status.code());
}
