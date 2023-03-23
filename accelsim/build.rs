#![allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]

use std::path::PathBuf;
use std::process::{Command, Output};

#[inline]
#[must_use]
pub fn output() -> PathBuf {
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

pub struct GitRepository {
    pub url: String,
    pub path: PathBuf,
    pub branch: Option<String>,
}

impl GitRepository {
    pub fn shallow_clone(&self) -> Result<(), std::io::Error> {
        use std::io::{Error, ErrorKind};
        use std::process::Command;

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
    let use_remote = std::env::var("USE_REMOTE_ACCELSIM").is_ok();
    let mut accelsim_path = manifest_path().join("accel-sim-framework-dev");
    // panic!("{}", accelsim_path.display());

    if use_remote {
        accelsim_path = output().join("accelsim");
        let repo = GitRepository {
            url: "https://github.com/accel-sim/accel-sim-framework.git".to_string(),
            path: accelsim_path.clone(),
            branch: Some("dev".to_string()),
        };
        repo.shallow_clone().unwrap();
    }

    let mut cmd = Command::new("bash");
    cmd.args([
        // "-j",
        // "-C",
        &accelsim_path
            .join("gpu-simulator/setup_environment.sh")
            .canonicalize()?
            .to_string_lossy(),
    ]);
    dbg!(&cmd);

    let result = cmd.output().unwrap();
    if !result.status.success() {
        println!("{}", String::from_utf8_lossy(&result.stdout));
        println!("{}", String::from_utf8_lossy(&result.stderr));
        panic!("{:?}", result);
    }

    let mut cmd = Command::new("make");
    cmd.args([
        "-j",
        "-C",
        &accelsim_path
            .join("gpu-simulator/")
            .canonicalize()?
            .to_string_lossy(),
    ]);
    dbg!(&cmd);

    let result = cmd.output().unwrap();
    if !result.status.success() {
        println!("{}", String::from_utf8_lossy(&result.stdout));
        println!("{}", String::from_utf8_lossy(&result.stderr));
        panic!("{:?}", result);
    }
}
