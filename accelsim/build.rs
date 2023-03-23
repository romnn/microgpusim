#![allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]

use std::path::PathBuf;

#[inline]
#[must_use]
pub fn output() -> PathBuf {
    PathBuf::from(std::env::var("OUT_DIR").unwrap())
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

pub fn main() {
    let repo = GitRepository {
        url: "https://github.com/accel-sim/accel-sim-framework.git".to_string(),
        path: output().join("accelsim"),
        branch: Some("release".to_string()),
    };
    repo.shallow_clone().unwrap();
}
