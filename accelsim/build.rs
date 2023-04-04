#![allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]

use anyhow::Result;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

static NVBIT_RELEASES: &str = "https://github.com/NVlabs/NVBit/releases/download";
static NVBIT_VERSION: &str = "1.5.5";

#[must_use]
pub fn output_path() -> PathBuf {
    PathBuf::from(std::env::var("OUT_DIR").unwrap())
        .canonicalize()
        .unwrap()
}

#[must_use]
pub fn manifest_path() -> PathBuf {
    PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .canonicalize()
        .unwrap()
}

fn decompress_tar_bz2(src: impl AsRef<Path>, dest: impl AsRef<Path>) -> Result<()> {
    let compressed = File::open(src)?;
    let stream = bzip2::read::BzDecoder::new(compressed);
    let mut archive = tar::Archive::new(stream);
    archive.unpack(&dest)?;
    Ok(())
}

fn download_nvbit(
    version: impl AsRef<str>,
    arch: impl AsRef<str>,
    dest: impl AsRef<Path>,
) -> Result<()> {
    let nvbit_release_name = format!("nvbit-Linux-{}-{}", arch.as_ref(), version.as_ref());
    let nvbit_release_archive_name = format!("{nvbit_release_name}.tar.bz2");
    let nvbit_release_archive_url = reqwest::Url::parse(&format!(
        "{}/{}/{}",
        NVBIT_RELEASES,
        version.as_ref(),
        nvbit_release_archive_name,
    ))?;

    let archive_path = output_path().join(nvbit_release_archive_name);
    // let nvbit_path = output_path().join(nvbit_release_name);

    // check if the archive already exists
    // let force = false;
    // if force || !nvbit_path.is_dir() {
    std::fs::remove_file(&archive_path).ok();
    let mut nvbit_release_archive_file = File::create(&archive_path)?;
    reqwest::blocking::get(nvbit_release_archive_url)?.copy_to(&mut nvbit_release_archive_file)?;

    // unarchive
    std::fs::remove_file(&dest).ok();
    decompress_tar_bz2(&archive_path, &dest)?;
    // }
    Ok(())
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

fn build_accelsim(
    accelsim_path: impl AsRef<Path>,
    cuda_path: impl AsRef<Path>,
    _force: bool,
) -> Result<()> {
    let artifact = accelsim_path
        .as_ref()
        .join("gpu-simulator/bin/release/accel-sim.out");

    let tmp_build_sh_path = output_path().join("build.tmp.sh");
    let tmp_build_sh = format!(
        "set -e
source {}
make -j -C {}",
        &accelsim_path
            .as_ref()
            .join("gpu-simulator/setup_environment.sh")
            .canonicalize()?
            .to_string_lossy(),
        &accelsim_path
            .as_ref()
            .join("gpu-simulator/")
            .canonicalize()?
            .to_string_lossy(),
    );
    dbg!(&tmp_build_sh);

    let mut tmp_build_sh_file = std::fs::OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(&tmp_build_sh_path)?;
    tmp_build_sh_file.write_all(tmp_build_sh.as_bytes())?;

    let mut cmd = Command::new("bash");
    cmd.arg(&*tmp_build_sh_path.canonicalize()?.to_string_lossy());
    cmd.env("CUDA_INSTALL_PATH", &*cuda_path.as_ref().to_string_lossy());
    dbg!(&cmd);

    let result = cmd.output()?;
    println!("{}", String::from_utf8_lossy(&result.stdout));
    println!("{}", String::from_utf8_lossy(&result.stderr));

    if !result.status.success() {
        anyhow::bail!("cmd failed with code {:?}", result.status.code());
    }

    println!("cargo:warning=built {}", &artifact.display());
    Ok(())
}

fn build_accelsim_tracer_tool(
    accelsim_path: impl AsRef<Path>,
    cuda_path: impl AsRef<Path>,
    force: bool,
) -> Result<()> {
    let nvbit_version =
        std::env::var("NVBIT_VERSION").unwrap_or_else(|_| NVBIT_VERSION.to_string());
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH")?;

    let tracer_nvbit_tool_path = accelsim_path.as_ref().join("util/tracer_nvbit");

    let nvbit_path = tracer_nvbit_tool_path.join("nvbit_release");
    if force || !nvbit_path.is_dir() {
        download_nvbit(nvbit_version, target_arch, &nvbit_path)?;
    }

    let artifact = tracer_nvbit_tool_path.join("tracer_tool");
    let mut cmd = Command::new("make");
    cmd.arg("-j");
    cmd.current_dir(tracer_nvbit_tool_path);
    cmd.env("CUDA_INSTALL_PATH", &*cuda_path.as_ref().to_string_lossy());
    dbg!(&cmd);

    let result = cmd.output()?;
    println!("{}", String::from_utf8_lossy(&result.stdout));
    println!("{}", String::from_utf8_lossy(&result.stderr));

    if !result.status.success() {
        anyhow::bail!("cmd failed with code {:?}", result.status.code());
    }
    println!("cargo:warning=built {}", &artifact.display());
    Ok(())
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
    } else {
        println!("cargo:rerun-if-changed=accel-sim-framework-dev/");
    }
    let cuda_paths = utils::find_cuda();
    dbg!(&cuda_paths);
    let cuda_path = cuda_paths.first().expect("CUDA install path not found");

    // build tracer tool only
    let force = false;
    build_accelsim_tracer_tool(&accelsim_path, cuda_path, force).unwrap();

    // try to build accelsim as well
    if let Err(err) = build_accelsim(&accelsim_path, cuda_path, force) {
        println!("cargo:warning=building accelsim failed: {}", &err,);
    }
}
