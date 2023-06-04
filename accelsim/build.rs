#![allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]

use color_eyre::eyre;
use duct::cmd;
use std::env;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

static NVBIT_RELEASES: &str = "https://github.com/NVlabs/NVBit/releases/download";
static NVBIT_VERSION: &str = "1.5.5";

static DEFAULT_TERM: &str = "xterm-256color";

#[must_use]
pub fn output_path() -> PathBuf {
    PathBuf::from(env::var("OUT_DIR").unwrap())
        .canonicalize()
        .unwrap()
}

#[must_use]
pub fn manifest_path() -> PathBuf {
    PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .canonicalize()
        .unwrap()
}

#[must_use]
pub fn is_debug_build() -> bool {
    matches!(env::var("PROFILE").ok().as_deref(), Some("DEBUG"))
}

fn decompress_tar_bz2(src: impl AsRef<Path>, dest: impl AsRef<Path>) -> eyre::Result<()> {
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
) -> eyre::Result<()> {
    let nvbit_release_name = format!("nvbit-Linux-{}-{}", arch.as_ref(), version.as_ref());
    let nvbit_release_archive_name = format!("{nvbit_release_name}.tar.bz2");
    let nvbit_release_archive_url = reqwest::Url::parse(&format!(
        "{}/{}/{}",
        NVBIT_RELEASES,
        version.as_ref(),
        nvbit_release_archive_name,
    ))?;
    println!("cargo:warning=downloading {}", nvbit_release_archive_url);

    let archive_path = output_path().join(nvbit_release_archive_name);
    std::fs::remove_file(&archive_path).ok();

    let mut nvbit_release_archive_file = File::create(&archive_path)?;
    reqwest::blocking::get(nvbit_release_archive_url)?.copy_to(&mut nvbit_release_archive_file)?;

    std::fs::remove_file(&dest).ok();
    decompress_tar_bz2(&archive_path, &dest)?;
    Ok(())
}

pub struct GitRepository {
    pub url: String,
    pub path: PathBuf,
    pub branch: Option<String>,
}

impl GitRepository {
    pub fn shallow_clone(&self) -> eyre::Result<()> {
        let _ = std::fs::remove_dir_all(&self.path);
        let mut args = vec!["clone", "--depth=1"];
        if let Some(branch) = &self.branch {
            args.extend(["-b", branch]);
        }
        let path = &*self.path.to_string_lossy();
        args.extend([self.url.as_str(), path]);

        let clone_cmd = cmd("git", &args)
            .env("TERM", env::var("TERM").as_deref().unwrap_or(DEFAULT_TERM))
            .unchecked();

        let result = clone_cmd.run()?;
        println!("{}", String::from_utf8_lossy(&result.stdout));
        eprintln!("{}", String::from_utf8_lossy(&result.stderr));

        if !result.status.success() {
            eyre::bail!(
                "git clone command {:?} exited with code {:?}",
                [&["git"], args.as_slice()].concat(),
                result.status.code()
            );
        }

        Ok(())
    }
}

fn build_accelsim(accel_path: &Path, cuda_path: &Path, _force: bool) -> eyre::Result<()> {
    let artifact = accel_path.join("gpu-simulator/bin/release/accel-sim.out");
    if _force || !artifact.is_file() {
        println!("cargo:warning=using existing {}", &artifact.display());
        return Ok(());
    }

    let tmp_build_sh_path = output_path().join("build.tmp.sh");
    let tmp_build_sh = format!(
        "set -e
source {} {profile}
make -C {src} clean
make -j -C {src}",
        &accel_path
            .join("gpu-simulator/setup_environment.sh")
            .canonicalize()?
            .to_string_lossy(),
        profile = if is_debug_build() { "debug" } else { "release" },
        src = accel_path
            .join("gpu-simulator/")
            .canonicalize()?
            .to_string_lossy()
            .to_string(),
    );
    dbg!(&tmp_build_sh);

    let mut tmp_build_sh_file = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(&tmp_build_sh_path)?;
    tmp_build_sh_file.write_all(tmp_build_sh.as_bytes())?;

    let make_cmd = cmd!(
        "bash",
        &*tmp_build_sh_path.canonicalize()?.to_string_lossy()
    )
    .env("TERM", env::var("TERM").as_deref().unwrap_or(DEFAULT_TERM))
    .env("CUDA_INSTALL_PATH", &*cuda_path.to_string_lossy())
    .stderr_capture()
    .stdout_capture()
    .unchecked();

    let result = make_cmd.run()?;
    println!("{}", &String::from_utf8_lossy(&result.stdout));
    eprintln!("{}", &String::from_utf8_lossy(&result.stderr));

    // write build logs
    for (log_name, content) in &[
        ("build.log.stdout", result.stdout),
        ("build.log.stderr", result.stderr),
    ] {
        let mut log_file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(manifest_path().join(log_name))?;
        // todo: strip ansi color escape codes here
        log_file.write_all(content)?;
    }

    if !result.status.success() {
        eyre::bail!("accelsim build exited with code {:?}", result.status.code());
    }

    println!("cargo:warning=built {}", &artifact.display());
    Ok(())
}

fn build_accelsim_tracer_tool(
    accel_path: &Path,
    cuda_path: &Path,
    force: bool,
) -> eyre::Result<()> {
    let nvbit_version = env::var("NVBIT_VERSION").unwrap_or_else(|_| NVBIT_VERSION.to_string());
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH")?;

    let tracer_nvbit_tool_path = accel_path.join("util/tracer_nvbit");

    let nvbit_path = tracer_nvbit_tool_path.join("nvbit_release");
    if force || !nvbit_path.is_dir() {
        download_nvbit(nvbit_version, target_arch, &nvbit_path)?;
    }

    let artifact = tracer_nvbit_tool_path.join("tracer_tool/tracer_tool.so");
    if force || !artifact.is_file() {
        let make_cmd = cmd!("make", "-j")
            .dir(tracer_nvbit_tool_path)
            .env("TERM", env::var("TERM").as_deref().unwrap_or(DEFAULT_TERM))
            .env("CUDA_INSTALL_PATH", &*cuda_path.to_string_lossy())
            .stderr_capture()
            .stdout_capture()
            .unchecked();

        let result = make_cmd.run()?;
        println!("{}", String::from_utf8_lossy(&result.stdout));
        eprintln!("{}", String::from_utf8_lossy(&result.stderr));

        if !result.status.success() {
            eyre::bail!(
                "tracer tool build exited with code {:?}",
                result.status.code()
            );
        }

        println!("cargo:warning=built {}", &artifact.display());
    } else {
        println!("cargo:warning=using existing {}", &artifact.display());
    }

    Ok(())
}

fn main() -> eyre::Result<()> {
    color_eyre::install()?;

    let use_upstream = env::var("USE_UPSTREAM_ACCELSIM")
        .map(|use_remote| use_remote.to_lowercase() == "yes")
        .unwrap_or(false);

    let accel_path = if use_upstream {
        let dest = output_path().join("accelsim");
        let repo = GitRepository {
            url: "https://github.com/accel-sim/accel-sim-framework.git".to_string(),
            path: dest.clone(),
            branch: Some("release".to_string()),
        };
        repo.shallow_clone()?;
        dest
    } else {
        println!("cargo:rerun-if-changed=accel-sim-framework-dev/");
        manifest_path().join("accel-sim-framework-dev")
    };

    println!(
        "cargo:warning=using accelsim source at {}",
        &accel_path.display()
    );

    let cuda_paths = utils::find_cuda();
    dbg!(&cuda_paths);
    let cuda_path = cuda_paths
        .first()
        .ok_or(eyre::eyre!("CUDA install path not found"))?;
    println!("cargo:warning=using cuda at {}", &cuda_path.display());

    let force = matches!(
        env::var("BUILD")
            .ok()
            .as_deref()
            .map(str::to_lowercase)
            .as_deref(),
        Some("yes")
    );
    println!("cargo:warning=force={}", &force);
    

    build_accelsim_tracer_tool(&accel_path, &cuda_path, force)?;
    build_accelsim(&accel_path, &cuda_path, force)?;
    Ok(())
}
