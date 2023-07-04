#![allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]

use color_eyre::eyre;
use std::env;
use std::fs::{self, File};
use std::path::Path;

fn decompress_tar_bz2(src: impl AsRef<Path>, dest: impl AsRef<Path>) -> eyre::Result<()> {
    let compressed = File::open(src)?;
    let stream = bzip2::read::BzDecoder::new(compressed);
    let mut archive = tar::Archive::new(stream);
    archive.set_overwrite(true);
    archive.set_preserve_mtime(true);
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
        "{NVBIT_RELEASES}/{}/{nvbit_release_archive_name}",
        version.as_ref(),
    ))?;
    println!("cargo:warning=downloading {nvbit_release_archive_url}");

    let archive_path = accelsim::build::output_path()?.join(&nvbit_release_archive_name);
    fs::remove_file(&archive_path).ok();

    let mut nvbit_release_archive_file = File::create(&archive_path)?;
    let mut data = reqwest::blocking::get(nvbit_release_archive_url)?;
    data.copy_to(&mut nvbit_release_archive_file)?;

    decompress_tar_bz2(&archive_path, &dest)?;

    println!(
        "cargo:warning=extracted {nvbit_release_archive_name} to {}",
        dest.as_ref().display()
    );
    Ok(())
}

static NVBIT_RELEASES: &str = "https://github.com/NVlabs/NVBit/releases/download";
static NVBIT_VERSION: &str = "1.5.5";

fn build_accelsim_tracer_tool(
    accel_path: &Path,
    cuda_path: &Path,
    force: bool,
) -> eyre::Result<()> {
    let nvbit_version = env::var("NVBIT_VERSION")
        .ok()
        .unwrap_or_else(|| NVBIT_VERSION.to_string());
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH")?;

    let tracer_nvbit_tool_path = accel_path.join("util/tracer_nvbit");

    let nvbit_path = tracer_nvbit_tool_path.join("nvbit_release");
    if force || !nvbit_path.is_dir() {
        download_nvbit(nvbit_version, target_arch, &tracer_nvbit_tool_path)?;
    }

    let artifact = tracer_nvbit_tool_path.join("tracer_tool/tracer_tool.so");
    if force || !artifact.is_file() {
        let make_cmd = duct::cmd!("make", "-j")
            .dir(tracer_nvbit_tool_path)
            // .env("TERM", env::var("TERM").as_deref().unwrap_or(DEFAULT_TERM))
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

    println!("cargo:rerun-if-env-changed=FORCE");
    println!("cargo:rerun-if-env-changed=BUILD");
    println!("cargo:rerun-if-env-changed=build.rs");

    let use_upstream = false;
    #[cfg(feature = "upstream")]
    let use_upstream = true;

    let accel_path = accelsim::locate(use_upstream)?;
    println!("cargo:rerun-if-changed={}", accel_path.display());

    println!(
        "cargo:warning=using accelsim source at {}",
        &accel_path.display()
    );

    let cuda_path = accelsim::find_cuda()?;
    println!("cargo:warning=using cuda at {}", &cuda_path.display());

    let force = accelsim::build::is_force();
    println!("cargo:warning=force={}", &force);

    build_accelsim_tracer_tool(&accel_path, &cuda_path, force)?;
    Ok(())
}
