use color_eyre::eyre::{self, WrapErr};
use std::path::Path;

fn build_accelsim(accel_path: &Path, cuda_path: &Path, force: bool) -> eyre::Result<()> {
    use std::fs;
    use std::io::Write;

    let artifact = accelsim::executable(accel_path);
    if !force && artifact.is_file() {
        println!("cargo:warning=using existing {}", &artifact.display());
        return Ok(());
    }

    let tmp_build_sh_path = accelsim::build::output_path()?.join("build.tmp.sh");

    let setup_script = accel_path.join("gpu-simulator/setup_environment.sh");
    let setup_script = setup_script
        .canonicalize()
        .wrap_err_with(|| format!("{} does not exist", setup_script.display()))?;

    let accel_build_src = accel_path.join("gpu-simulator/");
    let accel_build_src = accel_build_src
        .canonicalize()
        .wrap_err_with(|| format!("{} does not exist", accel_build_src.display()))?;

    let tmp_build_sh = format!(
        "set -e
source {setup_script} {profile}
make -C {src} clean
make -j -C {src}",
        setup_script = &setup_script.to_string_lossy(),
        profile = &accelsim::profile(),
        src = &accel_build_src.to_string_lossy()
    );
    dbg!(&tmp_build_sh);

    let mut tmp_build_sh_file = fs::OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(&tmp_build_sh_path)?;
    tmp_build_sh_file.write_all(tmp_build_sh.as_bytes())?;

    let tmp_build_sh_path = tmp_build_sh_path
        .canonicalize()
        .wrap_err_with(|| format!("{} does not exist", tmp_build_sh_path.display()))?;

    let make_cmd = duct::cmd!("bash", &*tmp_build_sh_path.to_string_lossy())
        .env("TERM", std::env::var("TERM").as_deref().unwrap_or_default())
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
        let mut log_file = fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(accelsim::build::manifest_path()?.join(log_name))?;
        // todo: strip ansi color escape codes here
        log_file.write_all(content)?;
    }

    if !result.status.success() {
        eyre::bail!("accelsim build exited with code {:?}", result.status.code());
    }

    println!("cargo:warning=built {}", &artifact.display());
    Ok(())
}

fn main() -> eyre::Result<()> {
    color_eyre::install()?;

    println!("cargo:rerun-if-env-changed=USE_UPSTREAM_ACCELSIM");
    println!("cargo:rerun-if-env-changed=FORCE");
    println!("cargo:rerun-if-env-changed=BUILD");
    println!("cargo:rerun-if-env-changed=build.rs");

    let (_use_upstream, accel_path) = accelsim::locate()?;
    println!("cargo:rerun-if-changed={}", accel_path.display());

    println!(
        "cargo:warning=using accelsim source at {}",
        &accel_path.display()
    );

    let cuda_path = accelsim::find_cuda()?;
    println!("cargo:warning=using cuda at {}", &cuda_path.display());

    let force = accelsim::build::is_force();
    println!("cargo:warning=force={}", &force);

    build_accelsim(&accel_path, &cuda_path, force)?;
    Ok(())
}
