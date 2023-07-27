#![allow(clippy::missing_errors_doc)]

use async_process::Command;
use color_eyre::eyre::{self, WrapErr};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct Options {
    pub traces_dir: PathBuf,
    pub nvbit_tracer_tool: Option<PathBuf>,
    pub kernel_number: Option<usize>,
    pub terminate_upon_limit: Option<usize>,
}

pub fn render_trace_script(
    exec: &Path,
    exec_args: &[String],
    traces_dir: &Path,
    nvbit_tracer_tool: &Path,
    kernel_number: Option<usize>,
    terminate_upon_limit: Option<usize>,
) -> eyre::Result<String> {
    let mut env: HashMap<&'static str, String> = HashMap::from_iter([
        // hide nvbit banner
        ("NOBANNER", "1".to_string()),
        // USER_DEFINED_FOLDERS must be set for TRACES_FOLDER variable to be read
        ("USER_DEFINED_FOLDERS", "1".to_string()),
        ("TRACES_FOLDER", traces_dir.to_string_lossy().to_string()),
        (
            "CUDA_INJECTION64_PATH",
            nvbit_tracer_tool
                .join("tracer_tool.so")
                .to_string_lossy()
                .to_string(),
        ),
        (
            "LD_PRELOAD",
            nvbit_tracer_tool
                .join("tracer_tool.so")
                .to_string_lossy()
                .to_string(),
        ),
        (
            "DYNAMIC_KERNEL_LIMIT_END",
            kernel_number.unwrap_or(0).to_string(),
        ),
    ]);
    match terminate_upon_limit {
        Some(terminate_upon_limit) if terminate_upon_limit > 0 => {
            env.insert("TERMINATE_UPON_LIMIT", terminate_upon_limit.to_string());
        }
        _ => {}
    }
    let post_traces_processing = nvbit_tracer_tool.join("traces-processing/post-traces-processing");

    let mut trace_sh: Vec<String> = vec![];
    trace_sh.push("#!/usr/bin/env bash".to_string());
    trace_sh.push("set -e".to_string());
    // trace_sh.push(r#"echo "tracing...""#.to_string());
    trace_sh.extend(env.iter().map(|(k, v)| format!("export {k}=\"{v}\"")));
    trace_sh.push(
        [exec.to_string_lossy().to_string()]
            .into_iter()
            .chain(exec_args.iter().cloned())
            .collect::<Vec<_>>()
            .join(" "),
    );
    let kernelslist = traces_dir.join("kernelslist");
    trace_sh.push(format!(
        "{} {}",
        post_traces_processing.display(),
        kernelslist.display(),
    ));
    trace_sh.push(format!("rm -f {}", traces_dir.join("*.trace").display()));
    trace_sh.push(format!("rm -f {}", kernelslist.display(),));

    Ok(trace_sh.join("\n"))
}

pub async fn trace(
    exec: impl AsRef<Path>,
    exec_args: &[String],
    options: &Options,
) -> eyre::Result<std::time::Duration> {
    #[cfg(feature = "upstream")]
    let use_upstream = true;
    #[cfg(not(feature = "upstream"))]
    let use_upstream = false;

    let exec = exec.as_ref();
    let exec = exec
        .canonicalize()
        .wrap_err_with(|| eyre::eyre!("executable at {} does not exist", exec.display()))?;

    let nvbit_tracer_root = accelsim::locate_nvbit_tracer(use_upstream)?;
    let nvbit_tracer_root = nvbit_tracer_root.canonicalize().wrap_err_with(|| {
        eyre::eyre!(
            "nvbit tracer root at {} does not exist",
            nvbit_tracer_root.display()
        )
    })?;

    let nvbit_tracer_tool = nvbit_tracer_root.join("tracer_tool");
    let nvbit_tracer_tool = nvbit_tracer_tool.canonicalize().wrap_err_with(|| {
        eyre::eyre!(
            "nvbit tracer tool at {} does not exist",
            nvbit_tracer_root.display()
        )
    })?;

    utils::fs::create_dirs(&options.traces_dir)?;

    let tmp_trace_sh = render_trace_script(
        &exec,
        exec_args,
        &options.traces_dir,
        &nvbit_tracer_tool,
        options.kernel_number,
        options.terminate_upon_limit,
    )?;
    log::debug!("{}", &tmp_trace_sh);

    let tmp_trace_sh_path = options.traces_dir.join("trace.tmp.sh");
    {
        let mut tmp_trace_sh_file = utils::fs::open_writable(&tmp_trace_sh_path)?;
        tmp_trace_sh_file.write_all(tmp_trace_sh.as_bytes())?;
    }

    let tmp_trace_sh_path = tmp_trace_sh_path.canonicalize().wrap_err_with(|| {
        eyre::eyre!(
            "temp trace file at {} does not exist",
            tmp_trace_sh_path.display()
        )
    })?;

    let cuda_path = utils::find_cuda().ok_or(eyre::eyre!("CUDA not found"))?;

    let mut cmd = Command::new("bash");
    cmd.arg(&*tmp_trace_sh_path.to_string_lossy());
    cmd.env("CUDA_INSTALL_PATH", &*cuda_path.to_string_lossy());
    log::debug!("command: {:?}", &cmd);

    let start = std::time::Instant::now();
    let result = cmd.output().await?;

    if !result.status.success() {
        return Err(utils::CommandError::new(&cmd, result).into_eyre());
    }

    let dur = start.elapsed();

    log::debug!("stdout:\n{}", utils::decode_utf8!(&result.stdout));
    log::debug!("stderr:\n{}", utils::decode_utf8!(&result.stderr));

    // std::fs::remove_file(&tmp_trace_sh_path).ok();
    Ok(dur)
}
