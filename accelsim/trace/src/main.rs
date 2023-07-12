#![allow(clippy::missing_errors_doc)]

use async_process::Command;
use clap::{CommandFactory, Parser};
use color_eyre::eyre::{self, WrapErr};
use std::collections::HashMap;
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::time::Instant;

const HELP_TEMPLATE: &str = "{bin} {version} {author}

{about}

USAGE: {usage}

{all-args}
";

const USAGE: &str = "./accelsim-trace [OPTIONS] -- <executable> [args]";

#[derive(Parser, Debug, Clone)]
#[clap(
    help_template=HELP_TEMPLATE,
    override_usage=USAGE,
    version = option_env!("CARGO_PKG_VERSION").unwrap_or("unknown"),
    about = "trace CUDA applications using accelsim tracer",
    author = "romnn <contact@romnn.com>",
)]
pub struct Options {
    #[clap(long = "traces-dir", help = "path to output traces dir")]
    pub traces_dir: Option<PathBuf>,
    #[clap(long = "tracer-tool", help = "custom path to nvbit tracer tool")]
    pub nvbit_tracer_tool: Option<PathBuf>,
    #[clap(long = "kernel-number", help = "kernel number", default_value = "0")]
    pub kernel_number: usize,
    #[clap(long = "terminate-upon-limit", help = "terminate upon limit")]
    pub terminate_upon_limit: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TraceOptions {
    pub traces_dir: PathBuf,
    pub nvbit_tracer_tool: Option<PathBuf>,
    pub kernel_number: usize,
    pub terminate_upon_limit: usize,
}

pub fn render_trace_script(
    exec: &Path,
    exec_args: &[String],
    traces_dir: &Path,
    nvbit_tracer_tool: &Path,
    kernel_number: usize,
    terminate_upon_limit: usize,
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
        ("DYNAMIC_KERNEL_LIMIT_END", "0".to_string()),
        ("DYNAMIC_KERNEL_LIMIT_END", kernel_number.to_string()),
    ]);
    if terminate_upon_limit > 0 {
        env.insert("TERMINATE_UPON_LIMIT", terminate_upon_limit.to_string());
    }
    let post_traces_processing = nvbit_tracer_tool.join("traces-processing/post-traces-processing");

    let mut trace_cmds: Vec<String> = vec![];
    trace_cmds.push("set -e".to_string());
    trace_cmds.push(r#"echo "hello from tracing file""#.to_string());
    trace_cmds.extend(env.iter().map(|(k, v)| format!("export {k}=\"{v}\"")));
    trace_cmds.push(
        [exec.to_string_lossy().to_string()]
            .into_iter()
            .chain(exec_args.iter().cloned())
            .collect::<Vec<_>>()
            .join(" "),
    );
    let kernelslist = traces_dir.join("kernelslist");
    trace_cmds.push(format!(
        "{} {}",
        post_traces_processing.display(),
        kernelslist.display(),
    ));
    trace_cmds.push(format!("rm -f {}", traces_dir.join("*.trace").display()));
    trace_cmds.push(format!("rm -f {}", kernelslist.display(),));

    let trace_sh = trace_cmds.join("\n");
    Ok(trace_sh)
}

async fn run_trace(
    exec: impl AsRef<Path>,
    exec_args: Vec<String>,
    options: &TraceOptions,
) -> eyre::Result<()> {
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
        &exec_args,
        &options.traces_dir,
        &nvbit_tracer_tool,
        options.kernel_number,
        options.terminate_upon_limit,
    )?;
    println!("{}", &tmp_trace_sh);

    let tmp_trace_sh_path = options.traces_dir.join("trace.tmp.sh");
    let mut tmp_trace_sh_file = std::fs::OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        // file needs to be executable
        .mode(0o777)
        .open(&tmp_trace_sh_path)?;
    tmp_trace_sh_file.write_all(tmp_trace_sh.as_bytes())?;

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
    dbg!(&cmd);

    let result = cmd.output().await?;
    println!("stdout:\n{}", utils::decode_utf8!(&result.stdout));
    eprintln!("stderr:\n{}", utils::decode_utf8!(&result.stderr));

    if !result.status.success() {
        return Err(utils::CommandError::new(&cmd, result).into_eyre());
    }

    // std::fs::remove_file(&tmp_trace_sh_path).ok();
    Ok(())
}

fn parse_args() -> Result<(PathBuf, Vec<String>, Options), clap::Error> {
    let args: Vec<_> = std::env::args().collect();

    // split arguments for tracer and application
    let split_idx = args
        .iter()
        .position(|arg| arg.trim() == "--")
        .unwrap_or(args.len());
    let mut trace_opts = args;
    let mut exec_args = trace_opts.split_off(split_idx).into_iter();

    // must parse options first for --help to work
    let options = Options::try_parse_from(trace_opts)?;

    exec_args.next(); // skip binary
    let exec = exec_args.next().ok_or(Options::command().error(
        clap::error::ErrorKind::MissingRequiredArgument,
        "missing executable",
    ))?;

    Ok((PathBuf::from(exec), exec_args.collect(), options))
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    color_eyre::install()?;

    let start = Instant::now();

    let (exec, exec_args, options) = match parse_args() {
        Ok(parsed) => parsed,
        Err(err) => err.exit(),
    };
    let Options {
        traces_dir,
        kernel_number,
        nvbit_tracer_tool,
        terminate_upon_limit,
    } = options;

    let traces_dir = if let Some(ref traces_dir) = traces_dir {
        traces_dir.clone()
    } else {
        utils::debug_trace_dir(&exec, exec_args.as_slice())?.join("accel-trace")
    };
    let traces_dir = utils::fs::normalize_path(traces_dir);
    utils::fs::create_dirs(&traces_dir)?;
    dbg!(&traces_dir);
    let trace_options = TraceOptions {
        traces_dir,
        nvbit_tracer_tool,
        kernel_number,
        terminate_upon_limit: terminate_upon_limit.unwrap_or(0),
    };

    run_trace(&exec, exec_args.clone(), &trace_options).await?;
    println!(
        "tracing {} {} took {:?}",
        exec.display(),
        exec_args.join(" "),
        start.elapsed()
    );
    Ok(())
}
