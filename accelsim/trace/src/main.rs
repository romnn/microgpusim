#![allow(clippy::missing_errors_doc)]

use clap::{CommandFactory, Parser};
use color_eyre::eyre;
use console::style;
use once_cell::sync::Lazy;
use std::path::PathBuf;
use std::time::Instant;

const HELP_TEMPLATE: &str = "{bin} {version} {author}

{about}

USAGE: {usage}

{all-args}
";

static USAGE: Lazy<String> = Lazy::new(|| {
    format!(
        "{} [OPTIONS] -- <executable> [args]",
        env!("CARGO_BIN_NAME")
    )
});

#[derive(Parser, Debug, Clone)]
#[clap(
    help_template=HELP_TEMPLATE,
    override_usage=USAGE.to_string(),
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
    env_logger::init();
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

    let temp_dir = tempfile::tempdir()?;
    let traces_dir = traces_dir
        .clone()
        .map(Result::<PathBuf, utils::TraceDirError>::Ok)
        .unwrap_or_else(|| {
            // Ok(utils::debug_trace_dir(&exec, exec_args.as_slice())?.join("accel-trace"))
            Ok(temp_dir.path().to_path_buf())
        })?;

    let traces_dir = utils::fs::normalize_path(traces_dir);
    utils::fs::create_dirs(&traces_dir)?;
    log::info!("trace dir: {}", traces_dir.display());

    let trace_options = accelsim_trace::Options {
        traces_dir,
        nvbit_tracer_tool,
        kernel_number: Some(kernel_number),
        terminate_upon_limit,
    };

    accelsim_trace::trace(&exec, &exec_args, &trace_options).await?;
    println!(
        "tracing {} took {:?}",
        style(format!("{} {}", exec.display(), exec_args.join(" "))).cyan(),
        start.elapsed()
    );
    Ok(())
}
