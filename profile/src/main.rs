use color_eyre::eyre;

use clap::{CommandFactory, Parser};
use std::path::PathBuf;

const HELP_TEMPLATE: &str = "{bin} {version} {author}

{about}

USAGE: {usage} -- <executable> [args]

{all-args}
";

const USAGE: &str = "./profile [OPTIONS] -- <executable> [args]";

#[derive(Parser, Debug, Clone)]
#[clap(
    help_template=HELP_TEMPLATE,
    override_usage=USAGE,
    version = option_env!("CARGO_PKG_VERSION").unwrap_or("unknown"),
    about = "profile CUDA applications using nvprof or nsight",
    author = "romnn <contact@romnn.com>",
)]
pub struct Options {
    #[clap(long = "log-file", help = "output log file")]
    pub log_file: Option<PathBuf>,
    #[clap(long = "metrics-file", help = "output metrics file")]
    pub metrics_file: Option<PathBuf>,
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

    let start = std::time::Instant::now();

    let (exec, exec_args, _options) = match parse_args() {
        Ok(parsed) => parsed,
        Err(err) => err.exit(),
    };

    let options = profile::nvprof::Options {};
    let profile::ProfilingResult { .. } = profile::nvprof::nvprof(exec, exec_args, &options)
        .await
        .map_err(|err| match err {
            profile::Error::Command(err) => err.into_eyre(),
            other => other.into(),
        })?;

    // todo: nice table view of the most important things
    // todo: dump the raw output
    // todo: dump the parsed output as json
    // println!("{:#?}", &metrics);
    println!("profiling done in {:?}", start.elapsed());
    Ok(())
}
