use clap::{CommandFactory, Parser};
use color_eyre::eyre;
use std::path::PathBuf;

const HELP_TEMPLATE: &str = "{bin} {version} {author}

{about}

USAGE: {usage}

{all-args}
";

const USAGE: &str = "./trace [OPTIONS] -- <executable> [args]";

#[derive(Parser, Debug, Clone)]
#[clap(
    help_template=HELP_TEMPLATE,
    override_usage=USAGE,
    version = option_env!("CARGO_PKG_VERSION").unwrap_or("unknown"),
    about = "trace CUDA applications",
    author = "romnn <contact@romnn.com>",
)]
pub struct Options {
    #[clap(long = "traces-dir", help = "path to output traces dir")]
    pub traces_dir: Option<PathBuf>,
    #[clap(long = "tracer", help = "path to tracer (e.g. libtrace.so)")]
    pub tracer: Option<PathBuf>,
    #[clap(
        long = "save-json",
        help = "whether to also save JSON traces (default: false)"
    )]
    pub save_json: bool,
    #[clap(
        long = "full-trace",
        help = "trace all instructions, including non-memory instructions (default: false)"
    )]
    pub full_trace: bool,
    #[clap(
        long = "validate",
        help = "perform validation on the traces after collection"
    )]
    pub validate: bool,
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

    let (exec, exec_args, options) = match parse_args() {
        Ok(parsed) => parsed,
        Err(err) => err.exit(),
    };

    let Options {
        traces_dir,
        save_json,
        full_trace,
        validate,
        tracer,
    } = options;

    let traces_dir = if let Some(ref traces_dir) = traces_dir {
        traces_dir.clone()
    } else {
        utils::debug_trace_dir(&exec, exec_args.as_slice())?.join("trace")
    };

    let traces_dir = utils::fs::normalize_path(&traces_dir);
    utils::fs::create_dirs(&traces_dir)?;
    let tracer_so = tracer.as_ref().map(utils::fs::normalize_path);

    let trace_options = invoke_trace::Options {
        traces_dir,
        save_json,
        full_trace,
        validate,
        tracer_so,
    };
    dbg!(&trace_options);
    invoke_trace::trace(exec, exec_args, &trace_options)
        .await
        .map_err(invoke_trace::Error::into_eyre)?;
    println!("tracing done in {:?}", start.elapsed());
    Ok(())
}
