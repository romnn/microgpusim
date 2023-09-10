use color_eyre::eyre;

use clap::{CommandFactory, Parser, Subcommand};
use std::path::PathBuf;

const HELP_TEMPLATE: &str = "{bin} {version} {author}

{about}

USAGE: {usage}

{all-args}
";

const USAGE: &str = "./profile [nvprof|nsight|auto] [OPTIONS] -- <executable> [args]";

/// Options for the nvprof profiler.
#[derive(Parser, Debug, Clone)]
pub struct NvprofOptions {
    #[clap(long = "log-file", help = "output log file")]
    pub log_file: Option<PathBuf>,
}

impl From<NvprofOptions> for profile::nvprof::Options {
    fn from(_options: NvprofOptions) -> Self {
        Self {}
    }
}

/// Options for the nsight profiler.
#[derive(Parser, Debug, Clone)]
pub struct NsightOptions {
    #[clap(long = "log-file", help = "output log file")]
    pub log_file: Option<PathBuf>,
}

#[derive(Subcommand, Debug, Clone)]
pub enum Command {
    Auto,
    /// Profile using `nvprof`
    Nvprof(NvprofOptions),
    /// Profile using `nsight-compute`
    Nsight(NsightOptions),
}

#[derive(Parser, Debug, Clone)]
#[clap(
    help_template=HELP_TEMPLATE,
    override_usage=USAGE,
    version = option_env!("CARGO_PKG_VERSION").unwrap_or("unknown"),
    about = "profile CUDA applications using nvprof or nsight",
    author = "romnn <contact@romnn.com>",
)]
pub struct Options {
    #[clap(long = "metrics-file", help = "output metrics file")]
    pub metrics_file: Option<PathBuf>,

    #[clap(subcommand)]
    pub command: Option<Command>,
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

    let _output = match options.command {
        None | Some(Command::Auto) => todo!(),
        Some(Command::Nvprof(nvprof_options)) => {
            let output = profile::nvprof::nvprof(exec, exec_args, &nvprof_options.into())
                .await
                .map_err(profile::Error::into_eyre)?;
            profile::Metrics::Nvprof(output)
        }
        Some(Command::Nsight(_nsight_options)) => todo!(),
    };

    // Err(source) => Err(Error::Parse { raw_log, source }),
    //     Ok(commands) => Ok((raw_log, commands)),
    // }

    // TODO: nice table view of the most important things
    // TODO: dump the raw output
    // TODO: dump the parsed output as json
    // println!("{:#?}", &output.metrics);
    println!("profiling done in {:?}", start.elapsed());
    Ok(())
}
