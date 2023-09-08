use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug, Default, Clone)]
pub struct Build {}

#[derive(Parser, Debug, Default, Clone)]
pub struct Clean {}

#[derive(Parser, Debug, Default, Clone)]
pub struct Profile {}

#[derive(Parser, Debug, Default, Clone)]
pub struct Trace {}

#[derive(Parser, Debug, Default, Clone)]
pub struct AccelsimTrace {}

#[derive(Parser, Debug, Default, Clone)]
pub struct Sim {}

#[derive(Parser, Debug, Default, Clone)]
pub struct AccelsimSim {}

#[derive(Parser, Debug, Default, Clone)]
pub struct PlaygroundSim {}

#[derive(Parser, Debug, Default, Clone)]
pub struct Expand {
    #[clap(long = "full", help = "expand full benchmark config")]
    pub full: bool,
}

#[derive(Parser, Debug, Default, Clone)]
pub struct Full {}

#[derive(Parser, Debug, Clone)]
pub enum Command {
    Profile(Profile),
    Trace(Trace),
    AccelsimTrace(AccelsimTrace),
    Simulate(Sim),
    AccelsimSimulate(AccelsimSim),
    PlaygroundSimulate(PlaygroundSim),
    Build(Build),
    Clean(Clean),
    Expand(Expand),
    Full(Full),
}

#[derive(Parser, Debug, Clone)]
pub struct Options {
    #[clap(short = 'p', long = "path", help = "path to benchmarks yaml file")]
    pub benches_file_path: Option<PathBuf>,

    #[clap(short = 'b', long = "bench", help = "name of benchmark to run")]
    pub selected_benchmarks: Vec<String>,

    #[clap(
        short = 'f',
        long = "force",
        help = "force re-run",
        default_value = "false"
    )]
    pub force: bool,

    #[clap(long = "clean", help = "clean results", default_value = "false")]
    pub clean: bool,

    #[clap(
        long = "dry",
        aliases = ["dry-run"],
        help = "dry-run without materializing the config",
        default_value = "false"
    )]
    pub dry_run: bool,

    #[clap(long = "fail-fast", help = "fail fast", default_value = "false")]
    pub fail_fast: bool,

    #[clap(
        short = 'c',
        long = "concurrency",
        help = "number of benchmarks to run concurrently"
    )]
    pub concurrency: Option<usize>,

    #[clap(subcommand)]
    pub command: Command,
}
