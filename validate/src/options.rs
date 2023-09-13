use crate::Target;
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

    #[clap(long = "target", help = "expand benchmark configs for given target")]
    pub target: Option<crate::Target>,
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

impl Command {
    pub fn targets(&self) -> Box<dyn Iterator<Item = Target>> {
        use strum::IntoEnumIterator;
        match self {
            Command::Full(_) => Box::new(Target::iter()), // all
            Command::Simulate(_) => Box::new([Target::Simulate].into_iter()),
            Command::Trace(_) => Box::new([Target::Trace].into_iter()),
            Command::AccelsimTrace(_) => Box::new([Target::AccelsimTrace].into_iter()),
            Command::AccelsimSimulate(_) => Box::new([Target::AccelsimSimulate].into_iter()),
            Command::PlaygroundSimulate(_) => Box::new([Target::PlaygroundSimulate].into_iter()),
            Command::Profile(_) => Box::new([Target::Profile].into_iter()),
            Command::Build(_) | Command::Clean(_) => Box::new([Target::Profile].into_iter()),
            Command::Expand(Expand { target, .. }) => {
                Box::new([target.unwrap_or(Target::Simulate)].into_iter())
            }
        }
    }
}

#[derive(Parser, Debug, Clone)]
pub struct Options {
    #[clap(short = 'p', long = "path", help = "path to benchmarks yaml file")]
    pub benches_file_path: Option<PathBuf>,

    #[clap(short = 'b', long = "bench", help = "name of benchmark to run")]
    pub selected_benchmarks: Vec<String>,

    #[clap(short = 'q', long = "query", help = "input query")]
    pub query: Vec<String>,

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
