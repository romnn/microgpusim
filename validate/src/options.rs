use crate::Target;
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug, Default, Clone)]
pub struct Build {}

#[derive(Parser, Debug, Default, Clone)]
pub struct Clean {}

#[derive(Parser, Debug, Default, Clone)]
pub struct Profile {
    #[clap(long = "nvprof", help = "use nvprof")]
    pub use_nvprof: Option<bool>,

    #[clap(long = "nvprof-path", help = "path to nvprof installation")]
    pub nvprof_path: Option<PathBuf>,

    #[clap(long = "nsight", help = "use nsight")]
    pub use_nsight: Option<bool>,

    #[clap(
        long = "nsight-path",
        help = "path to nsight installation",
        default_value = "/usr/local/NVIDIA-Nsight-Compute-2019.4/"
    )]
    pub nsight_path: Option<PathBuf>,
}

#[derive(Parser, Debug, Default, Clone)]
pub struct Trace {}

#[derive(Parser, Debug, Default, Clone)]
pub struct AccelsimTrace {
    #[clap(long = "save-json", help = "convert and save traces as JSON")]
    pub save_json: Option<bool>,
}

#[derive(Parser, Debug, Default, Clone)]
pub struct Sim {}

#[derive(Parser, Debug, Default, Clone)]
pub struct ExecDrivenSim {}

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
    /// Exec driven simulation.
    ExecSimulate(Sim),
    AccelsimSimulate(AccelsimSim),
    PlaygroundSimulate(PlaygroundSim),
    Build(Build),
    Clean(Clean),
    Expand(Expand),
    Full(Full),
}

impl std::fmt::Display for Command {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Profile(_) => "profile",
                Self::Trace(_) => "trace",
                Self::AccelsimTrace(_) => "trace[accelsim]",
                Self::Simulate(_) => "simulate[gpucachesim]",
                Self::ExecSimulate(_) => "exec-simulate[gpucachesim]",
                Self::AccelsimSimulate(_) => "simulate[accelsim]",
                Self::PlaygroundSimulate(_) => "simulate[playground]",
                Self::Build(_) => "build",
                Self::Clean(_) => "clean",
                Self::Expand(_) => "expand",
                Self::Full(_) => "full",
            }
        )
    }
}

impl Command {
    #[must_use]
    pub fn targets(&self) -> Box<dyn Iterator<Item = Target>> {
        use strum::IntoEnumIterator;
        match self {
            Command::Full(_) => Box::new(Target::iter()), // all
            Command::Simulate(_) => Box::new([Target::Simulate].into_iter()),
            Command::ExecSimulate(_) => Box::new([Target::ExecDrivenSimulate].into_iter()),
            Command::Trace(_) => Box::new([Target::Trace].into_iter()),
            Command::AccelsimTrace(_) => Box::new([Target::AccelsimTrace].into_iter()),
            Command::AccelsimSimulate(_) => Box::new([Target::AccelsimSimulate].into_iter()),
            Command::PlaygroundSimulate(_) => Box::new([Target::PlaygroundSimulate].into_iter()),
            Command::Profile(_) | Command::Build(_) | Command::Clean(_) => {
                Box::new([Target::Profile].into_iter())
            }
            Command::Expand(Expand { target, .. }) => {
                Box::new([target.unwrap_or(Target::Simulate)].into_iter())
            }
        }
    }
}

#[allow(clippy::struct_excessive_bools)]
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

    #[clap(long = "no-progress", help = "hide progress bar")]
    pub no_progress: bool,

    #[clap(subcommand)]
    pub command: Command,
}
