use crate::Target;
use clap::Parser;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Das {
    Das5,
    Das6,
}

#[derive(thiserror::Error, Debug)]
#[error("failed to parse das cluster {value:?} must be either 5 or 6")]
pub struct InvalidDas {
    value: String,
}

impl TryFrom<&str> for Das {
    type Error = InvalidDas;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value.parse::<u32>() {
            Ok(5) => Ok(Das::Das5),
            Ok(6) => Ok(Das::Das6),
            Err(_) | Ok(_) => Err(InvalidDas {
                value: value.to_string(),
            }),
        }
    }
}

impl std::str::FromStr for Das {
    type Err = InvalidDas;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::try_from(value)
    }
}

pub fn use_remote(das: &Option<Das>, gpu: &Option<String>) -> bool {
    #[cfg(feature = "remote")]
    return das.is_some() && gpu.is_some();
    false
}

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

    #[clap(long = "gpu", help = "gpu device to profile")]
    pub gpu: Option<String>,

    #[clap(long = "das", help = "das cluster to connect to")]
    pub das: Option<Das>,

    #[clap(long = "repo", help = "path to remote repository")]
    pub remote_repo: Option<PathBuf>,
}

#[derive(Parser, Debug, Default, Clone)]
pub struct Trace {
    #[clap(long = "das", help = "das cluster to connect to")]
    pub das: Option<Das>,

    #[clap(long = "gpu", help = "gpu device to profile")]
    pub gpu: Option<String>,

    #[clap(long = "container", help = "local path to singularity container image")]
    pub container_image: Option<PathBuf>,
}

#[derive(Parser, Debug, Default, Clone)]
pub struct AccelsimTrace {
    #[clap(long = "save-json", help = "convert and save traces as JSON")]
    pub save_json: Option<bool>,

    #[clap(long = "das", help = "das cluster to connect to")]
    pub das: Option<Das>,

    #[clap(long = "gpu", help = "gpu device to profile")]
    pub gpu: Option<String>,

    #[clap(long = "container", help = "local path to singularity container image")]
    pub container_image: Option<PathBuf>,
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
pub struct All {}

#[derive(Parser, Debug, Default, Clone)]
pub struct Run {
    #[clap(short = 't', long = "target", help = "target")]
    pub target: Option<Target>,
    #[clap(short = 'b', long = "benchmark", help = "benchmark")]
    pub benchmark: String,

    #[clap(trailing_var_arg = true)]
    pub args: Vec<String>,
}

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
    All(All),
    Run(Run),
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
                Self::All(_) => "all",
                Self::Run(_) => "run",
            }
        )
    }
}

impl Command {
    #[must_use]
    pub fn targets(&self) -> Box<dyn Iterator<Item = Target>> {
        use strum::IntoEnumIterator;
        match self {
            Command::All(_) => Box::new(Target::iter()),
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
            Command::Run(Run { target, .. }) => {
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

    #[clap(
        long = "baseline",
        help = "only run baseline benchmarks (serial, default configuration)"
    )]
    pub baseline: bool,

    #[clap(
        long = "parallel",
        help = "run parallel baseline benchmarks (default configuration)"
    )]
    pub parallel: bool,

    #[clap(subcommand)]
    pub command: Command,
}

impl Options {
    pub fn benchmark_file_path(&self) -> PathBuf {
        let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
        let default_benches_file_path = manifest_dir.join("../test-apps/test-apps.yml");
        self.benches_file_path
            .clone()
            .unwrap_or(default_benches_file_path)
    }
}
