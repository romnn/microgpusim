use clap::Parser;
use color_eyre::eyre;
use std::path::PathBuf;
use std::time::Duration;

fn parse_duration_string(duration: &str) -> eyre::Result<Duration> {
    let res = duration_string::DurationString::from_string(duration.into())
        .map_err(|_| eyre::eyre!("invalid duration string {}", duration))?;
    Ok(res.into())
}

#[derive(Parser, Debug)]
pub struct Options {
    #[clap(help = "directory containing accelsim traces (kernelslist.g)")]
    pub traces_dir: PathBuf,

    #[clap(flatten)]
    pub sim_config: SimConfig,

    #[clap(long = "log-file", help = "write simuation output to log file")]
    pub log_file: Option<PathBuf>,

    #[clap(long = "stats-file", help = "parse simulation stats into csv file")]
    pub stats_file: Option<PathBuf>,

    #[clap(
        long = "timeout",
        help = "timeout",
        value_parser = parse_duration_string,
    )]
    pub timeout: Option<Duration>,
}

#[derive(Parser, Debug)]
pub struct SimConfig {
    #[clap(help = "config directory")]
    pub config_dir: PathBuf,
    #[clap(long = "config", help = "config file")]
    pub config: Option<PathBuf>,
    #[clap(long = "trace-config", help = "trace config file")]
    pub trace_config: Option<PathBuf>,
    #[clap(long = "inter-config", help = "interconnect config file")]
    pub inter_config: Option<PathBuf>,
}

impl Options {
    pub fn kernelslist(&self) -> Result<PathBuf, std::io::Error> {
        self.traces_dir.join("kernelslist.g").canonicalize()
    }
}

impl SimConfig {
    pub fn config(&self) -> Result<PathBuf, std::io::Error> {
        self.config
            .as_ref()
            .unwrap_or(&self.config_dir.join("gpgpusim.config"))
            .canonicalize()
    }

    pub fn trace_config(&self) -> Result<PathBuf, std::io::Error> {
        self.trace_config
            .as_ref()
            .unwrap_or(&self.config_dir.join("gpgpusim.trace.config"))
            .canonicalize()
    }
}
