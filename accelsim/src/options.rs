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
    pub config_dir: Option<PathBuf>,
    #[clap(long = "config", help = "config file")]
    pub config: Option<PathBuf>,
    #[clap(long = "trace-config", help = "trace config file")]
    pub trace_config: Option<PathBuf>,
    #[clap(long = "inter-config", help = "interconnect config file")]
    pub inter_config: Option<PathBuf>,
}

impl Options {
    #[must_use]
    pub fn kernelslist(&self) -> PathBuf {
        self.traces_dir.join("kernelslist.g")
    }
}

impl SimConfig {
    #[must_use]
    pub fn config(&self) -> Option<PathBuf> {
        match (&self.config, &self.config_dir) {
            (None, None) => None,
            (Some(config), _) => Some(config.clone()),
            (None, Some(config_dir)) => Some(config_dir.join("gpgpusim.config")),
        }
        // self.config
        //     .as_ref()
        //     .or(self
        //         .config_dir
        //         .map(|config_dir| config_dir.join("gpgpusim.config"))
        //         .as_ref())
        //     .map(PathBuf::as_path)
        //     .map(std::path::Path::canonicalize)
    }

    // pub fn trace_config(&self) -> Option<Result<PathBuf, std::io::Error>> {
    #[must_use]
    pub fn trace_config(&self) -> Option<PathBuf> {
        match (&self.trace_config, &self.config_dir) {
            (None, None) => None,
            (Some(config), _) => Some(config.clone()),
            (None, Some(config_dir)) => Some(config_dir.join("gpgpusim.trace.config")),
        }
        // self.trace_config
        //     .as_ref()
        //     .unwrap_or(&self.config_dir.join("gpgpusim.trace.config"))
        //     .canonicalize()
    }
}
