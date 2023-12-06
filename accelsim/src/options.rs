use clap::Parser;
use color_eyre::eyre;
use std::path::{Path, PathBuf};
use std::time::Duration;

fn parse_duration_string(duration: &str) -> eyre::Result<Duration> {
    let res = duration_string::DurationString::from_string(duration.into())
        .map_err(|_| eyre::eyre!("invalid duration string {}", duration))?;
    Ok(res.into())
}

#[derive(Parser, Debug)]
pub struct Options {
    #[clap(help = "directory containing accelsim traces (kernelslist.g)")]
    pub traces_dir: Option<PathBuf>,

    #[clap(flatten)]
    pub sim_config: SimConfig,

    #[clap(long = "kernels", help = "path to kernelslist.g file")]
    pub kernelslist: Option<PathBuf>,

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

    #[clap(long = "progress", aliases = ["follow"], help = "stream simulation output")]
    pub stream_output: Option<bool>,

    #[clap(
        long = "upstream",
        help = "use upstream accelsim implementation (unmodified)"
    )]
    pub use_upstream: Option<bool>,

    #[clap(long = "cores-per-cluster", help = "cores per cluster")]
    pub cores_per_cluster: Option<usize>,

    #[clap(long = "num-clusters", help = "number of clusters")]
    pub num_clusters: Option<usize>,

    #[clap(long = "fill-l2", help = "fill l2 cache on CUDA memcopies")]
    pub fill_l2: Option<bool>,
}

#[derive(Parser, Debug, Default)]
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

fn missing_parent(path: &Path) -> eyre::Report {
    eyre::eyre!("{} missing parent", path.display())
}

impl Options {
    pub fn resolve(&mut self) -> eyre::Result<()> {
        match (&mut self.traces_dir, &mut self.kernelslist) {
            (Some(_), Some(_)) => {
                // fine
            }
            (Some(ref mut traces_dir), None) if traces_dir.is_file() => {
                // assume traces dir is the kernelslist
                let _ = self.kernelslist.insert(traces_dir.clone());
                *traces_dir = traces_dir
                    .parent()
                    .ok_or(missing_parent(traces_dir))?
                    .to_path_buf();
            }
            (Some(traces_dir), None) => {
                // assume default location for kernelslist
                let _ = self.kernelslist.insert(traces_dir.join("kernelslist.g"));
            }
            (None, Some(kernelslist)) => {
                // assume trace dir is parent of kernelslist
                let _ = self.traces_dir.insert(
                    kernelslist
                        .parent()
                        .ok_or(missing_parent(kernelslist))?
                        .to_path_buf(),
                );
            }
            (None, None) => {
                eyre::bail!("must specify either trace dir or kernelslist");
            }
        }
        Ok(())
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
    }

    #[must_use]
    pub fn trace_config(&self) -> Option<PathBuf> {
        match (&self.trace_config, &self.config_dir) {
            (None, None) => None,
            (Some(config), _) => Some(config.clone()),
            (None, Some(config_dir)) => Some(config_dir.join("gpgpusim.trace.config")),
        }
    }
}
