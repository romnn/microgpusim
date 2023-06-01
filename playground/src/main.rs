use clap::Parser;
use color_eyre::eyre;
use playground::bridge::main::{accelsim, accelsim_config};

#[derive(Parser, Debug)]
struct Options {
    // #[clap(
    //     // long = "traces-dir",
    //     help = "directory containing accelsim traces (kernelslist.g)"
    // )]
    // traces_dir: PathBuf,
    //
    // #[clap(flatten)]
    // sim_config: SimConfig,
    //
    // #[clap(long = "log-file", help = "write simuation output to log file")]
    // log_file: Option<PathBuf>,
    //
    // #[clap(long = "stats-file", help = "parse simulation stats into csv file")]
    // stats_file: Option<PathBuf>,
    //
    // #[clap(
    //     long = "timeout",
    //     help = "timeout",
    //     value_parser = parse_duration_string,
    // )]
    // timeout: Option<Duration>,
}

fn main() -> eyre::Result<()> {
    let config = accelsim_config { test: 0 };
    let ret_code = accelsim(config);
    if ret_code == 0 {
        Ok(())
    } else {
        Err(eyre::eyre!("accelsim exited with code {}", ret_code))
    }
}
