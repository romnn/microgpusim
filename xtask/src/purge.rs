use clap::Parser;
use color_eyre::eyre::{self, WrapErr};
use console::style;
use std::path::PathBuf;

#[derive(Parser, Debug, Clone)]
pub enum Command {
    JSON,
}

#[derive(Parser, Debug, Clone)]
pub struct Options {
    #[clap(short = 'p', long = "path", help = "path to directory to purge")]
    pub path: PathBuf,

    #[clap(subcommand)]
    pub command: Command,
}

fn purge_json_kernel_files(options: &Options) -> eyre::Result<()> {
    let match_options = glob::MatchOptions {
        case_sensitive: false,
        require_literal_separator: false,
        require_literal_leading_dot: false,
    };
    let pattern = options
        .path
        .join("**/*kernel-*.json")
        .to_string_lossy()
        .to_string();
    println!("glob pattern: {}", pattern);
    let mut removed = 0;
    for entry in glob::glob_with(&pattern, match_options)? {
        let path = entry?;
        std::fs::remove_file(&path)
            .wrap_err_with(|| eyre::eyre!("failed to remove file {}", path.display()))?;
        println!("{}", style(format!("removed {}", path.display())).red());
        removed += 1;
    }
    println!("removed {removed} files");
    Ok(())
}

pub fn run(options: &Options) -> eyre::Result<()> {
    match options.command {
        Command::JSON => {
            purge_json_kernel_files(options)?;
        }
    }
    Ok(())
}
