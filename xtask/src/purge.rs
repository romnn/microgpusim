use clap::Parser;
use color_eyre::eyre::{self, WrapErr};
use console::style;
use std::path::PathBuf;

#[derive(Parser, Debug, Clone)]
pub enum Command {
    JSON,
    PerTargetDirs,
}

#[derive(Parser, Debug, Clone)]
pub struct Options {
    #[clap(short = 'p', long = "path", help = "path to directory to purge")]
    pub path: PathBuf,

    #[clap(subcommand)]
    pub command: Command,
}

fn purge_deprecated_per_target_dirs(options: &Options) -> eyre::Result<()> {
    if options.path.file_name().unwrap() != "results" {
        panic!("warning");
    }

    let _valid_per_target_dirs = [
        "sim",
        "playground-sim",
        "accelsim-sim",
        "trace",
        "accelsim-trace",
        "profile",
    ];

    let deprecated_per_target_dirs = ["accelsim_trace"];

    let per_target_dirs = walkdir::WalkDir::new(&options.path)
        .min_depth(3)
        .max_depth(3)
        .into_iter()
        .filter_map(|entry| entry.ok());

    let mut seen: indexmap::IndexSet<(String, bool)> = indexmap::IndexSet::new();
    let mut removed = 0;
    for entry in per_target_dirs {
        let metadata = entry.metadata()?;
        let path = entry.path();
        let is_dir = metadata.is_dir();
        let dir_name = path
            .file_name()
            .ok_or_else(|| eyre::eyre!("path {} has no final directory name", path.display()))?
            .to_string_lossy()
            .to_string();

        log::debug!("checking {} (is_dir={})", path.display(), is_dir);
        if deprecated_per_target_dirs.contains(&dir_name.as_str()) {
            if is_dir {
                utils::fs::remove_dir(&path)?;
            } else {
                std::fs::remove_file(&path)
                    .wrap_err_with(|| eyre::eyre!("failed to remove file {}", path.display()))?;
            }
            println!("{}", style(format!("removed {}", path.display())).red());
            removed += 1;
        } else {
            seen.insert((dir_name.clone(), is_dir));
        }
    }
    for (dir, is_dir) in seen {
        println!("kept: {:>25} (directory={})", dir, is_dir);
    }
    println!("removed {removed} files");
    Ok(())
}

fn purge_json_kernel_files(options: &Options) -> eyre::Result<()> {
    if options.path.file_name().unwrap() != "results" {
        panic!("warning");
    }

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
        Command::PerTargetDirs => {
            purge_deprecated_per_target_dirs(options)?;
        }
        Command::JSON => {
            purge_json_kernel_files(options)?;
        }
    }
    Ok(())
}
