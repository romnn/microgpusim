use clap::Parser;
use color_eyre::eyre::{self, WrapErr};
use console::style;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug, Clone)]
pub struct PurgeDeprecatedConfigurationOptions {
    #[clap(long = "benches", help = "path to materialized benchmark file")]
    pub materialized_path: Option<PathBuf>,

    #[clap(long = "results", help = "path to results")]
    pub results_path: Option<PathBuf>,

    // #[clap(long = "target", help = "benchmark target")]
    // pub target: Option<validate::Target>,
    #[clap(short = 'b', long = "benchmark", help = "benchmark")]
    pub benchmark: Option<String>,
}

#[derive(Parser, Debug, Clone)]
pub struct PurgePerTargetDirsOptions {
    #[clap(short = 'p', long = "path", help = "path to directory to purge")]
    pub path: PathBuf,
}

#[derive(Parser, Debug, Clone)]
pub struct PurgeJSONOptions {
    #[clap(short = 'p', long = "path", help = "path to directory to purge")]
    pub path: PathBuf,
}

#[derive(Parser, Debug, Clone)]
pub enum PurgeCommand {
    JSONTraces(PurgeJSONOptions),
    PerTargetDirs(PurgePerTargetDirsOptions),
    DeprecatedConfigs(PurgeDeprecatedConfigurationOptions),
}

#[derive(Parser, Debug, Clone)]
pub struct Options {
    #[clap(subcommand)]
    pub command: PurgeCommand,
}

fn purge_deprecated_per_target_dirs(
    _options: &Options,
    purge_options: &PurgePerTargetDirsOptions,
) -> eyre::Result<()> {
    if purge_options.path.file_name().unwrap() != "results" {
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

    let per_target_dirs = walkdir::WalkDir::new(&purge_options.path)
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

fn purge_json_kernel_files(
    _options: &Options,
    purge_options: &PurgeJSONOptions,
) -> eyre::Result<()> {
    if purge_options.path.file_name().unwrap() != "results" {
        panic!("must run on results dir");
    }

    let match_options = glob::MatchOptions {
        case_sensitive: false,
        require_literal_separator: false,
        require_literal_leading_dot: false,
    };
    let pattern = purge_options
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

fn purge_deprecated_configs(
    _options: &Options,
    purge_options: &PurgeDeprecatedConfigurationOptions,
) -> eyre::Result<()> {
    use std::collections::HashSet;
    // use validate::materialized::TargetBenchmarkConfig;
    let materialized_path = purge_options.materialized_path.clone().unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../test-apps/test-apps-materialized.yml")
    });
    let reader = utils::fs::open_readable(materialized_path)?;
    let benches: validate::materialized::Benchmarks = serde_yaml::from_reader(reader)?;

    let mut valid_result_dirs: HashSet<&Path> = HashSet::new();
    for bench_config in benches.benchmark_configs() {
        match purge_options.benchmark {
            Some(ref benchmark) if bench_config.name.to_lowercase() != benchmark.to_lowercase() => {
                continue
            }
            _ => {}
        }

        // if let Some(target) = purge_options.benchmark {
        //     if  {
        //         continue;
        //     }
        // }

        // if let Some(target) = purge_options.target {
        //     if bench_config.target != target {
        //         continue;
        //     }
        // }
        dbg!(&bench_config);
        valid_result_dirs.insert(&bench_config.results_dir);
        // valid_result_dirs.extend(match bench_config.target_config {
        //     TargetBenchmarkConfig::Profile {
        //         ref profile_dir, ..
        //     } => vec![profile_dir],
        //     TargetBenchmarkConfig::Trace { ref traces_dir, .. } => vec![traces_dir],
        //     TargetBenchmarkConfig::AccelsimTrace { ref traces_dir, .. } => vec![traces_dir],
        //     TargetBenchmarkConfig::Simulate { ref stats_dir, .. } => vec![stats_dir],
        //     TargetBenchmarkConfig::ExecDrivenSimulate { ref stats_dir, .. } => vec![stats_dir],
        //     TargetBenchmarkConfig::AccelsimSimulate { ref stats_dir, .. } => vec![stats_dir],
        //     TargetBenchmarkConfig::PlaygroundSimulate { ref stats_dir, .. } => vec![stats_dir],
        // });
        // results_dir: "/home/roman/dev/box/results",
        // break;
    }

    let results_path = purge_options
        .results_path
        .as_deref()
        .unwrap_or(benches.config.results_dir.as_ref());

    let all_result_dirs = if let Some(ref bench) = purge_options.benchmark {
        let benchmark = benches
            .benchmarks
            .values()
            .flat_map(|target_benches| target_benches.keys())
            .find(|name| bench.to_lowercase() == name.to_lowercase())
            .unwrap_or(bench);
        walkdir::WalkDir::new(results_path.join(benchmark))
            .min_depth(1)
            .max_depth(1)
    } else {
        walkdir::WalkDir::new(results_path)
            .min_depth(2)
            .max_depth(2)
    };

    let all_result_dirs_vec: Vec<walkdir::DirEntry> = all_result_dirs
        .into_iter()
        .filter_map(|entry| entry.ok())
        .collect();

    let all_result_dirs: HashSet<&Path> = all_result_dirs_vec
        .iter()
        .map(|entry| entry.path())
        .collect();

    // let mut removed = 0;
    // for entry in per_target_dirs {
    //     let metadata = entry.metadata()?;
    //     let path = entry.path();
    //     let is_dir = metadata.is_dir();
    //     let dir_name = path
    //         .file_name()
    //         .ok_or_else(|| eyre::eyre!("path {} has no final directory name", path.display()))?
    //         .to_string_lossy()
    //         .to_string();
    //
    //     log::debug!("checking {} (is_dir={})", path.display(), is_dir);
    //     if deprecated_per_target_dirs.contains(&dir_name.as_str()) {
    //         if is_dir {
    //             utils::fs::remove_dir(&path)?;
    //         } else {
    //             std::fs::remove_file(&path)
    //                 .wrap_err_with(|| eyre::eyre!("failed to remove file {}", path.display()))?;
    //         }
    //         println!("{}", style(format!("removed {}", path.display())).red());
    //         removed += 1;
    //     } else {
    //         seen.insert((dir_name.clone(), is_dir));
    //     }
    // }

    // dbg!(&all_result_dirs);
    // dbg!(&valid_result_dirs);

    let deprecated: Vec<_> = all_result_dirs.difference(&valid_result_dirs).collect();
    let remaining: Vec<_> = all_result_dirs.intersection(&valid_result_dirs).collect();
    for path in &remaining {
        println!(
            "{}",
            style(format!("keeping:    {}", path.display())).green()
        );
    }
    for path in &deprecated {
        println!("{}", style(format!("deprecated: {}", path.display())).red());
    }

    println!("");
    println!(
        "have {}: {} and {} results.",
        style(format!("{} valid results", valid_result_dirs.len())).cyan(),
        style(format!("{} deprecated", deprecated.len())).red(),
        style(format!("{} remaining", remaining.len())).green(),
    );

    if deprecated.is_empty() {
        println!("{}", style("no deprecated files!").green());
        return Ok(());
    }

    for i in 0..3 {
        let confirmation = dialoguer::Confirm::new()
            .with_prompt(format!(
                "Do you want to remove the deprecated files {}/{}?",
                i + 1,
                3
            ))
            .interact()
            .unwrap();
        if !confirmation {
            println!("abort.");
            return Ok(());
        }
    }

    for path in &deprecated {
        utils::fs::remove_dir(&path)?;
        println!("{}", style(format!("removed {}", path.display())).red());
    }
    Ok(())
}

pub fn run(options: &Options) -> eyre::Result<()> {
    match options.command {
        PurgeCommand::DeprecatedConfigs(ref opts) => {
            purge_deprecated_configs(options, opts)?;
        }
        PurgeCommand::PerTargetDirs(ref opts) => {
            purge_deprecated_per_target_dirs(options, opts)?;
        }
        PurgeCommand::JSONTraces(ref opts) => {
            purge_json_kernel_files(options, opts)?;
        }
    }
    Ok(())
}
