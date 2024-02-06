use clap::Parser;
use color_eyre::eyre;
use itertools::Itertools;
use std::{ffi::OsStr, path::PathBuf};
use utils::diff;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub enum Format<'a> {
    Json(&'a str),
    Yaml(&'a str),
}

impl<'a> AsRef<str> for Format<'a> {
    fn as_ref(&self) -> &'a str {
        match self {
            Self::Json(inner) | Self::Yaml(inner) => inner,
        }
    }
}

#[derive(Parser, Debug, Clone)]
pub enum Command {
    ConvertConfig {
        #[clap(short = 'c', long = "config", help = "path to accelsim config file")]
        configs: Vec<PathBuf>,
        #[clap(short = 'o', long = "output", help = "converted output file path")]
        output: PathBuf,
    },
    CompareConfig {
        // #[arg(trailing_var_arg = true, allow_hyphen_values = true, hide = true)]
        #[clap(help = "path to accelsim config files")]
        configs: Vec<PathBuf>,
    },
}

#[derive(Parser, Debug, Clone)]
pub struct Options {
    #[clap(subcommand)]
    pub command: Command,
}

pub fn run(options: Options) -> eyre::Result<()> {
    use gpucachesim::config::accelsim::Config as AccelsimConfig;
    match options.command {
        Command::CompareConfig { configs } => {
            if configs.is_empty() {
                return Ok(());
            }
            let [left_config_path, right_config_path] = &configs[..] else {
                // if configs.len() != 2 {
                eyre::bail!(
                    "can only compare exactly two configuration files, got {}",
                    configs.len(),
                );
            };
            let left_config = std::fs::read_to_string(&left_config_path)?;
            let right_config = std::fs::read_to_string(&right_config_path)?;
            let left_config = AccelsimConfig::parse(left_config)?;
            let right_config = AccelsimConfig::parse(right_config)?;
            diff::diff!(
                left: left_config,
                right: right_config,
                "{} (left) vs {} (right)",
                left_config_path.to_string_lossy().to_string(),
                right_config_path.to_string_lossy().to_string(),
            );
        }
        Command::ConvertConfig { configs, output } => {
            if configs.is_empty() {
                return Ok(());
            }
            let configs: Vec<String> = configs.iter().map(std::fs::read_to_string).try_collect()?;
            let concatenated_config = configs.join("\n");
            let config = AccelsimConfig::parse(concatenated_config)?;
            let extension = output
                .extension()
                .and_then(OsStr::to_str)
                .map(str::to_ascii_lowercase);

            let format = match extension.as_deref() {
                None => Format::Yaml("yaml"),
                Some(ext @ "json") => Format::Json(ext),
                Some(ext @ ("yaml" | "yml")) => Format::Yaml(ext),
                Some(other) => eyre::bail!(
                    "output path {} has invalid extension {:?}. valid: [.yaml|.json|.yml]",
                    output.display(),
                    other
                ),
            };

            let output = output.with_extension(format.as_ref());

            let mut writer = utils::fs::open_writable(&output)?;

            match format {
                Format::Yaml(_) => serde_yaml::to_writer(&mut writer, &config)?,
                Format::Json(_) => serde_json::to_writer_pretty(&mut writer, &config)?,
            }
            println!("wrote config to {}", output.display());
        }
    }
    Ok(())
}
