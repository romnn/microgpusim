use clap::Parser;
use color_eyre::eyre;
use itertools::Itertools;
use std::{
    ffi::OsStr,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub enum Format<'a> {
    JSON(&'a str),
    YAML(&'a str),
}

impl<'a> AsRef<str> for Format<'a> {
    fn as_ref(&self) -> &'a str {
        match self {
            Self::JSON(inner) | Self::YAML(inner) => *inner,
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
}

#[derive(Parser, Debug, Clone)]
pub struct Options {
    #[clap(subcommand)]
    pub command: Command,
}

pub fn run(options: Options) -> eyre::Result<()> {
    match options.command {
        Command::ConvertConfig {
            configs,
            mut output,
        } => {
            if configs.is_empty() {
                return Ok(());
            }
            let config: Vec<String> = configs
                .iter()
                .map(|config_path| std::fs::read_to_string(config_path))
                .try_collect()?;

            let config = casimu::config::accelsim::Config::parse(config.join("\n"))?;
            let extension = output
                .extension()
                .and_then(OsStr::to_str)
                .map(str::to_ascii_lowercase);

            let format = match extension.as_deref() {
                None => Format::YAML("yaml"),
                Some(ext @ "json") => Format::JSON(ext),
                Some(ext @ ("yaml" | "yml")) => Format::YAML(ext),
                Some(other) => eyre::bail!(
                    "output path {} has invalid extension {:?}. valid: [.yaml|.json|.yml]",
                    output.display(),
                    other
                ),
            };

            let output = output.with_extension(format.as_ref());

            let mut writer = utils::fs::open_writable(&output)?;

            match format {
                Format::YAML(_) => serde_yaml::to_writer(&mut writer, &config)?,
                Format::JSON(_) => serde_json::to_writer_pretty(&mut writer, &config)?,
            }
            println!("wrote config to {}", output.display());
        }
    }
    Ok(())
}
