use color_eyre::eyre;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser as ClapParser;
use pest::Parser as PestParser;
// use ptx::parser::{Parser as PTXParser, Rule};

#[derive(ClapParser, Debug, Clone)]
pub struct ParsePTXOptions {
    pub ptx_path: PathBuf,
}

#[derive(ClapParser, Debug, Clone)]
pub enum Command {
    ParsePTX(ParsePTXOptions),
}

#[derive(ClapParser, Debug, Clone)]
pub struct Options {
    #[clap(subcommand)]
    pub command: Command,
}

fn main() -> eyre::Result<()> {
    color_eyre::install()?;
    let options = Options::parse();

    match options.command {
        Command::ParsePTX(ParsePTXOptions { ptx_path }) => {
            let ptx_code = std::fs::read_to_string(&ptx_path)?;
            let code_size_bytes = ptx_code.bytes().len();
            let start = Instant::now();
            // let parsed = PTXParser::parse(Rule::program, &ptx_code)?;
            let dur = start.elapsed();
            let dur_millis = dur.as_millis();
            let dur_secs = dur.as_secs_f64();
            let code_size_mib = code_size_bytes as f64 / (1024.0 * 1024.0);
            let mib_per_sec = code_size_mib / dur_secs;
            println!(
                "parsing {} took {} ms ({:3.3} MiB/s)",
                ptx_path.display(),
                dur_millis,
                mib_per_sec
            );
        }
    }

    Ok(())
}
