pub mod query;

use clap::Parser;
use color_eyre::eyre;

#[derive(Parser, Debug, Clone)]
pub enum Command {
    Info,
    Compare,
}

#[derive(Parser, Debug, Clone)]
pub struct Options {
    #[clap(subcommand)]
    pub command: Command,
}

pub fn run(options: &Options) -> eyre::Result<()> {
    todo!("implement cuda queries for devices here");
    // Ok(())
}
