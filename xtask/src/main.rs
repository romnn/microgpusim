mod coverage;
mod docs;
mod format;
mod util;

use clap::Parser;
use color_eyre::eyre;

#[derive(Parser, Debug, Clone)]
pub enum Command {
    Coverage(coverage::Options),
    Format(format::Options),
    Docs,
}

#[derive(Parser, Debug, Clone)]
pub struct Options {
    #[clap(subcommand)]
    pub command: Command,
}

fn main() -> eyre::Result<()> {
    color_eyre::install()?;
    let options = Options::parse();
    dbg!(&options);
    match options.command {
        Command::Coverage(opts) => coverage::coverage(opts),
        Command::Format(opts) => format::format(opts),
        Command::Docs => docs::docs(),
    }
}
