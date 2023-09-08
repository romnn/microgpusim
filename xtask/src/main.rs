mod accelsim;
mod coverage;
mod docs;
mod format;
mod trace;
mod util;

use clap::Parser;
use color_eyre::eyre;

#[derive(Parser, Debug, Clone)]
pub enum Command {
    Coverage(coverage::Options),
    Format(format::Options),
    Accelsim(self::accelsim::Options),
    Trace(trace::Options),
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
        Command::Coverage(ref opts) => coverage::coverage(opts),
        Command::Format(opts) => format::format(opts),
        Command::Accelsim(opts) => accelsim::run(opts),
        Command::Trace(opts) => trace::run(&opts),
        Command::Docs => docs::docs(),
    }
}
