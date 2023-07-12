use clap::Parser;
use color_eyre::eyre;
use std::time::Instant;

fn main() -> eyre::Result<()> {
    color_eyre::install()?;

    let start = Instant::now();
    let options = accelsim::parser::Options::parse();
    println!("options: {:#?}", &options);

    let stats = accelsim::parser::parse(&options)?;

    println!("stats: {:#?}", &stats);
    println!("done after {:?}", start.elapsed());
    Ok(())
}
