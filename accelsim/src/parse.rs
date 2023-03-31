use anyhow::Result;
use clap::Parser;
use std::time::Instant;

fn main() -> Result<()> {
    let start = Instant::now();
    let options = accelsim::parser::Options::parse();
    println!("options: {:#?}", &options);

    let stats = accelsim::parser::parse(options);

    println!("stats: {:#?}", &stats);
    println!("done after {:?}", start.elapsed());
    Ok(())
}
