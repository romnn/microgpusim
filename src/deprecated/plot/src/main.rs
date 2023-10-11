use clap::Parser;
use color_eyre::eyre;
use serde::Deserializer;
use std::fs::{File, OpenOptions};
use std::io::BufReader;
use std::path::{Path, PathBuf};

#[derive(Debug, Parser)]
struct Options {
    trace: PathBuf,
    allocations: PathBuf,
    output: PathBuf,
}

fn parse_allocations(path: impl AsRef<Path>) -> eyre::Result<Vec<trace_model::MemAllocation>> {
    let file = OpenOptions::new().read(true).open(path.as_ref())?;
    let reader = BufReader::new(file);
    let allocations = serde_json::from_reader(reader)?;
    Ok(allocations)
}

fn read_trace(
    path: impl AsRef<Path>,
) -> eyre::Result<rmp_serde::Deserializer<rmp_serde::decode::ReadReader<BufReader<File>>>> {
    let file = OpenOptions::new().read(true).open(path.as_ref())?;
    let reader = BufReader::new(file);
    let reader = rmp_serde::Deserializer::new(reader);
    Ok(reader)
}

fn main() -> eyre::Result<()> {
    color_eyre::install()?;

    let options = Options::parse();
    dbg!(&options);

    let allocations = parse_allocations(&options.allocations)?;

    let mut access_plot = plot::MemoryAccesses::default();

    for allocation in &allocations {
        access_plot.register_allocation(allocation);
    }

    let mut reader = read_trace(options.trace)?;
    let decoder = nvbit_io::Decoder::new(|access: trace_model::MemAccessTraceEntry| {
        // dbg!(&access);
        access_plot.add(access, None);
    });
    reader.deserialize_seq(decoder)?;

    println!("drawing to {}", options.output.display());
    // dbg!(std::env::current_dir());
    // dbg!(options.output.canonicalize());
    access_plot.draw(&options.output)?;
    Ok(())
}
