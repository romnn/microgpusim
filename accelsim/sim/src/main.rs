#![allow(warnings)]

use accelsim::parser::{parse, Options as ParseOptions};
use accelsim::{Options, SimConfig};
use accelsim_sim as sim;
use clap::Parser;
use color_eyre::eyre::{self, WrapErr};
use std::time::{Duration, Instant};

// todo add an output dir

#[tokio::main]
async fn main() -> eyre::Result<()> {
    color_eyre::install()?;

    let options = Options::parse();

    let start = Instant::now();

    dbg!(&options.traces_dir);
    dbg!(&options.sim_config);
    dbg!(&options.timeout);

    let output =
        sim::simulate_trace(&options.traces_dir, options.sim_config, options.timeout).await?;

    println!("simulating took {:?}", start.elapsed());

    // write log
    let stdout = utils::decode_utf8!(&output.stdout);
    let stderr = utils::decode_utf8!(&output.stderr);
    println!("stdout:\n{}", &stdout);
    eprintln!("stderr:\n{}", &stderr);

    if stdout.is_empty() && stderr.is_empty() {
        eyre::bail!("empty stdout and stderr: parsing omitted");
    }

    let log_file_path = options
        .log_file
        .unwrap_or(options.traces_dir.join("accelsim_log.txt"));
    {
        use std::io::Write;
        let mut log_file = utils::fs::open_writable(&log_file_path)?;
        log_file.write_all(stdout.as_bytes())?;
    }

    // parse stats
    let stats_file_path = options
        .stats_file
        .unwrap_or(log_file_path.with_extension("csv"));
    let mut parse_options = ParseOptions::new(log_file_path);
    parse_options.save_to(stats_file_path);
    let stats = parse(&parse_options)?;

    let mut preview: Vec<_> = stats
        .iter()
        .map(|(idx, val)| (format!("{} / {} / {}", idx.0, idx.1, idx.2), val))
        .collect();
    preview.sort_by(|a, b| a.0.cmp(&b.0));

    for (key, val) in preview {
        println!(" => {key}: {val}");
    }
    Ok(())
}
