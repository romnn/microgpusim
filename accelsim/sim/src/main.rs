use accelsim::Options;
use accelsim_sim as sim;
use clap::Parser;
use color_eyre::eyre::{self, WrapErr};
use std::io::Write;
use std::time::Instant;

#[tokio::main]
async fn main() -> eyre::Result<()> {
    env_logger::init();
    color_eyre::install()?;

    let mut options = Options::parse();
    options.resolve()?;

    let start = Instant::now();

    log::debug!("options: {:#?}", &options);

    let traces_dir = options
        .traces_dir
        .as_ref()
        .ok_or(eyre::eyre!("missing traces_dir"))?;

    let kernelslist = options
        .kernelslist
        .as_ref()
        .ok_or(eyre::eyre!("missing kernelslist"))?;

    let kernelslist = kernelslist
        .canonicalize()
        .wrap_err_with(|| format!("kernelslist at {} does not exist", kernelslist.display()))?;

    println!(
        "simulating {} [upstream={:?}]",
        kernelslist.display(),
        options.use_upstream
    );

    let stream_output = options.stream_output.unwrap_or(true);
    let use_upstream = options.use_upstream.unwrap_or(sim::has_upstream());
    let extra_args: &[String] = &[];
    let (output, _dur) = sim::simulate_trace(
        &traces_dir,
        &kernelslist,
        &options.sim_config,
        options.timeout,
        extra_args,
        stream_output,
        use_upstream,
    )
    .await?;

    log::info!("simulating took {:?}", start.elapsed());

    let stdout = utils::decode_utf8!(&output.stdout);
    let stderr = utils::decode_utf8!(&output.stderr);
    log::debug!("stdout:\n{}", &stdout);
    log::debug!("stderr:\n{}", &stderr);

    if stdout.is_empty() && stderr.is_empty() {
        eyre::bail!("empty stdout and stderr: parsing omitted");
    }

    // write log
    let log_file_path = options
        .log_file
        .unwrap_or(traces_dir.join("accelsim_log.txt"));

    let mut log_file = utils::fs::open_writable(&log_file_path)?;
    log_file.write_all(stdout.as_bytes())?;

    println!("wrote log to {}", log_file_path.display());

    // let log_reader = std::io::Cursor::new(stdout);
    // let parse_options = accelsim::parser::Options::default();
    // let stats = accelsim::parser::parse_stats(log_reader, &parse_options)?;

    // println!("{:#?}", &stats);

    // let converted: Result<stats::Stats, _> = stats.clone().try_into();
    // println!("{:#?}", &converted);
    // for stat in stats.iter() {
    //     println!("{}", &stat);
    // }

    // let stats_file_path = options
    //     .stats_file
    //     .unwrap_or(log_file_path.with_extension("json"));

    // let flat_stats: Vec<_> = stats.into_iter().collect();
    // serde_json::to_writer_pretty(utils::fs::open_writable(stats_file_path)?, &flat_stats)?;
    Ok(())
}
