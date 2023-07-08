#![allow(warnings)]

mod progress;

#[cfg(feature = "remote")]
mod remote;

use clap::Parser;
use color_eyre::eyre::{self, WrapErr};
use color_eyre::owo_colors::colors::css::Indigo;
use indicatif::ProgressBar;
use progress::ProgressStyle;
use validate::{Benchmark, Benchmarks, Input};
// use std::fs::{self, OpenOptions};
// use std::io::{BufWriter, Write};
// use std::os::unix::fs::DirBuilderExt;
use console::style;
use futures::stream::{self, StreamExt};
use futures::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;

// fn open_writable(path: &Path) -> eyre::Result<BufWriter<fs::File>, std::io::Error> {
//     let file = OpenOptions::new()
//         .create(true)
//         .write(true)
//         .truncate(true)
//         .open(path)?;
//     Ok(BufWriter::new(file))
// }
//

async fn profile_executable(
    exec: &Path,
    exec_args: &Vec<&String>,
    traces_dir: &Path,
) -> eyre::Result<()> {
    let profiling_results = profile::nvprof::nvprof(exec, exec_args).await?;
    // let writer = open_writable(&traces_dir.join("nvprof.json"))?;
    // serde_json::to_writer_pretty(writer, &profiling_results.metrics)?;
    // let mut writer = open_writable(&traces_dir.join("nvprof.log"))?;
    // writer.write_all(profiling_results.raw.as_bytes())?;
    Ok(())
}

// async fn trace_exec(exec: &Path, exec_args: &Vec<&String>, traces_dir: &Path) -> eyre::Result<()> {
//     invoke_trace::trace(exec, exec_args, traces_dir).await?;
//     Ok(())
// }

#[derive(Parser, Debug, Clone)]
pub struct BuildOptions {}

#[derive(Parser, Debug, Clone)]
pub struct ProfileOptions {}

#[derive(Parser, Debug, Clone)]
pub struct Options {
    #[clap(short = 'b', long = "benches", help = "path to benchmarks yaml file")]
    pub benchmarks_path: Option<PathBuf>,

    #[clap(long = "force", help = "force re-run", default_value = "false")]
    force: bool,

    #[clap(
        short = 'c',
        long = "concurrency",
        help = "number of benchmarks to run concurrently"
    )]
    pub concurrency: Option<usize>,

    #[clap(subcommand)]
    pub command: Command,
}

#[derive(Parser, Debug, Clone)]
pub enum Command {
    Profile(ProfileOptions),
    Trace(ProfileOptions),
    Simulate(ProfileOptions),
    Build(BuildOptions),
}

async fn run_bechmark(
    name: &String,
    bench: &validate::Benchmark,
    input: Input,
    options: &Options,
    bar: &ProgressBar,
) -> eyre::Result<()> {
    bar.set_message(name.clone());
    let input = input?;
    let exec = bench.executable();

    let res: eyre::Result<()> = match options.command {
        Command::Profile(ref opts) => {
            let results = profile::nvprof::nvprof(exec, input).await?;
            dbg!(&results);
            // let traces_dir = exec_dir.join("traces").join(format!(
            //     "{}-trace",
            //     &trace_model::app_prefix(option_env!("CARGO_BIN_NAME"))
            // ));

            Ok(())
        }
        Command::Trace(ref opts) => {
            // invoke_trace::trace(exec, exec_args, traces_dir).await?;
            Ok(())
        }
        Command::Simulate(ref opts) => Ok(()),
        Command::Build(ref opts) => Ok(()),
    };

    res
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    color_eyre::install()?;

    let start = std::time::Instant::now();

    // load env variables from .env files
    dotenv::dotenv().ok();

    let options = Arc::new(Options::parse());

    // parse benchmarks
    let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
    let default_benchmarks_file = manifest_dir.join("../test-apps/test-apps.yml");
    let benchmarks_file = options
        .benchmarks_path
        .as_ref()
        .unwrap_or(&default_benchmarks_file);
    let benchmarks_file = benchmarks_file
        .canonicalize()
        .wrap_err_with(|| format!("{} does not exist", benchmarks_file.display()))?;
    let benchmarks = Benchmarks::from(&benchmarks_file)?;

    let benchmark_concurrency = match options.command {
        Command::Profile(_) => benchmarks.profile_config.common.concurrency,
        Command::Trace(_) => benchmarks.trace_config.common.concurrency,
        Command::Simulate(_) => benchmarks.sim_config.common.concurrency,
        Command::Build(_) => Some(1),
    };

    let concurrency = options
        .concurrency
        .or(benchmark_concurrency)
        .unwrap_or_else(|| num_cpus::get_physical());
    println!("concurrency: {}", &concurrency);

    // get benchmark configurations
    let enabled_benches: Vec<_> = benchmarks.enabled_benchmark_configurations().collect();
    let num_bench_configs = enabled_benches.len();

    // create progress bar
    let bar = Arc::new(ProgressBar::new(enabled_benches.len() as u64));
    let bar_style = ProgressStyle::default();
    bar.set_style(bar_style.into());

    let results: Vec<_> = stream::iter(enabled_benches.into_iter())
        .map(|(name, bench, input)| {
            let name = name.clone();
            let options = options.clone();
            let bar = bar.clone();
            async move {
                let input_args = input.as_ref().cloned().unwrap_or_default();
                let res = run_bechmark(&name, &bench, input, &options, &bar).await;
                bar.inc(1);

                let op = match options.command {
                    Command::Profile(_) => "profiling",
                    Command::Trace(_) => "tracing",
                    Command::Simulate(_) => "simulating",
                    Command::Build(_) => "building",
                };
                bar.println(format!(
                    "{:>15} {:>20} [ {} {} ] {}",
                    op,
                    if res.is_ok() {
                        style(name).green()
                    } else {
                        style(name).red()
                    },
                    bench.executable().display(),
                    &input_args.join(" "),
                    if res.is_ok() {
                        style("succeeded")
                    } else {
                        style("failed").red()
                        // style(format!("failed: {:?}", res)).red()
                    },
                ));

                res
            }
        })
        .buffer_unordered(concurrency)
        .collect()
        .await;
    bar.finish();

    let (succeeded, failed): (Vec<_>, Vec<_>) = utils::partition_results(results);
    assert_eq!(num_bench_configs, succeeded.len() + failed.len());

    let failed_msg = format!("{} failed", failed.len());
    println!(
        "\n\n => ran {} benchmarks + input configurations in {:?}: {}",
        num_bench_configs,
        start.elapsed(),
        if failed.is_empty() {
            style(failed_msg)
        } else {
            style(failed_msg).red()
        },
    );

    // dbg!(&failed);

    // let args: Vec<_> = std::env::args().collect();
    // let exec = PathBuf::from(args.get(1).expect("usage: ./validate <executable> [args]"));
    // let exec_args = args.iter().skip(2).collect::<Vec<_>>();
    //
    // let exec_dir = exec.parent().expect("executable has no parent dir");
    // let traces_dir = exec_dir.join("traces").join(format!(
    //     "{}-trace",
    //     &trace_model::app_prefix(option_env!("CARGO_BIN_NAME"))
    // ));
    //
    // #[cfg(feature = "remote")]
    // remote::connect().await?;
    //
    // fs::DirBuilder::new()
    //     .recursive(true)
    //     .mode(0o777)
    //     .create(&traces_dir)?;
    //
    // profile_exec(&exec, &exec_args, &traces_dir).await?;
    // trace_exec(&exec, &exec_args, &traces_dir).await?;

    Ok(())
}
