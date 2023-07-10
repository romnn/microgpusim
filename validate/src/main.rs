#![allow(warnings)]

mod progress;

#[cfg(feature = "remote")]
mod remote;

use clap::Parser;
use color_eyre::eyre::{self, WrapErr};
use color_eyre::owo_colors::colors::css::Indigo;
use indicatif::ProgressBar;
use progress::ProgressStyle;
use validate::{template, Benchmark, Benchmarks, Config, Input, PathExt};
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

// async fn profile_executable(
//     exec: &Path,
//     exec_args: &Vec<&String>,
//     traces_dir: &Path,
// ) -> eyre::Result<()> {
//     let profiling_results = profile::nvprof::nvprof(exec, exec_args).await?;
//     // let writer = open_writable(&traces_dir.join("nvprof.json"))?;
//     // serde_json::to_writer_pretty(writer, &profiling_results.metrics)?;
//     // let mut writer = open_writable(&traces_dir.join("nvprof.log"))?;
//     // writer.write_all(profiling_results.raw.as_bytes())?;
//     Ok(())
// }

// async fn trace_exec(exec: &Path, exec_args: &Vec<&String>, traces_dir: &Path) -> eyre::Result<()> {
//     invoke_trace::trace(exec, exec_args, traces_dir).await?;
//     Ok(())
// }
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum Format {
    JSON,
    YAML,
}

fn serialize_to_writer(
    format: Format,
    writer: &mut impl std::io::Write,
    data: impl serde::Serialize,
) -> eyre::Result<()> {
    match format {
        Format::JSON => {
            serde_json::to_writer(writer, &data)?;
        }
        Format::YAML => {
            serde_yaml::to_writer(writer, &data)?;
        }
    }
    Ok(())
}

#[derive(Parser, Debug, Clone)]
pub struct BuildOptions {}

#[derive(Parser, Debug, Clone)]
pub struct ProfileOptions {}

#[derive(Parser, Debug, Clone)]
pub struct ExpandOptions {}

#[derive(Parser, Debug, Clone)]
pub struct Options {
    #[clap(short = 'p', long = "path", help = "path to benchmarks yaml file")]
    pub benchmarks_path: Option<PathBuf>,

    #[clap(short = 'b', long = "bench", help = "name of benchmark to run")]
    pub benchmarks: Vec<String>,

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
    Expand(ExpandOptions),
}

async fn run_bechmark(
    name: String,
    bench: &validate::Benchmark,
    input: Result<Input, validate::CallTemplateError>,
    config: &Config,
    options: &Options,
    bar: &ProgressBar,
) -> eyre::Result<()> {
    bar.set_message(name.clone());
    let input = input?;
    let exec = bench.executable();

    let mut tmpl_values = template::Values {
        inputs: input.values,
        // todo: could also just clone the full benchmark and just set a private name?
        bench: template::BenchmarkValues { name: name.clone() },
    };

    let res: eyre::Result<()> = match options.command {
        Command::Profile(ref opts) => {
            let results = profile::nvprof::nvprof(exec, &input.cmd_args).await?;

            // let metrics_file = bench.profile.metrics_file(&tmpl_values);
            let default_log_file = config
                .results_dir
                .join(&name)
                .join(format!("{}-{}", &name, input.cmd_args.join("-")))
                .join("profile.log");
            dbg!(&default_log_file);
            let log_file = bench
                .profile
                .log_file(&tmpl_values)?
                .unwrap_or(default_log_file);
            dbg!(&log_file);
            //
            // let writer = open_writable(&traces_dir.join("nvprof.json"))?;
            // serde_json::to_writer_pretty(writer, &profiling_results.metrics)?;
            // let mut writer = open_writable(&traces_dir.join("nvprof.log"))?;
            // writer.write_all(profiling_results.raw.as_bytes())?;

            // dbg!(&results);
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
        _ => Ok(()),
    };

    res
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    color_eyre::install()?;

    let start = std::time::Instant::now();

    // load env variables from .env files
    dotenv::dotenv().ok();

    let cwd = std::env::current_dir()?;

    let options = Arc::new(Options::parse());
    dbg!(&options);

    // parse benchmarks
    let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
    let default_benchmarks_path = manifest_dir.join("../test-apps/test-apps.yml");

    let benchmarks_path = options
        .benchmarks_path
        .as_ref()
        .unwrap_or(&default_benchmarks_path);
    let benchmarks_path = benchmarks_path
        .canonicalize()
        .wrap_err_with(|| format!("{} does not exist", benchmarks_path.display()))?;
    let mut benchmarks = Benchmarks::from(&benchmarks_path)?;
    benchmarks.resolve(benchmarks_path.parent().unwrap());

    if let Some(ref materialize_path) = benchmarks.config.materialize {
        // generate source of truth for any downstream consumers based on configuration
        let materialized = benchmarks.clone().materialize()?;
        let materialize_file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&materialize_path)?;
        let mut materialize_writer = std::io::BufWriter::new(materialize_file);
        use std::io::Write;
        write!(
            &mut materialize_writer,
            r#"
    ##
    ## AUTO GENERATED! DO NOT EDIT
    ##
    ## this file was materialized based on {}
    ##

    "#,
            benchmarks_path.display()
        )?;
        serialize_to_writer(Format::YAML, &mut materialize_writer, &materialized)?;
        println!(
            "materialized to {}",
            materialize_path.relative_to(cwd).display()
        );
    }

    let config = &benchmarks.config;
    let benchmark_concurrency = match options.command {
        Command::Profile(_) => config.profile.common.concurrency,
        Command::Trace(_) => config.trace.common.concurrency,
        Command::Simulate(_) => config.sim.common.concurrency,
        Command::Build(_) => Some(1),
        _ => Some(1),
    };

    let concurrency = options
        .concurrency
        .or(benchmark_concurrency)
        .unwrap_or_else(|| num_cpus::get_physical());
    println!("concurrency: {}", &concurrency);

    // get benchmark configurations
    let enabled_benches: Vec<_> = benchmarks
        .enabled_benchmark_configurations()
        .filter(|(name, _bench, _input)| {
            if options.benchmarks.is_empty() {
                true
            } else {
                options
                    .benchmarks
                    .iter()
                    .any(|b| b.to_lowercase() == name.to_lowercase())
            }
        })
        .collect();
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
                let input_clone = input.as_ref().cloned().unwrap_or_default();
                let res = run_bechmark(name.clone(), &bench, input, &config, &options, &bar).await;
                bar.inc(1);

                let op = match options.command {
                    Command::Profile(_) => "profiling",
                    Command::Trace(_) => "tracing",
                    Command::Simulate(_) => "simulating",
                    Command::Build(_) => "building",
                    _ => "",
                };
                let executable = std::env::current_dir()
                    .ok()
                    .map(|cwd| bench.executable().relative_to(cwd))
                    .unwrap_or_else(|| bench.executable());
                bar.println(format!(
                    "{:>15} {:>20} [ {} {} ] {}",
                    op,
                    if res.is_ok() {
                        style(name).green()
                    } else {
                        style(name).red()
                    },
                    executable.display(),
                    &input_clone.cmd_args.join(" "),
                    match res {
                        Ok(_) => {
                            style("succeeded".to_string())
                        }
                        Err(ref err) => {
                            static PREVIEW_LEN: usize = 75;
                            let err_preview = err.to_string();
                            if err_preview.len() > PREVIEW_LEN {
                                let err_preview = format!(
                                    "{} ...",
                                    &err_preview[..err_preview.len().min(PREVIEW_LEN)]
                                );
                            }
                            style(format!("failed: {}", err_preview)).red()
                        }
                    }
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

    for err in &failed {
        dbg!(&err);
    }

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
