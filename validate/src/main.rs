#![allow(warnings)]

mod progress;

#[cfg(feature = "remote")]
mod remote;

use chrono::{offset::Local, DateTime};
use clap::Parser;
use color_eyre::eyre::{self, WrapErr};
use color_eyre::owo_colors::colors::css::Indigo;
use console::style;
use futures::stream::{self, StreamExt};
use futures::Future;
use indicatif::ProgressBar;
use progress::ProgressStyle;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use validate::benchmark::{paths::PathExt, template};
use validate::{materialize, Benchmark, Benchmarks, Config};

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
pub struct TraceOptions {}

#[derive(Parser, Debug, Clone)]
pub struct ExpandOptions {}

#[derive(Parser, Debug, Clone)]
pub struct Options {
    #[clap(short = 'p', long = "path", help = "path to benchmarks yaml file")]
    pub benches_file_path: Option<PathBuf>,

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
    Trace(TraceOptions),
    Simulate(ProfileOptions),
    Build(BuildOptions),
    Expand(ExpandOptions),
}

async fn run_benchmark(
    bench: &materialize::BenchmarkConfig,
    // name: String,
    // bench: &validate::Benchmark,
    // input: Result<Input, validate::CallTemplateError>,
    // config: &Config,
    options: &Options,
    bar: &ProgressBar,
) -> eyre::Result<()> {
    bar.set_message(bench.name.clone());
    // let input = input?;
    // let exec = bench.executable;

    // let mut tmpl_values = template::Values {
    //     inputs: input.values,
    //     // todo: could also just clone the full benchmark and just set a private name?
    //     bench: template::BenchmarkValues { name: name.clone() },
    // };

    let res: eyre::Result<()> = match options.command {
        Command::Profile(ref opts) => {
            let options = profile::nvprof::Options {};
            let results = profile::nvprof::nvprof(&bench.executable, &bench.args, &options).await?;

            let writer = utils::fs::open_writable_as_nobody(&bench.profile.metrics_file)?;
            serde_json::to_writer_pretty(writer, &results.metrics)?;
            let mut writer = utils::fs::open_writable_as_nobody(&bench.profile.log_file)?;
            writer.write_all(results.raw.as_bytes())?;
            Ok(())
        }
        Command::Trace(ref opts) => {
            // create traces dir
            let traces_dir = &bench.trace.traces_dir;
            utils::fs::create_dirs_as_nobody(traces_dir)
                .wrap_err_with(|| format!("failed to create dir {}", traces_dir.display()))?;

            let options = invoke_trace::Options {
                traces_dir: traces_dir.clone(),
                tracer_so: None, // auto detect
                save_json: bench.trace.save_json,
                full_trace: bench.trace.full_trace,
            };
            let results = invoke_trace::trace(&bench.executable, &bench.args, &options).await?;
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
    let default_benches_file_path = manifest_dir.join("../test-apps/test-apps.yml");

    let benches_file_path = options
        .benches_file_path
        .as_ref()
        .unwrap_or(&default_benches_file_path);
    let benches_file_path = benches_file_path
        .canonicalize()
        .wrap_err_with(|| format!("{} does not exist", benches_file_path.display()))?;

    let base_dir = benches_file_path
        .parent()
        .ok_or_else(|| eyre::eyre!("{} has no parent base path", benches_file_path.display()))?;
    let mut benchmarks = Benchmarks::from(&benches_file_path)?;
    let materialize_path = benchmarks
        .config
        .materialize_to
        .as_ref()
        .map(|p| p.resolve(base_dir));

    // materialize config: source of truth for downstream consumers
    let materialized = benchmarks.materialize(base_dir)?;

    if let Command::Expand(_) = options.command {
        dbg!(&materialized);
        return Ok(());
    }

    if let Some(materialize_path) = materialize_path {
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
## this configuration was materialized from {} on {}
##"#,
            benches_file_path.display(),
            Local::now().format("%d/%m/%Y %T"),
        )?;
        write!(&mut materialize_writer, "\n\n");
        serialize_to_writer(Format::YAML, &mut materialize_writer, &materialized)?;
        println!(
            "materialized to {}",
            materialize_path.relative_to(cwd).display()
        );
    }

    let config = &materialized.config;
    let benchmark_concurrency = match options.command {
        Command::Profile(_) => config.profile.common.concurrency,
        Command::Trace(_) => Some(1), // config.trace.common.concurrency,
        Command::Simulate(_) => Some(1), // config.sim.common.concurrency,
        Command::Build(_) => Some(1),
        _ => Some(1),
    };

    let concurrency = options
        .concurrency
        .or(benchmark_concurrency)
        .unwrap_or_else(|| num_cpus::get_physical());
    println!("concurrency: {}", &concurrency);

    // get benchmark configurations
    let enabled_benches: Vec<_> = materialized
        .enabled()
        // .enabled_benchmark_configurations()
        // .filter(|(name, _bench, _input)| {
        .filter(|bench_config| {
            if options.benchmarks.is_empty() {
                return true;
            }
            options
                .benchmarks
                .iter()
                .any(|b| b.to_lowercase() == bench_config.name.to_lowercase())
            // if options.benchmarks.is_empty() {
            //
            //     true
            // } else {
            //     options
            //         .benchmarks
            //         .iter()
            //         .any(|b| b.to_lowercase() == name.to_lowercase())
            // }
        })
        .collect();
    let num_bench_configs = enabled_benches.len();

    // create progress bar
    let bar = Arc::new(ProgressBar::new(enabled_benches.len() as u64));
    bar.enable_steady_tick(std::time::Duration::from_secs_f64(1.0 / 10.0));
    let bar_style = ProgressStyle::default();
    bar.set_style(bar_style.into());

    let results: Vec<_> = stream::iter(enabled_benches.into_iter())
        // .map(|(name, bench, input)| {
        .map(|bench_config| {
            // let name = name.clone();
            let options = options.clone();
            let bar = bar.clone();
            async move {
                // let input_clone = input.as_ref().cloned().unwrap_or_default();
                // let res = run_benchmark(name.clone(), &bench, input, &config, &options, &bar).await;
                let res = run_benchmark(bench_config, &options, &bar).await;
                // let res: Result<(), eyre::Report> = Ok(());
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
                    // .map(|cwd| bench.executable().relative_to(cwd))
                    .map(|cwd| bench_config.executable.relative_to(cwd))
                    // .as_ref()
                    // .unwrap_or_else(|| bench.executable());
                    .unwrap_or_else(|| bench_config.executable.clone());
                bar.println(format!(
                    "{:>15} {:>20} [ {} {} ] {}",
                    op,
                    if res.is_ok() {
                        style(&bench_config.name).green()
                    } else {
                        style(&bench_config.name).red()
                    },
                    executable.display(),
                    bench_config.args.join(" "),
                    // &input_clone.cmd_args.join(" "),
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

    let _ = utils::chown(
        materialized.config.results_dir,
        utils::UID_NOBODY,
        utils::GID_NOBODY,
        false,
    );

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
        eprintln!("{:?}", err);
    }
    Ok(())
}
