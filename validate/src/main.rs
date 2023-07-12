#![allow(warnings)]

mod progress;

#[cfg(feature = "remote")]
mod remote;

use chrono::{offset::Local, DateTime};
use clap::Parser;
use color_eyre::eyre::{self, WrapErr};
use color_eyre::owo_colors::colors::css::Indigo;
use color_eyre::owo_colors::OwoColorize;
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
pub struct CleanOptions {}

#[derive(Parser, Debug, Clone)]
pub struct ProfileOptions {}

#[derive(Parser, Debug, Clone)]
pub struct TraceOptions {}

#[derive(Parser, Debug, Clone)]
pub struct AccelsimTraceOptions {}

#[derive(Parser, Debug, Clone)]
pub struct SimOptions {}

#[derive(Parser, Debug, Clone)]
pub struct AccelsimSimOptions {}

#[derive(Parser, Debug, Clone)]
pub struct PlaygroundSimOptions {}

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

    #[clap(long = "fail-fast", help = "fail fast", default_value = "false")]
    fail_fast: bool,

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
    AccelsimTrace(AccelsimTraceOptions),
    Simulate(SimOptions),
    AccelsimSimulate(AccelsimSimOptions),
    PlaygroundSimulate(PlaygroundSimOptions),
    Build(BuildOptions),
    Clean(CleanOptions),
    Expand(ExpandOptions),
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("benchmark {0} skipped")]
    Skipped(materialize::BenchmarkConfig),
    #[error("benchmark {bench} failed")]
    Failed {
        bench: materialize::BenchmarkConfig,
        #[source]
        source: eyre::Report,
    },
}

// #[inline]
// fn create_dirs(path: impl AsRef<Path>) -> eyre::Result<()> {
//     let path = path.as_ref();
//     utils::fs::create_dirs(path)
//         .wrap_err_with(|| format!("failed to create dir {}", path.display()))?;
//     Ok(())
// }

#[inline]
fn open_writable(path: impl AsRef<Path>) -> eyre::Result<std::io::BufWriter<std::fs::File>> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        utils::fs::create_dirs(path)?;
        // create_dirs(parent)?;
    }
    let writer = utils::fs::open_writable(path)?;
    // .wrap_err_with(|| format!("failed to open {} for writing", path.display()))?;
    Ok(writer)
}

async fn run_benchmark(
    bench: &materialize::BenchmarkConfig,
    // name: String,
    // bench: &validate::Benchmark,
    // input: Result<Input, validate::CallTemplateError>,
    // config: &Config,
    options: &Options,
    bar: &ProgressBar,
    // ) -> Result<materialize::BenchmarkConfig, Error> {
    // ) -> eyre::Result<materialize::BenchmarkConfig>
) -> eyre::Result<()> {
    bar.set_message(bench.name.clone());
    // let input = input?;
    // let exec = bench.executable;

    // let mut tmpl_values = template::Values {
    //     inputs: input.values,
    //     // todo: could also just clone the full benchmark and just set a private name?
    //     bench: template::BenchmarkValues { name: name.clone() },
    // };

    // let res: eyre::Result<()> = match options.command {
    match options.command {
        Command::Profile(ref opts) => {
            let options = profile::nvprof::Options {};
            let results = profile::nvprof::nvprof(&bench.executable, &bench.args, &options)
                .await
                .map_err(|err| match err {
                    profile::Error::Command(err) => err.into_eyre(),
                    err => err.into(),
                })?;

            let writer = open_writable(&bench.profile.metrics_file)?;
            serde_json::to_writer_pretty(writer, &results.metrics)?;
            let mut writer = open_writable(&bench.profile.log_file)?;
            writer.write_all(results.raw.as_bytes())?;
        }
        Command::Trace(ref opts) => {
            // create traces dir
            let traces_dir = &bench.trace.traces_dir;
            utils::fs::create_dirs(traces_dir)?;

            let options = invoke_trace::Options {
                traces_dir: traces_dir.clone(),
                tracer_so: None, // auto detect
                save_json: bench.trace.save_json,
                full_trace: bench.trace.full_trace,
            };
            invoke_trace::trace(&bench.executable, &bench.args, &options)
                .await
                .map_err(|err| match err {
                    invoke_trace::Error::Command(err) => err.into_eyre(),
                    err => err.into(),
                })?;

            // let dur = std::time::Duration::from_secs(3);
            // println!("sleeping for {:?}", &dur);
            // tokio::time::sleep(dur).await;
        }
        // Command::Simulate(ref opts) => {}
        Command::Build(_) | Command::Clean(_) => {
            let should_build = options.force || !bench.executable.is_file();
            let makefile = bench.path.join("Makefile");
            if !makefile.is_file() {
                eyre::bail!("Makefile at {} not found", makefile.display());
            }
            if should_build {
                let mut cmd = async_process::Command::new("make");
                cmd.args(["-C", &*bench.path.to_string_lossy()]);
                if let Command::Clean(_) = options.command {
                    cmd.arg("clean");
                }
                let result = cmd.output().await?;
                if !result.status.success() {
                    return Err(utils::CommandError::new(&cmd, result).into_eyre());
                }
            }
        }
        _ => {}
    }

    Ok(())
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
        Command::Trace(_) => config.trace.common.concurrency,
        Command::Simulate(_) => Some(1), // config.sim.common.concurrency,
        Command::Build(_) | Command::Clean(_) => None, // no limit on concurrency
        _ => Some(1),
    };

    let concurrency = options
        .concurrency
        .or(benchmark_concurrency)
        .unwrap_or_else(num_cpus::get_physical);
    println!("concurrency: {}", &concurrency);

    // get benchmark configurations
    let mut enabled_benches: Vec<_> = materialized
        .enabled()
        .filter(|bench_config| {
            if options.benchmarks.is_empty() {
                return true;
            }
            options
                .benchmarks
                .iter()
                .any(|b| b.to_lowercase() == bench_config.name.to_lowercase())
        })
        .collect();

    if let Command::Build(_) | Command::Clean(_) = options.command {
        // do not build the same executables multiple times
        enabled_benches.dedup_by_key(|bench_config| bench_config.executable.clone());
    }

    let num_bench_configs = enabled_benches.len();

    // create progress bar
    let bar = Arc::new(ProgressBar::new(enabled_benches.len() as u64));
    bar.enable_steady_tick(std::time::Duration::from_secs_f64(1.0 / 10.0));
    let bar_style = ProgressStyle::default();
    bar.set_style(bar_style.into());
    bar.hidden();

    let should_exit = Arc::new(std::sync::atomic::AtomicBool::new(false));

    let results: Vec<Result<_, Error>> = stream::iter(enabled_benches.into_iter())
        .map(|bench_config| {
            let options = options.clone();
            let bar = bar.clone();
            let should_exit = should_exit.clone();
            async move {
                if should_exit.load(std::sync::atomic::Ordering::Relaxed) {
                    return Err(Error::Skipped(bench_config.clone()));
                }
                let start = std::time::Instant::now();
                let res = run_benchmark(bench_config, &options, &bar).await;
                bar.inc(1);

                let op = match options.command {
                    Command::Profile(_) => "profiling",
                    Command::Trace(_) => "tracing [box]",
                    Command::AccelsimTrace(_) => "tracing [accelsim]",
                    Command::Simulate(_) => "simulating [box]",
                    Command::AccelsimSimulate(_) => "simulating [accelsim]",
                    Command::PlaygroundSimulate(_) => "simulating [playground]",
                    Command::Build(_) => "building",
                    Command::Clean(_) => "cleaning",
                    Command::Expand(_) => "expanding",
                };
                let executable = std::env::current_dir().ok().map_or_else(
                    || bench_config.executable.clone(),
                    |cwd| bench_config.executable.relative_to(cwd),
                );
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
                    match res {
                        Ok(_) => {
                            format!("succeeded in {:?}", start.elapsed())
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
                            format!(
                                "failed after {:?}: {}",
                                start.elapsed(),
                                style(err_preview).red()
                            )
                        }
                    }
                ));

                if options.fail_fast && res.is_err() {
                    should_exit.store(true, std::sync::atomic::Ordering::Relaxed);
                }
                res.map_err(|source| Error::Failed {
                    source,
                    bench: bench_config.clone(),
                })
            }
        })
        .buffer_unordered(concurrency)
        .collect()
        .await;
    bar.finish_and_clear();

    let _ = utils::fs::rchmod_writable(&materialized.config.results_dir);

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

    for err in failed {
        match err {
            Error::Failed { ref source, bench } => {
                eprintln!(
                    "============ {} ============",
                    style(format!("{} failed", &bench)).red()
                );
                eprintln!("{source:?}\n");
            }
            Error::Skipped(_bench) => {}
        }
    }
    Ok(())
}
