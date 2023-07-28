// #![allow(warnings)]

mod accelsim;
mod options;
mod playground;
mod profile;
mod progress;
mod simulate;
mod stats;
mod trace;

// #[cfg(feature = "remote")]
// mod remote;

use chrono::offset::Local;
use clap::Parser;
use color_eyre::eyre::{self, WrapErr};
use console::{style, Style};
use futures::stream::{self, StreamExt};
use options::{Command, Options};

use indicatif::ProgressBar;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use utils::fs::create_dirs;
use validate::benchmark::paths::PathExt;
use validate::materialize::{BenchmarkConfig, Benchmarks};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("benchmark {0} skipped")]
    Skipped(BenchmarkConfig),
    #[error("benchmark {0} canceled")]
    Canceled(BenchmarkConfig),
    #[error("benchmark {bench} failed")]
    Failed {
        bench: BenchmarkConfig,
        #[source]
        source: eyre::Report,
    },
}

#[derive(thiserror::Error, Debug)]
pub enum RunError {
    #[error("benchmark skipped")]
    Skipped,
    #[error(transparent)]
    Failed(#[from] eyre::Report),
}

#[inline]
fn open_writable(path: impl AsRef<Path>) -> eyre::Result<std::io::BufWriter<std::fs::File>> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        create_dirs(parent)?;
    }
    let writer = utils::fs::open_writable(path)?;
    Ok(writer)
}

#[allow(clippy::too_many_lines)]
async fn run_benchmark(
    bench: BenchmarkConfig,
    options: &Options,
    bar: &ProgressBar,
) -> Result<(), RunError> {
    bar.set_message(bench.name.clone());
    match options.command {
        Command::Expand(ref _opts) => {
            // do nothing
            Ok(())
        }
        Command::Profile(ref opts) => profile::profile(bench, options, opts).await,
        Command::AccelsimTrace(ref opts) => accelsim::trace(bench, options, opts).await,
        Command::Trace(ref opts) => trace::trace(bench, options, opts).await,
        Command::Simulate(ref opts) => simulate::simulate(bench, options, opts).await,
        Command::AccelsimSimulate(ref opts) => accelsim::simulate(bench, options, opts).await,
        Command::PlaygroundSimulate(ref opts) => playground::simulate(bench, options, opts).await,
        Command::Build(_) | Command::Clean(_) => {
            if let Command::Build(_) = options.command {
                if !options.force && bench.executable.is_file() {
                    return Err(RunError::Skipped);
                }
            }

            let makefile = bench.path.join("Makefile");
            if !makefile.is_file() {
                return Err(RunError::from(eyre::eyre!(
                    "Makefile at {} not found",
                    makefile.display()
                )));
            }
            let mut cmd = async_process::Command::new("make");
            cmd.args(["-C", &*bench.path.to_string_lossy()]);
            if let Command::Clean(_) = options.command {
                cmd.arg("clean");
            }
            log::debug!("{:?}", &cmd);
            let result = cmd.output().await.map_err(eyre::Report::from)?;
            if !result.status.success() {
                return Err(RunError::Failed(
                    utils::CommandError::new(&cmd, result).into_eyre(),
                ));
            }
            Ok(())
        }
    }
}

fn parse_benchmarks(options: &Options) -> eyre::Result<Benchmarks> {
    let cwd = std::env::current_dir()?;
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
        .ok_or_else(|| eyre::eyre!("{} has no parent base path", benches_file_path.display()))?
        .to_path_buf();
    let benchmarks = validate::Benchmarks::from(&benches_file_path)?;

    let materialize_path = benchmarks
        .config
        .materialize_to
        .as_ref()
        .map(|p| p.resolve(&base_dir));

    // materialize config: source of truth for downstream consumers
    let materialized = benchmarks.materialize(&base_dir)?;

    if let Some(materialize_path) = materialize_path {
        use std::io::Write;
        let mut materialize_file = open_writable(&materialize_path)?;
        write!(
            &mut materialize_file,
            r"
##
## AUTO GENERATED! DO NOT EDIT
##
## this configuration was materialized from {} on {}
##

",
            benches_file_path.display(),
            Local::now().format("%d/%m/%Y %T"),
        )?;

        serde_yaml::to_writer(&mut materialize_file, &materialized)?;
        println!(
            "materialized to {}",
            materialize_path.relative_to(cwd).display()
        );
    }

    Ok(materialized)
}

/// get benchmark configurations
pub fn filter_benchmarks(benches: &mut Vec<BenchmarkConfig>, options: &Options) {
    benches.retain(|bench_config| {
        if options.selected_benchmarks.is_empty() {
            // keep all benchmarks when no filters provided
            return true;
        }

        let name = bench_config.name.to_lowercase();
        for b in &options.selected_benchmarks {
            let valid_patterns = [
                // try "benchmark_name"
                &name,
                // try "benchmark_name[input_idx]"
                &format!("{}[{}]", name, bench_config.input_idx + 1),
                // try "benchmark_name@input_idx"
                &format!("{}@{}", name, bench_config.input_idx + 1),
            ];
            if valid_patterns.into_iter().any(|p| b.to_lowercase() == *p) {
                // keep benchmark config
                return true;
            }
        }
        // skip
        false
    });

    if let Command::Build(_) | Command::Clean(_) = options.command {
        // do not build the same executables multiple times
        benches.dedup_by_key(|bench_config| bench_config.executable.clone());
    }
}

fn print_benchmark_result(
    bench_config: &BenchmarkConfig,
    result: &Result<(), RunError>,
    elapsed: std::time::Duration,
    bar: &ProgressBar,
    options: &Options,
) {
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
    let (color, status) = match result {
        Ok(_) => (Style::new().green(), format!("succeeded in {elapsed:?}")),
        Err(RunError::Skipped) => (
            Style::new().yellow(),
            "skipped (already exists)".to_string(),
        ),
        Err(RunError::Failed(ref err)) => {
            static PREVIEW_LEN: usize = 75;
            let mut err_preview = err.to_string();
            if err_preview.len() > PREVIEW_LEN {
                err_preview = format!("{} ...", &err_preview[..err_preview.len().min(PREVIEW_LEN)]);
            }
            (
                Style::new().red(),
                format!("failed after {elapsed:?}: {err_preview}"),
            )
        }
    };
    match options.command {
        Command::Build(_) | Command::Clean(_) => {
            bar.println(format!(
                "{:>15} {:>20} [ {} ] {}",
                op,
                color.apply_to(&bench_config.name),
                executable.display(),
                color.apply_to(status),
            ));
        }
        _ => {
            let benchmark_config_id =
                format!("{}@{:<3}", &bench_config.name, bench_config.input_idx + 1);
            bar.println(format!(
                "{:>15} {:>20} [ {} {} ] {}",
                op,
                color.apply_to(benchmark_config_id),
                executable.display(),
                bench_config.args.join(" "),
                color.apply_to(status),
            ));
        }
    };
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> eyre::Result<()> {
    env_logger::init();
    color_eyre::install()?;

    let start = std::time::Instant::now();

    // load env variables from .env files
    dotenv::dotenv().ok();

    let options = Arc::new(Options::parse());

    // parse benchmarks
    let materialized = parse_benchmarks(&options)?;

    if let Command::Expand(ref opts) = options.command {
        if opts.full {
            println!("{:#?}", &materialized);
            return Ok(());
        }
    }

    let config = &materialized.config;
    let benchmark_concurrency = match options.command {
        Command::Profile(_) => config.profile.common.concurrency,
        Command::Trace(_) => config.trace.common.concurrency,
        Command::AccelsimTrace(_) => config.accelsim_trace.common.concurrency,
        Command::Simulate(_) => config.simulate.common.concurrency,
        Command::AccelsimSimulate(_) => config.accelsim_simulate.common.concurrency,
        Command::PlaygroundSimulate(_) => config.playground_simulate.common.concurrency,
        Command::Build(_) | Command::Clean(_) => None, // no limit on concurrency
        Command::Expand(_) => Some(1),                 // to keep deterministic ordering
    };

    let concurrency = options
        .concurrency
        .or(benchmark_concurrency)
        .unwrap_or_else(num_cpus::get_physical);
    println!("concurrency: {}", &concurrency);

    let mut enabled_benches: Vec<_> = materialized.enabled().cloned().collect();
    filter_benchmarks(&mut enabled_benches, &options);
    let num_bench_configs = enabled_benches.len();

    // create progress bar
    let bar = Arc::new(ProgressBar::new(enabled_benches.len() as u64));
    bar.enable_steady_tick(std::time::Duration::from_secs_f64(1.0 / 10.0));
    let bar_style = progress::Style::default();
    bar.set_style(bar_style.into());

    let should_exit = Arc::new(std::sync::atomic::AtomicBool::new(false));

    let results: Vec<Result<_, Error>> = stream::iter(enabled_benches)
        .map(|bench_config| {
            let options = options.clone();
            let bar = bar.clone();
            let should_exit = should_exit.clone();
            async move {
                if should_exit.load(std::sync::atomic::Ordering::Relaxed) {
                    return Err(Error::Canceled(bench_config.clone()));
                }
                let start = std::time::Instant::now();
                let res = run_benchmark(bench_config.clone(), &options, &bar).await;
                bar.inc(1);
                print_benchmark_result(&bench_config, &res, start.elapsed(), &bar, &options);

                if options.fail_fast && res.is_err() {
                    should_exit.store(true, std::sync::atomic::Ordering::Relaxed);
                }
                match res {
                    Ok(()) => Ok(()),
                    Err(RunError::Skipped) => Err(Error::Skipped(bench_config)),
                    Err(RunError::Failed(source)) => Err(Error::Failed {
                        source,
                        bench: bench_config,
                    }),
                }
            }
        })
        .buffer_unordered(concurrency)
        .collect()
        .await;
    bar.finish();

    let _ = utils::fs::rchmod_writable(&materialized.config.results_dir);

    let (succeeded, failed): (Vec<_>, Vec<_>) = utils::partition_results(results);
    assert_eq!(num_bench_configs, succeeded.len() + failed.len());

    let mut num_failed = 0;
    let mut num_skipped = 0;
    let mut num_canceled = 0;
    for err in failed {
        match err {
            Error::Failed { ref source, bench } => {
                num_failed += 1;
                eprintln!(
                    "============ {} ============",
                    style(format!("{} failed", &bench)).red()
                );
                eprintln!("{source:?}\n");
            }
            Error::Skipped(_) => num_skipped += 1,
            Error::Canceled(_) => num_canceled += 1,
        }
    }

    let failed_msg = style(format!("{num_failed} failed"));
    println!(
        "\n\n => ran {} benchmark configurations in {:?}: {} canceled, {} skipped, {}",
        num_bench_configs,
        start.elapsed(),
        num_canceled,
        num_skipped,
        if num_failed > 0 {
            failed_msg.red()
        } else {
            failed_msg
        },
    );

    std::process::exit(i32::from(num_failed > 0));
}
