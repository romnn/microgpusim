// #![allow(warnings)]

mod progress;

// #[cfg(feature = "remote")]
// mod remote;

use chrono::offset::Local;
use clap::Parser;
use color_eyre::eyre::{self, WrapErr};
use console::{style, Style};
use futures::stream::{self, StreamExt};
use std::io::Write;
use validate::options::{self, Command, Options};

use indicatif::ProgressBar;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use validate::benchmark::paths::PathExt;
use validate::materialize::{self, BenchmarkConfig, Benchmarks};

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

impl From<Error> for Result<(), validate::RunError> {
    fn from(err: Error) -> Self {
        match err {
            Error::Failed { source, .. } => Err(validate::RunError::Failed(source)),
            _ => Ok(()),
        }
    }
}

async fn run_make(
    bench: &BenchmarkConfig,
    options: &Options,
    _bar: &indicatif::ProgressBar,
) -> Result<(), validate::RunError> {
    if let Command::Build(_) = options.command {
        if !options.force && bench.executable.is_file() {
            return Err(validate::RunError::Skipped);
        }
    }

    let makefile = bench.path.join("Makefile");
    if !makefile.is_file() {
        return Err(validate::RunError::from(eyre::eyre!(
            "Makefile at {} not found",
            makefile.display()
        )));
    }
    let mut cmd = async_process::Command::new("make");
    cmd.args(["-B", "-C", &*bench.path.to_string_lossy()]);
    if let Command::Clean(_) = options.command {
        cmd.arg("clean");
    }
    log::debug!("{:?}", &cmd);
    let result = cmd.output().await.map_err(eyre::Report::from)?;
    if !result.status.success() {
        return Err(validate::RunError::Failed(
            utils::CommandError::new(&cmd, result).into_eyre(),
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
async fn run_command(
    command: Command,
    bench: &BenchmarkConfig,
    options: &Options,
    bar: &ProgressBar,
) -> Result<(), validate::RunError> {
    let start = Instant::now();
    let res = match command {
        Command::Profile(ref opts) => validate::profile::profile(bench, options, opts, bar).await,
        Command::AccelsimTrace(ref opts) => {
            validate::accelsim::trace(bench, options, opts, bar).await
        }
        Command::Trace(ref opts) => validate::trace::trace(bench, options, opts, bar).await,
        Command::Simulate(ref opts) => {
            validate::simulate::simulate(bench.clone(), options, opts, bar).await
        }
        Command::AccelsimSimulate(ref opts) => {
            validate::accelsim::simulate(bench, options, opts, bar).await
        }
        Command::PlaygroundSimulate(ref opts) => {
            validate::playground::simulate(bench.clone(), options, opts, bar).await
        }
        Command::Build(_) => run_make(bench, options, bar).await,
        _ => Ok(()),
    };

    let res = res.map_err(|err| Error::new(err, bench.clone()));
    print_benchmark_result(
        &command,
        bench,
        res.as_ref().err(),
        start.elapsed(),
        bar,
        options,
    );
    // }
    match res {
        Ok(()) => Ok(()),
        Err(err) => err.into(),
    }
}

#[allow(clippy::too_many_lines)]
async fn run_benchmark(
    command: &Command,
    bench: BenchmarkConfig,
    options: &Options,
    bar: &ProgressBar,
) -> Result<(), validate::RunError> {
    bar.set_message(bench.name.clone());
    match command {
        Command::Full(ref _opts) => {
            for command in [
                Command::Build(options::Build::default()),
                Command::Profile(options::Profile::default()),
                Command::Trace(options::Trace::default()),
                Command::AccelsimTrace(options::AccelsimTrace::default()),
                Command::Simulate(options::Sim::default()),
                Command::AccelsimSimulate(options::AccelsimSim::default()),
                Command::PlaygroundSimulate(options::PlaygroundSim::default()),
            ] {
                run_command(command, &bench, options, bar).await?;
            }
            // run_command(
            //     &bench,
            //     options,
            //     bar,
            // )
            // .await?;
            // run_command(
            //     &bench,
            //     options,
            //     bar,
            // )
            // .await?;
            // run_command(
            //     &bench,
            //     options,
            //     bar,
            // )
            // .await?;

            // let start = Instant::now();
            // let build_options = options::Build::default();
            // let res = run_make(&bench, options, bar)
            //     .await
            //     .map_err(|err| Error::new(err, bench.clone()));
            // print_benchmark_result(
            //     &Command::Build(build_options),
            //     &bench,
            //     &res,
            //     start.elapsed(),
            //     &bar,
            //     &options,
            // );
            //
            // let start = Instant::now();
            // let profile_options = options::Profile::default();
            // let res = validate::profile::profile(&bench, options, &profile_options, bar)
            //     .await
            //     .map_err(|err| Error::new(err, bench.clone()));
            // print_benchmark_result(
            //     &Command::Profile(profile_options),
            //     &bench,
            //     &res,
            //     start.elapsed(),
            //     &bar,
            //     &options,
            // );
            //
            // let start = Instant::now();
            // let trace_options = options::Trace::default();
            // let res = validate::trace::trace(&bench, options, &trace_options, bar)
            //     .await
            //     .map_err(|err| Error::new(err, bench.clone()));
            // print_benchmark_result(
            //     &Command::Trace(trace_options),
            //     &bench,
            //     &res,
            //     start.elapsed(),
            //     &bar,
            //     &options,
            // );
            //
            // let start = Instant::now();
            // let trace_options = options::Trace::default();
            // let res = validate::accelsim::acce(&bench, options, &trace_options, bar)
            //     .await
            //     .map_err(|err| Error::new(err, bench.clone()));
            // print_benchmark_result(
            //     &Command::Trace(trace_options),
            //     &bench,
            //     &res,
            //     start.elapsed(),
            //     &bar,
            //     &options,
            // );

            Ok(())
        }
        Command::Expand(ref _opts) => {
            // do nothing
            Ok(())
        }
        Command::Profile(ref opts) => validate::profile::profile(&bench, options, opts, bar).await,
        Command::AccelsimTrace(ref opts) => {
            validate::accelsim::trace(&bench, options, opts, bar).await
        }
        Command::Trace(ref opts) => validate::trace::trace(&bench, options, opts, bar).await,
        Command::Simulate(ref opts) => {
            validate::simulate::simulate(bench, options, opts, bar).await
        }
        Command::AccelsimSimulate(ref opts) => {
            validate::accelsim::simulate(&bench, options, opts, bar).await
        }
        Command::PlaygroundSimulate(ref opts) => {
            validate::playground::simulate(bench, options, opts, bar).await
        }
        Command::Build(_) | Command::Clean(_) => run_make(&bench, options, bar).await,
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

    match materialize_path {
        Some(materialize_path) if !options.dry_run => {
            let mut materialize_file = validate::open_writable(&materialize_path)?;
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
        _ => {
            println!("dry-run: skipped materialization");
        }
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
                &format!("{}[{}]", name, bench_config.input_idx),
                // try "benchmark_name@input_idx"
                &format!("{}@{}", name, bench_config.input_idx),
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
    command: &Command,
    bench_config: &BenchmarkConfig,
    result: Option<&Error>,
    elapsed: std::time::Duration,
    bar: &ProgressBar,
    _options: &Options,
) {
    let op = match command {
        Command::Profile(_) => "profiling",
        Command::Trace(_) => "tracing [box]",
        Command::AccelsimTrace(_) => "tracing [accelsim]",
        Command::Simulate(_) => "simulating [box]",
        Command::AccelsimSimulate(_) => "simulating [accelsim]",
        Command::PlaygroundSimulate(_) => "simulating [playground]",
        Command::Build(_) => "building",
        Command::Clean(_) => "cleaning",
        Command::Expand(_) => "expanding",
        Command::Full(_) => "validating",
    };
    let op = style(op).cyan();
    let executable = std::env::current_dir().ok().map_or_else(
        || bench_config.executable.clone(),
        |cwd| bench_config.executable.relative_to(cwd),
    );
    let (color, status) = match result {
        None => (Style::new().green(), format!("succeeded in {elapsed:?}")),
        Some(Error::Canceled(_)) => (Style::new().color256(0), "canceled".to_string()),
        Some(Error::Skipped(_)) => (
            Style::new().yellow(),
            "skipped (already exists)".to_string(),
        ),
        Some(Error::Failed { source, .. }) => {
            static PREVIEW_LEN: usize = 75;
            let mut err_preview = source.to_string();
            if err_preview.len() > PREVIEW_LEN {
                err_preview = format!("{} ...", &err_preview[..err_preview.len().min(PREVIEW_LEN)]);
            }
            (
                Style::new().red(),
                format!("failed after {elapsed:?}: {err_preview}"),
            )
        }
    };
    match command {
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
                format!("{}@{:<3}", &bench_config.name, bench_config.input_idx);
            bar.println(format!(
                "{:>15} {:>20} [ {} ][ {} {} ] {}",
                op,
                color.apply_to(benchmark_config_id),
                materialize::bench_config_name(&bench_config.name, &bench_config.values),
                executable.display(),
                bench_config.args.join(" "),
                color.apply_to(status),
            ));
        }
    };
}

fn available_concurrency(options: &Options, config: &materialize::Config) -> usize {
    let benchmark_concurrency = match options.command {
        Command::Profile(_) => config.profile.common.concurrency,
        Command::Trace(_) => config.trace.common.concurrency,
        Command::AccelsimTrace(_) => config.accelsim_trace.common.concurrency,
        Command::Simulate(_) => config.simulate.common.concurrency,
        Command::AccelsimSimulate(_) => config.accelsim_simulate.common.concurrency,
        Command::PlaygroundSimulate(_) => config.playground_simulate.common.concurrency,
        Command::Build(_) | Command::Clean(_) => None, // no limit on concurrency
        Command::Full(_) | Command::Expand(_) => Some(1),
    };

    let max_concurrency = num_cpus::get_physical();
    let concurrency = options
        .concurrency
        .or(benchmark_concurrency)
        .unwrap_or(max_concurrency);
    concurrency.min(max_concurrency)
}

impl Error {
    #[must_use]
    pub fn new(err: validate::RunError, bench_config: BenchmarkConfig) -> Self {
        match err {
            validate::RunError::Skipped => Error::Skipped(bench_config),
            validate::RunError::Failed(source) => Error::Failed {
                source,
                bench: bench_config,
            },
        }
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> eyre::Result<()> {
    env_logger::init();
    color_eyre::install()?;
    dotenv::dotenv().ok();

    let start = Instant::now();
    let options = Arc::new(Options::parse());

    // parse benchmarks
    let materialized = parse_benchmarks(&options)?;

    if let Command::Expand(ref opts) = options.command {
        if opts.full {
            println!("{:#?}", &materialized);
            return Ok(());
        }
    }

    let concurrency = available_concurrency(&options, &materialized.config);
    println!("concurrency: {concurrency}");

    let mut enabled_benches: Vec<_> = materialized.enabled().cloned().collect();
    filter_benchmarks(&mut enabled_benches, &options);
    let num_bench_configs = enabled_benches.len();

    // create progress bar
    let bar = Arc::new(ProgressBar::new(enabled_benches.len() as u64));
    bar.enable_steady_tick(std::time::Duration::from_secs_f64(1.0 / 10.0));
    bar.set_style(progress::Style::default().into());

    let should_exit = Arc::new(std::sync::atomic::AtomicBool::new(false));

    let results: Vec<Result<_, Error>> = stream::iter(enabled_benches)
        .map(|bench_config| {
            let options = options.clone();
            let bar = bar.clone();
            let should_exit = should_exit.clone();
            async move {
                use std::sync::atomic::Ordering::Relaxed;

                let start = Instant::now();
                let res: Result<_, Error> = if should_exit.load(Relaxed) {
                    Err(Error::Canceled(bench_config.clone()))
                } else {
                    run_benchmark(&options.command, bench_config.clone(), &options, &bar)
                        .await
                        .map_err(|err| Error::new(err, bench_config.clone()))
                };
                bar.inc(1);
                print_benchmark_result(
                    &options.command,
                    &bench_config,
                    res.as_ref().err(),
                    start.elapsed(),
                    &bar,
                    &options,
                );

                match res {
                    Err(Error::Failed { .. }) if options.fail_fast => {
                        should_exit.store(true, Relaxed);
                    }
                    _ => {}
                }
                res
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
