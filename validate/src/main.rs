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

use indicatif::ProgressBar;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use validate::{
    benchmark::paths::PathExt,
    materialized::{self, BenchmarkConfig, Benchmarks},
    options::{self, Command, Options},
};

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

async fn run_benchmark(
    command: &Command,
    bench: BenchmarkConfig,
    options: &Options,
    bar: &ProgressBar,
) -> Result<(), validate::RunError> {
    bar.set_message(bench.name.clone());
    match command {
        Command::Full(ref _opts) => unreachable!(),
        Command::Expand(ref _opts) => {
            println!("{:#?}", &bench);
            Ok(())
        }
        Command::Profile(ref opts) => {
            validate::profile::profile(&bench, options, opts, bar).await
            // Ok(())
        }
        Command::AccelsimTrace(ref opts) => {
            validate::accelsim::trace(&bench, options, opts, bar).await
            // Ok(())
        }
        Command::Trace(ref opts) => {
            validate::trace::trace(&bench, options, opts, bar).await
            // Ok(())
        }
        Command::Simulate(ref opts) => {
            validate::simulate::simulate(bench, options, opts, bar).await
            // Ok(())
        }
        Command::AccelsimSimulate(ref opts) => {
            validate::accelsim::simulate(&bench, options, opts, bar).await
            // Ok(())
        }
        Command::PlaygroundSimulate(ref opts) => {
            validate::playground::simulate(bench, options, opts, bar).await
            // Ok(())
        }
        Command::Build(_) | Command::Clean(_) => {
            run_make(&bench, options, bar).await
            // Ok(())
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

    // the materialized config is the source of truth for downstream consumers
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
        Command::Full(_) => unreachable!(),
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
                "{:<25} {:<20} [ {} ] {}",
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
                "{:<25} {:<20} [ {} ][ {} {} ] {}",
                op,
                color.apply_to(benchmark_config_id),
                materialized::bench_config_name(&bench_config.name, &bench_config.values, true),
                executable.display(),
                bench_config.args.join(" "),
                color.apply_to(status),
            ));
        }
    };
}

fn available_concurrency(options: &Options, config: &materialized::Config) -> usize {
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

use itertools::Itertools;

fn compute_per_command_bench_configs<'a>(
    materialized: &'a Benchmarks,
    commands: Vec<Command>,
    options: &'a Options,
) -> eyre::Result<Vec<(Command, Vec<&'a BenchmarkConfig>)>> {
    let queries: Vec<validate::benchmark::Input> = options
        .query
        .iter()
        .map(|q| serde_json::from_str(q).wrap_err_with(|| format!("failed to parse query {q:?}")))
        .try_collect()?;

    let per_command_bench_configs: Vec<(_, _)> = commands
        .into_iter()
        .map(|command| {
            use std::collections::HashSet;
            let targets: HashSet<_> = command.targets().collect();
            let mut bench_configs: Vec<_> = materialized
                .benchmark_configs()
                .filter(|bench_config| {
                    if !targets.contains(&bench_config.target) {
                        return false;
                    }

                    if !options.selected_benchmarks.is_empty() {
                        let name = bench_config.name.to_lowercase();
                        let is_match = options.selected_benchmarks.iter().any(|b| {
                            let valid_patterns = [
                                // try "benchmark_name"
                                name.clone(),
                                // try "benchmark_name[input_idx]"
                                format!("{}[{}]", name, bench_config.input_idx),
                                // try "benchmark_name@input_idx"
                                format!("{}@{}", name, bench_config.input_idx),
                            ];
                            valid_patterns
                                .into_iter()
                                .any(move |p| b.to_lowercase() == *p)
                        });
                        if !is_match {
                            return false;
                        }
                    }

                    if !options.query.is_empty() {
                        let is_match = queries
                            .iter()
                            .any(|query| bench_config.input_matches(query));
                        if !is_match {
                            return false;
                        }
                    }

                    true
                })
                .collect();

            if let Command::Build(_) | Command::Clean(_) = command {
                // do not build the same executables multiple times
                bench_configs.dedup_by_key(|bench_config| bench_config.executable.clone());
            }

            // sort benchmarks
            bench_configs.sort_by_key(|bench_config| {
                (
                    bench_config.target,
                    bench_config.name.clone(),
                    bench_config.input_idx,
                )
            });

            (command, bench_configs)
        })
        .collect();
    Ok(per_command_bench_configs)
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

    let commands = match &options.command {
        Command::Full(_) => vec![
            Command::Build(options::Build::default()),
            Command::Profile(options::Profile::default()),
            Command::Trace(options::Trace::default()),
            Command::AccelsimTrace(options::AccelsimTrace::default()),
            Command::Simulate(options::Sim::default()),
            Command::AccelsimSimulate(options::AccelsimSim::default()),
            Command::PlaygroundSimulate(options::PlaygroundSim::default()),
        ],
        other => vec![other.clone()],
    };

    let per_command_bench_configs =
        compute_per_command_bench_configs(&materialized, commands, &options)?;
    let num_bench_configs = per_command_bench_configs
        .iter()
        .flat_map(|(_command, bench_configs)| bench_configs)
        .count();

    // create progress bar
    let bar = Arc::new(ProgressBar::new(num_bench_configs as u64));
    bar.enable_steady_tick(std::time::Duration::from_secs_f64(1.0 / 10.0));
    bar.set_style(progress::Style::default().into());

    let should_exit = Arc::new(std::sync::atomic::AtomicBool::new(false));

    let mut results: Vec<Result<_, Error>> = Vec::new();

    for (command, bench_configs) in per_command_bench_configs {
        let step_results: Vec<Result<_, Error>> = stream::iter(bench_configs.into_iter())
            .map(|bench_config| {
                let options = options.clone();
                let bar = bar.clone();
                let should_exit = should_exit.clone();
                let bench_config = bench_config.clone();
                let command = command.clone();
                async move {
                    use std::sync::atomic::Ordering::Relaxed;

                    let start = Instant::now();
                    let res: Result<_, Error> = if should_exit.load(Relaxed) {
                        Err(Error::Canceled(bench_config.clone()))
                    } else {
                        run_benchmark(&command, bench_config.clone(), &options, &bar)
                            .await
                            .map_err(|err| Error::new(err, bench_config.clone()))
                    };
                    bar.inc(1);
                    print_benchmark_result(
                        &command,
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
        results.extend(step_results);
        if results
            .iter()
            .any(|res| matches!(res, Err(Error::Failed { .. })))
        {
            break;
        }
    }
    // do not finish the bar if a stage failed
    if results.len() == num_bench_configs {
        bar.finish();
    }

    let _ = utils::fs::rchmod_writable(&materialized.config.results_dir);

    let (_succeeded, failed): (Vec<_>, Vec<_>) = utils::partition_results(results);
    // assert_eq!(num_bench_configs, succeeded.len() + failed.len());

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
