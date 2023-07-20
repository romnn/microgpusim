// #![allow(warnings)]

mod progress;

// #[cfg(feature = "remote")]
// mod remote;

use chrono::offset::Local;
use clap::Parser;
use color_eyre::{
    eyre::{self, WrapErr},
    Help,
};
use console::style;
use futures::stream::{self, StreamExt};

use indicatif::ProgressBar;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use validate::benchmark::paths::PathExt;
use validate::materialize::{self, BenchmarkConfig, Benchmarks};

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
pub struct ExpandOptions {
    #[clap(long = "full", help = "expand full benchmark config")]
    pub full: bool,
}

#[derive(Parser, Debug, Clone)]
pub struct Options {
    #[clap(short = 'p', long = "path", help = "path to benchmarks yaml file")]
    pub benches_file_path: Option<PathBuf>,

    #[clap(short = 'b', long = "bench", help = "name of benchmark to run")]
    pub selected_benchmarks: Vec<String>,

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
    Skipped(BenchmarkConfig),
    #[error("benchmark {bench} failed")]
    Failed {
        bench: BenchmarkConfig,
        #[source]
        source: eyre::Report,
    },
}

#[inline]
fn open_writable(path: impl AsRef<Path>) -> eyre::Result<std::io::BufWriter<std::fs::File>> {
    let path = path.as_ref();
    if let Some(_parent) = path.parent() {
        utils::fs::create_dirs(path)?;
    }
    let writer = utils::fs::open_writable(path)?;
    Ok(writer)
}

#[allow(clippy::too_many_lines)]
async fn run_benchmark(
    bench: BenchmarkConfig,
    options: &Options,
    bar: &ProgressBar,
) -> eyre::Result<()> {
    bar.set_message(bench.name.clone());
    match options.command {
        Command::Expand(ref _opts) => {
            // do nothing
        }
        Command::Profile(ref _opts) => {
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
        Command::AccelsimTrace(ref _opts) => {
            let traces_dir = &bench.accelsim_trace.traces_dir;
            utils::fs::create_dirs(traces_dir)?;

            let options = accelsim_trace::Options {
                traces_dir: traces_dir.clone(),
                nvbit_tracer_tool: None, // auto detect
                ..accelsim_trace::Options::default()
            };
            accelsim_trace::trace(&bench.executable, &bench.args, &options).await?;
        }
        Command::Trace(ref _opts) => {
            // create traces dir
            let traces_dir = &bench.trace.traces_dir;
            utils::fs::create_dirs(traces_dir)?;

            let options = invoke_trace::Options {
                traces_dir: traces_dir.clone(),
                tracer_so: None, // auto detect
                save_json: bench.trace.save_json,
                #[cfg(debug_assertions)]
                validate: true,
                #[cfg(not(debug_assertions))]
                validate: false,
                full_trace: bench.trace.full_trace,
            };
            invoke_trace::trace(&bench.executable, &bench.args, &options)
                .await
                .map_err(|err| match err {
                    err @ invoke_trace::Error::MissingExecutable(_) => eyre::Report::from(err)
                        .suggestion(
                            "did you build the benchmarks first using `cargo validate build`?",
                        ),
                    err => err.into_eyre(),
                })?;
        }
        Command::Simulate(ref _opts) => {
            // get traces dir from trace config
            let traces_dir = bench.trace.traces_dir;

            // let stats_out_file = bench.simulate.stats_file;
            let _stats =
                tokio::task::spawn_blocking(move || casimu::ported::accelmain(traces_dir, None))
                    .await??;
        }
        Command::AccelsimSimulate(ref _opts) => {
            // get traces dir from accelsim trace config
            let traces_dir = &bench.accelsim_trace.traces_dir;
            let common = &bench.accelsim_simulate.common;

            let timeout = common.timeout.map(Into::into);

            let materialize::AccelsimSimConfigFiles {
                config,
                config_dir,
                trace_config,
                inter_config,
            } = bench.accelsim_simulate.configs.clone();

            let config = accelsim::SimConfig {
                config: Some(config),
                config_dir: Some(config_dir),
                trace_config: Some(trace_config),
                inter_config: Some(inter_config),
            };

            accelsim_sim::simulate_trace(traces_dir, config, timeout).await?;
        }
        Command::PlaygroundSimulate(ref _opts) => {
            // get traces dir from accelsim trace config

            let materialize::AccelsimSimConfigFiles {
                config,
                trace_config,
                inter_config,
                ..
            } = bench.playground_simulate.configs.clone();

            let _stats = tokio::task::spawn_blocking(move || {
                let traces_dir = &bench.accelsim_trace.traces_dir;
                let kernelslist = traces_dir
                    .join("kernelslist.g")
                    .to_string_lossy()
                    .to_string();
                let gpgpusim_config = config.to_string_lossy().to_string();
                let trace_config = trace_config.to_string_lossy().to_string();
                let inter_config = inter_config.to_string_lossy().to_string();

                let args = [
                    "-trace",
                    &kernelslist,
                    "-config",
                    &gpgpusim_config,
                    "-config",
                    &trace_config,
                    "-inter_config_file",
                    &inter_config,
                ];
                dbg!(&args);

                let config = playground::Config::default();
                playground::run(&config, args.as_slice())
                // Ok::<_, eyre::Report>(())
            })
            .await??;
        }
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
    }
    Ok(())
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
        serialize_to_writer(Format::YAML, &mut materialize_file, &materialized)?;
        println!(
            "materialized to {}",
            materialize_path.relative_to(cwd).display()
        );
    }

    Ok(materialized)
}

/// get benchmark configurations
pub fn filter_benchmarks(benches: &mut Vec<&BenchmarkConfig>, options: &Options) {
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
    result: &eyre::Result<()>,
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
    let benchmark_config_id = format!("{}@{:<3}", bench_config.name, bench_config.input_idx);
    bar.println(format!(
        "{:>15} {:>20} [ {} {} ] {}",
        op,
        if result.is_ok() {
            style(benchmark_config_id).green()
        } else {
            style(benchmark_config_id).red()
        },
        executable.display(),
        bench_config.args.join(" "),
        match result {
            Ok(_) => {
                format!("succeeded in {elapsed:?}")
            }
            Err(ref err) => {
                static PREVIEW_LEN: usize = 75;
                let err_preview = err.to_string();
                if err_preview.len() > PREVIEW_LEN {
                    let _err_preview =
                        format!("{} ...", &err_preview[..err_preview.len().min(PREVIEW_LEN)]);
                }
                format!("failed after {:?}: {}", elapsed, style(err_preview).red())
            }
        }
    ));
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

    let mut enabled_benches: Vec<_> = materialized.enabled().collect();
    filter_benchmarks(&mut enabled_benches, &options);
    let num_bench_configs = enabled_benches.len();

    // create progress bar
    let bar = Arc::new(ProgressBar::new(enabled_benches.len() as u64));
    bar.enable_steady_tick(std::time::Duration::from_secs_f64(1.0 / 10.0));
    let bar_style = progress::Style::default();
    bar.set_style(bar_style.into());

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
                let res = run_benchmark(bench_config.clone(), &options, &bar).await;
                bar.inc(1);
                print_benchmark_result(bench_config, &res, start.elapsed(), &bar, &options);

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
    bar.finish();

    let _ = utils::fs::rchmod_writable(&materialized.config.results_dir);

    let (succeeded, failed): (Vec<_>, Vec<_>) = utils::partition_results(results);
    assert_eq!(num_bench_configs, succeeded.len() + failed.len());

    let num_failed = failed.len();
    let failed_msg = format!("{num_failed} failed");
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
    std::process::exit(i32::from(num_failed > 0));
}
