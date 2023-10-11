use clap::Parser;
use color_eyre::eyre::{self, WrapErr};
use gpucachesim::config;
use gpucachesim_benchmarks::pchase;
use itertools::Itertools;
use std::collections::HashSet;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

const KB: usize = 1024;
const DEFAULT_ITER_SIZE: usize = ((48 * KB) / 2) / std::mem::size_of::<u32>();

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Bytes(pub u64);

impl std::fmt::Display for Bytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", human_bytes::human_bytes(self.0 as f64))
    }
}

impl std::str::FromStr for Bytes {
    type Err = parse_size::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Ok(Self(parse_size::parse_size(value)?))
    }
}

#[derive(Parser, Debug, Clone)]
#[clap(
    version = option_env!("CARGO_PKG_VERSION").unwrap_or("unknown"),
    about = "trace CUDA applications",
    author = "romnn <contact@romnn.com>",
)]
pub struct Options {
    #[clap(short = 'm', long = "mem", help = "the memory to microbenchmark")]
    pub memory: pchase::Memory,
    #[clap(short = 'n', long = "size", help = "size of the memory in bytes")]
    pub size_bytes: Option<Bytes>,
    #[clap(long = "start-size", help = "start size of the memory in bytes")]
    pub start_size_bytes: Option<Bytes>,
    #[clap(long = "end-size", help = "end size of the memory in bytes")]
    pub end_size_bytes: Option<Bytes>,
    #[clap(
        long = "step-size",
        help = "step size when iterating over the memory size in bytes."
    )]
    pub step_size_bytes: Option<Bytes>,
    #[clap(short = 's', long = "stride", help = "memory access stride in bytes")]
    pub stride_bytes: Bytes,
    #[clap(short = 'w', long = "warmup", help = "number of warmup iterations")]
    pub warmup_iterations: usize,
    #[clap(short = 'k', long = "iterations", help = "number of iterations")]
    pub iter_size: Option<usize>,
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    color_eyre::install()?;
    gpucachesim::init_logging();

    let start = std::time::Instant::now();

    let args: Vec<_> = std::env::args().skip(1).collect();

    let options = match Options::try_parse() {
        Ok(options) => options,
        Err(err) => {
            // parse without flags
            // let mut size_bytes = None;
            // let mut start_size_bytes = None;
            // let mut stop_size_bytes = None;
            // let mut step_size_bytes = 1;
            let memory = pchase::Memory::from_str(&args[0])?;

            if args.len() == 5 {
                Options {
                    memory,
                    size_bytes: Some(args[1].parse()?),
                    start_size_bytes: None,
                    end_size_bytes: None,
                    step_size_bytes: None,
                    // start_size_bytes: Some(args[1].parse()?),
                    // end_size_bytes: Some(args[1].parse()?),
                    // step_size_bytes: Some(Bytes(1)),
                    stride_bytes: args[2].parse()?,
                    warmup_iterations: args[3].parse()?,
                    iter_size: args
                        .get(4)
                        .map(String::as_str)
                        .map(str::parse)
                        .transpose()?,
                }
            } else if args.len() == 7 {
                Options {
                    memory,
                    size_bytes: None,
                    start_size_bytes: Some(args[1].parse()?),
                    end_size_bytes: Some(args[2].parse()?),
                    step_size_bytes: Some(args[3].parse()?),
                    stride_bytes: args[2].parse()?,
                    warmup_iterations: args[3].parse()?,
                    iter_size: args
                        .get(4)
                        .map(String::as_str)
                        .map(str::parse)
                        .transpose()?,
                }

                // start_size_bytes = Some(args[1].parse()?);
                // end_size_bytes = Some(args[2].parse()?);
                // step_size_bytes = Some(args[3].parse()?);
            } else {
                err.exit();
                // return Err(
                //     eyre::Report::new(err).wrap_err("need eitehr 5 or 7 command line arguments")
                // );
            }
        }
    };

    let Options {
        memory,
        size_bytes,
        start_size_bytes,
        end_size_bytes,
        step_size_bytes,
        stride_bytes,
        warmup_iterations,
        iter_size,
    } = options;

    let start_size_bytes = start_size_bytes
        .or(size_bytes)
        .ok_or(eyre::eyre!("missing start size in bytes"))?;
    let end_size_bytes = end_size_bytes
        .or(size_bytes)
        .ok_or(eyre::eyre!("missing end size in bytes"))?;
    let step_size_bytes = step_size_bytes.unwrap_or(Bytes(0));
    if step_size_bytes.0 < 1 {
        eyre::bail!(
            "invalid step size ({:?}) will cause infinite loop",
            step_size_bytes
        );
    }

    let (commands, kernel_traces) = pchase::pchase(
        memory,
        start_size_bytes.0 as usize,
        end_size_bytes.0 as usize,
        step_size_bytes.0 as usize,
        stride_bytes.0 as usize,
        warmup_iterations,
        iter_size.unwrap_or(DEFAULT_ITER_SIZE),
    )
    .await?;

    for command in &commands {
        eprintln!("{}", command);
    }
    for (_launch_config, kernel_trace) in &kernel_traces {
        let warp_traces = kernel_trace.clone().to_warp_traces();
        let first_warp = &warp_traces[&(trace_model::Dim::ZERO, 0)];

        let simplified_trace =
            gpucachesim::exec::tracegen::fmt::simplify_warp_trace(&first_warp, true)
                .collect::<Vec<_>>();
        for inst in &simplified_trace {
            eprintln!("{}", inst);
        }
    }
    eprintln!(
        "trace reconstruction completed with {} command(s) and {} kernel trace(s) in {:?}",
        commands.len(),
        kernel_traces.len(),
        start.elapsed()
    );

    let start = std::time::Instant::now();
    let temp_dir = tempfile::tempdir()?;
    let traces_dir = temp_dir.path();
    gpucachesim::exec::write_traces(commands, kernel_traces, &traces_dir)?;

    let all_addresses = Arc::new(Mutex::new(HashSet::new()));
    let all_latencies = Arc::new(Mutex::new(HashSet::new()));
    let all_addresses_cb = all_addresses.clone();
    let all_latencies_cb = all_latencies.clone();

    let fetch_return_callback = Box::new(
        move |cycle: u64, fetch: &gpucachesim::mem_fetch::MemFetch| {
            let Some(inject_cycle) = fetch.inject_cycle else {
            return;
        };
            let rel_addr = fetch.relative_byte_addr();
            let latency = cycle - inject_cycle;
            dbg!(cycle, latency, rel_addr);
            all_addresses_cb.lock().unwrap().insert(rel_addr);
            all_latencies_cb.lock().unwrap().insert(latency);
        },
    );

    let sim_config = config::gtx1080::build_config(&config::Input::default())?;
    gpucachesim::init_deadlock_detector();
    let mut sim = gpucachesim::config::GTX1080::new(Arc::new(sim_config));
    for cluster in &sim.clusters {
        for core in &cluster.read().cores {
            core.write().fetch_return_callback = Some(fetch_return_callback.clone());
        }
    }

    let (traces_dir, commands_path) = if traces_dir.is_dir() {
        (traces_dir.to_path_buf(), traces_dir.join("commands.json"))
    } else {
        (
            traces_dir
                .parent()
                .map(std::path::Path::to_path_buf)
                .ok_or_else(|| {
                    eyre::eyre!(
                        "could not determine trace dir from file {}",
                        traces_dir.display()
                    )
                })?,
            traces_dir.to_path_buf(),
        )
    };
    sim.add_commands(commands_path, traces_dir)?;
    sim.run()?;

    let stats = sim.stats();
    // for kernel_stats in &stats.inner {
    //     // dbg!(&kernel_stats.l1d_stats);
    //     // dbg!(&kernel_stats.l2d_stats);
    //     // dbg!(&kernel_stats.dram.reduce());
    //     // dbg!(&kernel_stats.sim);
    // }

    let reduced = stats.clone().reduce();
    let l1d_stats = reduced.l1d_stats.reduce();
    dbg!(&l1d_stats);

    let l1d_read_hits: usize = l1d_stats
        .iter()
        .filter(|((_, access), _)| access.is_read() && access.is_hit())
        .map(|(_, count)| count)
        .sum();
    let l1d_read_misses: usize = l1d_stats
        .iter()
        .filter(|((_, access), _)| {
            access.is_read() && (access.is_miss() || access.is_pending_hit())
        })
        .map(|(_, count)| count)
        .sum();

    eprintln!("L1D read hits: {:<10}", l1d_read_hits);
    eprintln!("L1D read misses: {:<10}", l1d_read_misses);

    // eprintln!("L1D misses:            {:<10}", l1d_stats.num_misses());
    // eprintln!(
    //     "L1D pending hits:      {:<10}",
    //     l1d_stats.num_pending_hits()
    // );
    // eprintln!(
    //     "L1D reservation fails: {:<10}",
    //     l1d_stats.num_reservation_fails()
    // );

    let num_kernels_launched = stats.inner.len();
    assert_eq!(num_kernels_launched, 1);

    drop(sim);
    drop(fetch_return_callback);
    let all_adresses: Vec<_> = Arc::into_inner(all_addresses)
        .unwrap()
        .into_inner()
        .unwrap()
        .into_iter()
        .sorted()
        .collect();
    let all_latencies: Vec<_> = Arc::into_inner(all_latencies)
        .unwrap()
        .into_inner()
        .unwrap()
        .into_iter()
        .sorted()
        .collect();
    dbg!(all_adresses.len());
    dbg!(all_latencies);

    eprintln!("simulated completed in {:?}", start.elapsed());
    Ok(())
}
