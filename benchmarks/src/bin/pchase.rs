use clap::Parser;
use color_eyre::eyre;
use gpucachesim::config;
use gpucachesim_benchmarks::pchase;
use std::collections::HashSet;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

const KB: usize = 1024;
const DEFAULT_ITER_SIZE: usize = ((48 * KB) / 2) / std::mem::size_of::<u32>();

// #[derive(thiserror::Error, Debug)]
// pub struct InvalidSizeError {
//     value: String,
//     err: parse_size::Error,
// }
//
// impl std::fmt::Display for InvalidSizeError {
//     fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
//         write!(f, "invalid size {:?}: {}", self.value, self.err)
//     }
// }

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Bytes(pub u64);

impl std::fmt::Display for Bytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", human_bytes::human_bytes(self.0 as f64))
    }
}

// impl TryFrom<&str> for Bytes {
//     type Error = parse_size::Error;
//
//     fn try_from(value: &str) -> Result<Self, Self::Error> {
//         Ok(Self(parse_size::parse_size(value)?))
//     }
// }

impl std::str::FromStr for Bytes {
    type Err = parse_size::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Ok(Self(parse_size::parse_size(value)?))
    }
}

// impl std::str::FromStr for Bytes {
//     type Err = InvalidSizeError;
//
//     fn from_str(value: &str) -> Result<Self, Self::Err> {
//         let bytes = parse_size::parse_size(value).map_err(|err| InvalidSizeError {
//             err,
//             value: value.to_string(),
//         })?;
//         Ok(Self(bytes))
//     }
// }

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
    pub size_bytes: Bytes,
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
    env_logger::init();
    let start = std::time::Instant::now();

    let args: Vec<_> = std::env::args().skip(1).collect();

    let options = match Options::try_parse() {
        Ok(options) => options,
        Err(_) => {
            // parse without flags
            let memory = pchase::Memory::from_str(&args[0])?;
            let size_bytes = args[1].parse()?;
            let stride_bytes = args[2].parse()?;
            let warmup_iterations = args[3].parse()?;

            let iter_size = args
                .get(4)
                .map(String::as_str)
                .map(str::parse)
                .transpose()?;
            Options {
                memory,
                size_bytes,
                stride_bytes,
                warmup_iterations,
                iter_size,
            }
        }
    };

    let Options {
        memory,
        size_bytes,
        stride_bytes,
        warmup_iterations,
        iter_size,
    } = options;

    let (commands, kernel_traces) = pchase::pchase(
        memory,
        size_bytes.0 as usize,
        stride_bytes.0 as usize,
        warmup_iterations,
        iter_size.unwrap_or(DEFAULT_ITER_SIZE),
    )
    .await?;
    for (_launch_config, kernel_trace) in &kernel_traces {
        let warp_traces = kernel_trace.clone().to_warp_traces();
        let first_warp = &warp_traces[&(trace_model::Dim::ZERO, 0)];

        let simplified_trace =
            gpucachesim::exec::tracegen::fmt::simplify_warp_trace(&first_warp).collect::<Vec<_>>();
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
            let rel_addr = fetch.relative_addr();
            let latency = cycle - inject_cycle;
            // dbg!(cycle, latency, rel_addr);
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
    for kernel_stats in &stats.inner {
        dbg!(&kernel_stats.l1d_stats);
        // dbg!(&kernel_stats.l2d_stats);
        // dbg!(&kernel_stats.dram.reduce());
        // dbg!(&kernel_stats.sim);
    }

    // let reduced = stats.clone().reduce();
    // dbg!(&reduced.dram.reduce());
    let num_kernels_launched = stats.inner.len();
    dbg!(num_kernels_launched);

    drop(sim);
    drop(fetch_return_callback);
    let all_adresses = Arc::into_inner(all_addresses)
        .unwrap()
        .into_inner()
        .unwrap();
    let all_latencies = Arc::into_inner(all_latencies)
        .unwrap()
        .into_inner()
        .unwrap();
    dbg!(all_adresses.len());
    dbg!(all_latencies);

    eprintln!("simulated completed in {:?}", start.elapsed());
    Ok(())
}
