use clap::Parser;
use color_eyre::eyre;
use console::style;
use gpucachesim::config;
use gpucachesim_benchmarks::pchase;
use itertools::Itertools;
use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use trace_model::ToBitString;
use utils::fs::Bytes;

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
    #[clap(short = 'r', long = "repetitions", help = "number of repetitions")]
    pub repetitions: usize,

    #[clap(short = 'k', long = "iterations", help = "number of iterations")]
    pub iter_size: Option<usize>,
    #[clap(long = "max-rounds", help = "maximum number of rounds")]
    pub max_rounds: Option<usize>,

    #[clap(long = "csv", help = "write csv formatted latencies to stdout")]
    pub csv_latencies: bool,
}

async fn simulate_pchase<W>(
    memory: pchase::Memory,
    size_bytes: usize,
    stride_bytes: usize,
    warmup_iterations: usize,
    iter_size: usize,
    repetition: usize,
    mut csv_writer: Option<&mut csv::Writer<W>>,
) -> eyre::Result<()>
where
    W: std::io::Write,
{
    // let start = std::time::Instant::now();
    //
    // let rounds = iter_size as f32 / (size_bytes as f32 / stride_bytes as f32);

    let (commands, kernel_traces) = pchase::pchase(
        memory,
        size_bytes,
        stride_bytes,
        warmup_iterations,
        iter_size,
    )
    .await?;

    if false {
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
    }
    // eprintln!(
    //     "trace reconstruction completed with {} command(s) and {} kernel trace(s) in {:?}",
    //     commands.len(),
    //     kernel_traces.len(),
    //     start.elapsed()
    // );

    let start = std::time::Instant::now();
    let temp_dir = tempfile::tempdir()?;
    let traces_dir = temp_dir.path();
    gpucachesim::exec::write_traces(commands, kernel_traces, &traces_dir)?;

    let accesses = Arc::new(Mutex::new(Vec::new()));
    let all_addresses = Arc::new(Mutex::new(HashSet::new()));
    let all_latencies = Arc::new(Mutex::new(HashSet::new()));

    let accesses_cb = accesses.clone();
    let all_addresses_cb = all_addresses.clone();
    let all_latencies_cb = all_latencies.clone();
    let fetch_return_callback = Box::new(
        move |cycle: u64, fetch: &gpucachesim::mem_fetch::MemFetch| {
            let inject_cycle = fetch.inject_cycle.unwrap();
            let rel_addr = fetch.relative_byte_addr();
            let latency = cycle - inject_cycle;
            if gpucachesim::DEBUG_PRINT {
                eprintln!(
                    "{}",
                    style(format!(
                        "cycle={:<6} RETURNED TO CORE fetch {:<30} rel_addr={:<4} latency={:<4}",
                        cycle, fetch, rel_addr, latency
                    ))
                    .red()
                );
            }
            accesses_cb.lock().unwrap().push((fetch.clone(), latency));
            all_addresses_cb.lock().unwrap().insert(rel_addr);
            all_latencies_cb.lock().unwrap().insert(latency);
        },
    );

    let accesses_cb = accesses.clone();
    let all_addresses_cb = all_addresses.clone();
    let all_latencies_cb = all_latencies.clone();
    let l1_access_callback = Box::new(
        move |cycle: u64,
              fetch: &gpucachesim::mem_fetch::MemFetch,
              access_status: gpucachesim::cache::RequestStatus| {
            let inject_cycle = fetch.inject_cycle.unwrap();
            let rel_addr = fetch.relative_byte_addr();
            let latency = cycle - inject_cycle;
            if gpucachesim::DEBUG_PRINT {
                eprintln!(
                    "{}",
                    style(format!(
                        "cycle={:<6} L1 ACCESS {:<30} {:?} rel_addr={:<4} latency={:<4}",
                        cycle, fetch, access_status, rel_addr, latency
                    ))
                    .red()
                );
            }
            if access_status.is_hit() {
                accesses_cb.lock().unwrap().push((fetch.clone(), latency));
                all_addresses_cb.lock().unwrap().insert(rel_addr);
                all_latencies_cb.lock().unwrap().insert(latency);
            }
        },
    );

    let mut sim_config = config::gtx1080::build_config(&config::Input::default())?;
    sim_config.parallelization = config::Parallelization::Deterministic;
    sim_config.fill_l2_on_memcopy = true;
    // sim_config.num_memory_controllers = 1;
    // sim_config.parallelization = config::Parallelization::Serial;
    if false {
        sim_config.data_cache_l1 = Some(Arc::new(gpucachesim::config::L1DCache {
            // l1_latency: 1,
            l1_latency: 1,
            l1_hit_latency: 80,
            // l1_banks_hashing_function: CacheSetIndexFunc::LINEAR_SET_FUNCTION,
            // l1_banks_hashing_function: Box::<cache::set_index::linear::SetIndex>::default(),
            l1_banks_byte_interleaving: 32,
            l1_banks: 1,
            inner: Arc::new(gpucachesim::config::Cache {
                // accelsim_compat: sim_config.accelsim_compat,
                kind: gpucachesim::config::CacheKind::Sector,
                // kind: CacheKind::Normal,
                // 128B cache
                // num_sets: 2,
                // line_size: 32,
                // associativity: 1,
                num_sets: 4, // 64,
                line_size: 128,
                associativity: 48, // 6,
                replacement_policy: gpucachesim::cache::config::ReplacementPolicy::LRU,
                write_policy:
                    gpucachesim::cache::config::WritePolicy::LOCAL_WRITE_BACK_GLOBAL_WRITE_THROUGH,
                allocate_policy: gpucachesim::cache::config::AllocatePolicy::ON_MISS,
                write_allocate_policy:
                    gpucachesim::cache::config::WriteAllocatePolicy::NO_WRITE_ALLOCATE,
                // set_index_function: CacheSetIndexFunc::FERMI_HASH_SET_FUNCTION,
                // set_index_function: Box::<cache::set_index::fermi::SetIndex>::default(),
                mshr_kind: gpucachesim::mshr::Kind::ASSOC,
                // mshr_kind: mshr::Kind::SECTOR_ASSOC,
                mshr_entries: 128,
                mshr_max_merge: 8,
                miss_queue_size: 4,
                // result_fifo_entries: None,
                l1_cache_write_ratio_percent: 0,
                data_port_width: None,
            }),
        }));
    }

    // dbg!(&sim_config.memory_only);
    // dbg!(&sim_config.num_schedulers_per_core);
    // dbg!(&sim_config.num_simt_clusters);
    // dbg!(&sim_config.num_cores_per_simt_cluster);
    // dbg!(&sim_config.simulate_clock_domains);
    // dbg!(&sim_config.l2_rop_latency);

    gpucachesim::init_deadlock_detector();
    let mut sim = gpucachesim::config::GTX1080::new(Arc::new(sim_config));
    for cluster in &sim.clusters {
        for core in &cluster.cores {
            core.write().fetch_return_callback = Some(fetch_return_callback.clone());
            // core.write().load_store_unit.lock().l1_access_callback =
            core.write().load_store_unit.l1_access_callback = Some(l1_access_callback.clone());
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
    let output_cache_state = std::env::var("PCHASE_OUTPUT_CACHE_STATE")
        .as_deref()
        .unwrap_or("")
        .to_lowercase()
        == "yes";

    sim.add_commands(commands_path, traces_dir)?;
    sim.run()?;

    // for cluster in &sim.clusters {
    //     for core in &cluster.read().cores {
    //         // core.write().fetch_return_callback = Some(fetch_return_callback.clone());
    //     }
    // }

    if output_cache_state {
        let debug_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("debug");
        let cache_state_dir = debug_dir.join("pchase_cache_state");
        std::fs::create_dir_all(&cache_state_dir)?;
        let l2_cache_state_file = cache_state_dir.join(format!(
            "pchase_{:0>12}_{}_l2_cache_state_after.csv",
            size_bytes.to_string(),
            Bytes(size_bytes)
                .to_string()
                .replace(" ", "_")
                .replace(".", "_")
        ));
        sim.write_l2_cache_state(&l2_cache_state_file)?;
        eprintln!("wrote L2 cache state to {}", l2_cache_state_file.display());
    }

    let stats = sim.stats();
    // for kernel_stats in &stats.inner {
    //     // dbg!(&kernel_stats.l1d_stats);
    //     // dbg!(&kernel_stats.l2d_stats);
    //     // dbg!(&kernel_stats.dram.reduce());
    //     // dbg!(&kernel_stats.sim);
    // }

    let reduced = stats.clone().reduce();
    let l1d_stats = reduced.l1d_stats.reduce();
    let l2d_stats = reduced.l2d_stats.reduce();
    // dbg!(&l1d_stats);

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

    // eprintln!("L1D read hits: {:<10}", l1d_read_hits);
    // eprintln!("L1D read misses: {:<10}", l1d_read_misses);

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
    drop(l1_access_callback);

    #[derive(Debug, serde::Serialize)]
    struct CsvRow {
        /// Repetition
        pub r: usize,
        /// Size of the array N.
        pub n: usize,
        /// Monotonic index per (r,n) in range 0..iter_size.
        pub k: usize,
        /// Accesses array index.
        pub index: u64,
        /// Virtual memory address of accessed index.
        pub virt_addr: u64,
        /// Latency of memory access for `virt_addr`.
        pub latency: u64,
    }

    let accesses: Vec<_> = Arc::into_inner(accesses)
        .unwrap()
        .into_inner()
        .unwrap()
        .into_iter()
        .filter(|(fetch, _)| fetch.access_kind().is_global())
        .collect();
    // for (k, (fetch, latency)) in accesses.iter().enumerate() {
    //     eprintln!(
    //         "access {:<3}: {:<40} rel addr={:<4} ({:<4}, {:<4}, {:<4}) bytes={} latency={}",
    //         k,
    //         fetch.to_string(),
    //         fetch.relative_byte_addr(),
    //         fetch.relative_addr().unwrap(),
    //         fetch.byte_addr(),
    //         fetch.addr(),
    //         fetch.access.byte_mask[..128].to_bit_string(),
    //         style(latency).yellow()
    //     );
    // }

    // for (fetch, latency) in &accesses {
    //     eprintln!("latency={:<4} fetch={}", latency, fetch);
    // }
    let post_warmup_index = warmup_iterations * iter_size;
    let valid_accesses = &accesses[post_warmup_index..post_warmup_index + iter_size];
    for (k, (fetch, latency)) in valid_accesses.iter().enumerate() {
        if gpucachesim::DEBUG_PRINT {
            eprintln!(
                "access {:<3}: {:<40} rel addr={:<4} ({:<4}, {:<4}, {:<4}) bytes={} latency={}",
                k,
                fetch.to_string(),
                fetch.relative_byte_addr(),
                fetch.relative_addr().unwrap(),
                fetch.byte_addr(),
                fetch.addr(),
                fetch.access.byte_mask[..128].to_bit_string(),
                style(latency).yellow()
            );
        }

        // dbg!(i, stride_bytes, size_bytes);
        let index = (fetch.relative_byte_addr() as usize + stride_bytes) % size_bytes;
        // (stride_bytes + fetch.relative_byte_addr() as usize + stride_bytes) % size_bytes;
        // let index = ((i + 1) * stride_bytes) % size_bytes;
        // let load_index = ;
        // dbg!(load_index, index);
        // assert_eq!(
        //     load_index as usize,
        //     (size_bytes + index - stride_bytes) % size_bytes
        // );
        if let Some(ref mut csv_writer) = csv_writer.as_mut() {
            csv_writer.serialize(CsvRow {
                r: repetition,
                n: size_bytes,
                k,
                index: index as u64,
                virt_addr: fetch.byte_addr(),
                latency: *latency,
            })?;
        }
    }

    // dbg!(rounds);

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
    // dbg!(all_adresses.len());
    // dbg!(all_latencies);

    eprintln!(
        "L1D hit rate: {:4.2}% ({} hits / {} accesses)",
        &l1d_stats.global_hit_rate() * 100.0,
        &l1d_stats.num_global_hits(),
        &l1d_stats.num_global_accesses(),
    );
    eprintln!(
        "L2D hit rate: {:4.2}% ({} hits / {} accesses)",
        &l2d_stats.global_hit_rate() * 100.0,
        &l2d_stats.num_global_hits(),
        &l2d_stats.num_global_accesses(),
    );
    eprintln!("simulation completed in {:?}", start.elapsed());
    Ok(())
}

fn parse_max_rounds_or_iter_size(
    arg: Option<&str>,
) -> Result<(Option<usize>, Option<usize>), std::num::ParseIntError> {
    match arg {
        Some(rounds) if rounds.starts_with("R") => {
            Ok((rounds.strip_prefix("R").map(str::parse).transpose()?, None))
        }
        size => Ok((None, size.map(str::parse).transpose()?)),
    }
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let start = std::time::Instant::now();
    color_eyre::install()?;
    gpucachesim::init_logging();

    let args: Vec<_> = std::env::args().skip(1).collect();

    let options = match Options::try_parse() {
        Ok(options) => options,
        Err(err) => {
            // parse without flags
            let memory = pchase::Memory::from_str(&args[0])?;

            if args.len() >= 6 {
                let (max_rounds, iter_size) =
                    parse_max_rounds_or_iter_size(args.get(7).map(String::as_str))?;
                Options {
                    memory,
                    size_bytes: None,
                    start_size_bytes: Some(args[1].parse()?),
                    end_size_bytes: Some(args[2].parse()?),
                    step_size_bytes: Some(args[3].parse()?),
                    stride_bytes: args[4].parse()?,
                    warmup_iterations: args[5].parse()?,
                    repetitions: args[6].parse()?,
                    iter_size,
                    max_rounds,
                    csv_latencies: false,
                }
            } else if args.len() >= 4 {
                let (max_rounds, iter_size) =
                    parse_max_rounds_or_iter_size(args.get(5).map(String::as_str))?;
                Options {
                    memory,
                    size_bytes: Some(args[1].parse()?),
                    start_size_bytes: None,
                    end_size_bytes: None,
                    step_size_bytes: None,
                    stride_bytes: args[2].parse()?,
                    warmup_iterations: args[3].parse()?,
                    repetitions: args[4].parse()?,
                    iter_size,
                    max_rounds,
                    csv_latencies: false,
                }
            } else {
                err.exit();
            }
        }
    };

    // eprintln!("options: {:#?}", &options);

    let Options {
        memory,
        size_bytes,
        start_size_bytes,
        end_size_bytes,
        step_size_bytes,
        stride_bytes,
        warmup_iterations,
        repetitions,
        iter_size,
        max_rounds,
        ..
    } = options;

    let start_size_bytes = start_size_bytes
        .or(size_bytes)
        .ok_or(eyre::eyre!("missing start size in bytes"))?;
    let end_size_bytes = end_size_bytes
        .or(size_bytes)
        .ok_or(eyre::eyre!("missing end size in bytes"))?;
    let step_size_bytes = step_size_bytes.unwrap_or(Bytes(1));

    // validate
    if step_size_bytes.0 < 1 {
        eyre::bail!(
            "invalid step size ({:?}) will cause infinite loop",
            step_size_bytes
        );
    }

    let mut csv_writer = csv::WriterBuilder::new()
        .flexible(false)
        .from_writer(std::io::stdout());

    let sizes = (start_size_bytes.0..=end_size_bytes.0).step_by(step_size_bytes.0);
    let num_sizes = sizes.clone().count();
    if num_sizes < 1 {
        eprintln!("WARNING: testing zero sizes");
    }
    if repetitions < 1 {
        eprintln!("WARNING: repetitions is zero");
    }

    for (i, size_bytes) in sizes.enumerate() {
        if size_bytes == 0 {
            continue;
        }

        eprintln!("[{:>3}/{:<3}] size={}", i + 1, num_sizes, Bytes(size_bytes));
        let one_round_size = size_bytes as f32 / stride_bytes.0 as f32;

        let iter_size = match (iter_size, max_rounds) {
            (Some(iter_size), _) => iter_size,
            (_, Some(max_rounds)) => max_rounds * one_round_size as usize,
            _ => 1 * one_round_size as usize,
        };

        let stride_bytes = stride_bytes.0 as usize;
        if size_bytes < stride_bytes {
            eyre::bail!(
                "size ({}) is smaller than stride ({})",
                size_bytes,
                stride_bytes
            );
        }

        for repetition in 0..repetitions {
            simulate_pchase(
                memory,
                size_bytes,
                stride_bytes,
                warmup_iterations,
                iter_size,
                repetition,
                Some(&mut csv_writer),
            )
            .await?;
        }
    }

    std::io::stderr().flush()?;
    std::io::stdout().flush()?;
    eprintln!("completed in {:?}", start.elapsed());
    Ok(())
}
