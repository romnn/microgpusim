use clap::Parser;
use color_eyre::eyre;
use itertools::Itertools;
use std::path::PathBuf;
use std::time::Instant;

#[cfg(all(feature = "jemalloc", not(target_env = "msvc")))]
use tikv_jemallocator::Jemalloc;

#[cfg(all(feature = "jemalloc", not(target_env = "msvc")))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
struct Options {
    /// Input to operate on
    #[arg(value_name = "TRACE_DIR")]
    pub trace_dir: PathBuf,

    /// Stats output file
    #[arg(short = 'o', long = "stats", value_name = "STATS_OUT")]
    pub stats_out_file: Option<PathBuf>,

    /// Turn debugging information on
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub debug: u8,

    /// Use multi-threading
    #[arg(long = "parallel")]
    pub parallel: bool,

    /// Use non-deterministic simulation
    #[arg(long = "nondeterministic")]
    pub non_deterministic: Option<usize>,

    // /// Interleave serial part for non-deterministic simulation
    // #[arg(long = "interleave-serial")]
    // pub interleave_serial: Option<bool>,
    #[clap(long = "cores-per-cluster", help = "cores per cluster")]
    pub cores_per_cluster: Option<usize>,

    #[clap(long = "num-clusters", help = "number of clusters")]
    pub num_clusters: Option<usize>,

    #[clap(
        long = "threads",
        help = "number of threads to use for parallel simulation"
    )]
    pub num_threads: Option<usize>,

    #[clap(long = "mem-only", help = "simulate only memory instructions")]
    pub memory_only: Option<bool>,

    #[clap(long = "fill-l2", help = "fill L2 cache on CUDA memcopy")]
    pub fill_l2: Option<bool>,

    #[clap(long = "flush-l1", help = "flush L1 cache between kernel launches")]
    pub flush_l1: Option<bool>,

    #[clap(long = "flush-l2", help = "flush L2 cache between kernel launches")]
    pub flush_l2: Option<bool>,

    #[clap(long = "accelsim-compat", help = "accelsim compat mode")]
    pub accelsim_compat_mode: Option<bool>,

    #[clap(long = "simulate-clock-domains", help = "simulate clock domains")]
    pub simulate_clock_domains: Option<bool>,

    #[clap(flatten)]
    pub accelsim: gpucachesim::config::accelsim::Config,
}

fn main() -> eyre::Result<()> {
    color_eyre::install()?;
    gpucachesim::init_deadlock_detector();

    let start = Instant::now();
    let options = Options::parse();
    #[cfg(debug_assertions)]
    std::env::set_var("RUST_BACKTRACE", "full");

    let log_after_cycle = std::env::var("LOG_AFTER")
        .unwrap_or_default()
        .parse::<u64>()
        .ok();

    if log_after_cycle.is_none() {
        gpucachesim::init_logging();
    }

    let deadlock_check = std::env::var("DEADLOCK_CHECK")
        .unwrap_or_default()
        .to_lowercase()
        == "yes";

    let parallelization = match (options.parallel, options.non_deterministic) {
        (false, _) => gpucachesim::config::Parallelization::Serial,
        #[cfg(feature = "parallel")]
        (true, None) => gpucachesim::config::Parallelization::Deterministic,
        #[cfg(feature = "parallel")]
        (true, Some(run_ahead)) => {
            gpucachesim::config::Parallelization::Nondeterministic { run_ahead }
        }
        #[cfg(not(feature = "parallel"))]
        _ => eyre::bail!(
            "{} was compiled with parallel simulation disabled",
            env!("CARGO_BIN_NAME")
        ),
    };

    let mut config = gpucachesim::config::GPU {
        // num_simt_clusters: options.num_clusters.unwrap_or(28),
        // num_cores_per_simt_cluster: options.cores_per_cluster.unwrap_or(1),
        // num_schedulers_per_core: 4,                  // 4
        // num_memory_controllers: 12,                  // 8
        // num_dram_chips_per_memory_controller: 1,     // 1
        // num_sub_partitions_per_memory_controller: 2, // 2
        // simulate_clock_domains: options.simulate_clock_domains.unwrap_or(false),
        // fill_l2_on_memcopy: options.fill_l2.unwrap_or(false),
        // flush_l1_cache: options.flush_l1.unwrap_or(false),
        // flush_l2_cache: options.flush_l2.unwrap_or(false),
        // accelsim_compat: options.accelsim_compat_mode.unwrap_or(false),
        // memory_only: options.memory_only.unwrap_or(false),
        parallelization,
        deadlock_check,
        log_after_cycle,
        simulation_threads: options.num_threads,
        ..gpucachesim::config::GPU::default()
    };
    if let Some(accelsim_compat_mode) = options.accelsim_compat_mode {
        config.fill_l2_on_memcopy &= !accelsim_compat_mode;
        config.perfect_inst_const_cache |= accelsim_compat_mode;
        config.accelsim_compat = accelsim_compat_mode;
        config.memory_only &= !accelsim_compat_mode;
    }
    if let Some(num_simt_clusters) = options.num_clusters {
        config.num_simt_clusters = num_simt_clusters;
    }
    if let Some(num_cores_per_simt_cluster) = options.cores_per_cluster {
        config.num_cores_per_simt_cluster = num_cores_per_simt_cluster
    }
    if let Some(simulate_clock_domains) = options.simulate_clock_domains {
        config.simulate_clock_domains = simulate_clock_domains;
    }
    if let Some(fill_l2) = options.fill_l2 {
        config.fill_l2_on_memcopy = fill_l2;
    }
    if let Some(flush_l1) = options.flush_l1 {
        config.flush_l1_cache = flush_l1;
    }
    if let Some(flush_l2) = options.flush_l2 {
        config.flush_l2_cache = flush_l2;
    }
    if let Some(memory_only) = options.memory_only {
        config.memory_only = memory_only;
    }

    dbg!(&config.accelsim_compat);
    dbg!(&config.memory_only);
    dbg!(&config.num_schedulers_per_core);
    dbg!(&config.num_simt_clusters);
    dbg!(&config.num_cores_per_simt_cluster);
    dbg!(&config.simulate_clock_domains);
    dbg!(&config.perfect_inst_const_cache);
    dbg!(&config.fill_l2_on_memcopy);

    let sim = gpucachesim::accelmain(&options.trace_dir, config)?;
    let stats = sim.stats();

    // save stats to file
    if let Some(stats_out_file) = options.stats_out_file.as_ref() {
        gpucachesim::save_stats_to_file(&stats, stats_out_file)?;
    }

    eprintln!("STATS:\n");
    eprintln!("SIM[no-kernel]: {:#?}", &stats.no_kernel.sim);
    eprintln!("L1I[no-kernel]: {:#?}", &stats.no_kernel.l1i_stats.reduce());
    eprintln!("L1D[no-kernel]: {:#?}", &stats.no_kernel.l1d_stats.reduce());
    eprintln!("L2D[no-kernel]: {:#?}", &stats.no_kernel.l2d_stats.reduce());
    eprintln!("DRAM[no-kernel]: {:#?}", &stats.no_kernel.dram.reduce());
    eprintln!("ACCESSES[no-kernel]: {:#?}", &stats.no_kernel.accesses,);

    for (kernel_launch_id, kernel_stats) in stats.as_ref().iter().enumerate() {
        eprintln!(
            "\n ===== kernel launch {kernel_launch_id:<3}: {}  =====\n",
            kernel_stats.sim.kernel_name
        );
        eprintln!("DRAM: {:#?}", &kernel_stats.dram.reduce());
        eprintln!("SIM: {:#?}", &kernel_stats.sim);
        eprintln!("INSTRUCTIONS: {:#?}", &kernel_stats.instructions);
        eprintln!("ACCESSES: {:#?}", &kernel_stats.accesses);

        let l1i_stats = kernel_stats.l1i_stats.reduce();
        eprintln!("L1I: {:#?}", &l1i_stats);
        eprintln!(
            "L1I hit rate: {:4.2}% ({} hits / {} accesses)",
            &l1i_stats.hit_rate() * 100.0,
            &l1i_stats.num_hits(),
            &l1i_stats.num_accesses(),
        );

        let l1d_stats = kernel_stats.l1d_stats.reduce();
        eprintln!("L1D: {:#?}", &l1d_stats);
        eprintln!(
            "L1D hit rate: {:4.2}% ({} hits / {} accesses)",
            &l1d_stats.global_hit_rate() * 100.0,
            &l1d_stats.num_global_hits(),
            &l1d_stats.num_global_accesses(),
        );

        let l2d_stats = kernel_stats.l2d_stats.reduce();
        eprintln!("L2D: {:#?}", &l2d_stats);
        eprintln!(
            "L2D hit rate: {:4.2}% ({} hits / {} accesses)",
            &l2d_stats.global_hit_rate() * 100.0,
            &l2d_stats.num_global_hits(),
            &l2d_stats.num_global_accesses(),
        );
        eprintln!(
            "L2D read hit rate: {:4.2}% ({} read hits / {} reads)",
            &l2d_stats.global_read_hit_rate() * 100.0,
            &l2d_stats.num_global_read_hits(),
            &l2d_stats.num_global_reads(),
        );
        eprintln!(
            "L2D write hit rate: {:4.2}% ({} write hits / {} writes)",
            &l2d_stats.global_write_hit_rate() * 100.0,
            &l2d_stats.num_global_write_hits(),
            &l2d_stats.num_global_writes(),
        );
    }
    let timings: Vec<_> = gpucachesim::TIMINGS
        .lock()
        .clone()
        .into_iter()
        .sorted_by_key(|(label, _)| label.to_string())
        .collect();
    if !timings.is_empty() {
        eprintln!("TIMINGS:");
    }

    let total_time = start.elapsed();
    let norm_time = if gpucachesim::config::Parallelization::Serial != parallelization {
        timings
            .iter()
            .map(|(_, dur)| dur.total())
            .sum::<std::time::Duration>()
        // .max()
        // .copied()
        // .unwrap_or(std::time::Duration::ZERO)
    } else {
        total_time
    };
    for (label, value) in timings {
        let mean = value.mean();
        let total = value.total();
        let percent = total.as_secs_f64() / norm_time.as_secs_f64();
        eprintln!(
            "\t{:<35} {: >15} ({: >4.2}% total: {: >15})",
            label,
            format!("{:?}", mean),
            percent * 100.0,
            format!("{:?}", total),
        );
    }
    eprintln!("completed in {:?}", total_time);
    Ok(())
}
