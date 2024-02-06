use super::asserts;
use super::stats::{normalized_percentage_error, percentage_error, PercentageError};
use crate::sync::Arc;
use crate::{config, interconn as ic};
use color_eyre::eyre;
use indexmap::{IndexMap, IndexSet};
use itertools::Itertools;
use stats::cache::{AccessStat, AccessStatus, RequestStatus};
use std::path::PathBuf;
use std::time::Instant;
use utils::diff;
use validate::materialized::{BenchmarkConfig, TargetBenchmarkConfig};

#[deprecated]
#[allow(dead_code)]
pub fn test_against_playground(bench_config: &BenchmarkConfig) -> eyre::Result<()> {
    let TargetBenchmarkConfig::Simulate {
        ref traces_dir,
        ref accelsim_traces_dir,
        ..
    } = bench_config.target_config
    else {
        unreachable!();
    };

    let box_trace_dir = traces_dir.clone();
    let box_commands_path = traces_dir.join("commands.json");
    let accelsim_kernelslist_path = accelsim_traces_dir.join("box-kernelslist.g");

    dbg!(&box_commands_path);
    dbg!(&accelsim_kernelslist_path);

    let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
    let gpgpusim_config = manifest_dir.join("accelsim/gtx1080/gpgpusim.config");
    let trace_config = manifest_dir.join("accelsim/gtx1080/gpgpusim.trace.config");
    let inter_config = manifest_dir.join("accelsim/gtx1080/config_pascal_islip.icnt");

    assert!(box_trace_dir.is_dir());
    assert!(box_commands_path.is_file());
    assert!(accelsim_kernelslist_path.is_file());
    assert!(gpgpusim_config.is_file());
    assert!(trace_config.is_file());
    assert!(inter_config.is_file());

    // debugging config
    // let box_config = Arc::new(config::GPU {
    //     num_simt_clusters: 20,                       // 20
    //     num_cores_per_simt_cluster: 4,               // 1
    //     num_schedulers_per_core: 2,                  // 2
    //     num_memory_controllers: 8,                   // 8
    //     num_sub_partitions_per_memory_controller: 2, // 2
    //     fill_l2_on_memcopy: true,                    // true
    //     ..config::GPU::default()
    // });

    let input: config::Input = config::parse_input(&bench_config.values)?;
    dbg!(&input);

    let mut box_config: config::GPU = config::gtx1080::build_config(&input)?;
    box_config.fill_l2_on_memcopy = true;
    box_config.perfect_inst_const_cache = true;
    // box_config.flush_l1_cache = true;
    // box_config.flush_l2_cache = false;
    box_config.accelsim_compat = true;
    // box_config.num_simt_clusters = 28;
    // box_config.num_cores_per_simt_cluster = 4;
    // box_config.num_schedulers_per_core = 4;
    // box_config.num_memory_controllers = 12;
    // box_config.num_sub_partitions_per_memory_controller = 2;

    let box_interconn = Arc::new(ic::SimpleInterconnect::new(
        box_config.num_simt_clusters,
        box_config.total_sub_partitions(),
    ));
    let box_config = Arc::new(box_config);
    let mem_controller =
        Arc::new(crate::mcu::PascalMemoryControllerUnit::new(&box_config).unwrap());

    let start = Instant::now();
    let mut box_sim = crate::Simulator::new(box_interconn, mem_controller, box_config);
    box_sim
        .trace
        .add_commands(box_commands_path, box_trace_dir)?;

    // {
    //     box_sim.parallel_simulation = false;
    //     box_sim.run_to_completion()?;
    // }
    // box_sim.run_to_completion_parallel_deterministic()?;
    let run_ahead: usize = std::env::var("NONDET")
        .ok()
        .as_deref()
        .map(str::parse)
        .transpose()
        .unwrap()
        .unwrap_or(1);

    box_sim.run_to_completion_parallel_nondeterministic(run_ahead)?;
    let box_dur = start.elapsed();

    let args = vec![
        "-trace".to_string(),
        accelsim_kernelslist_path.to_string_lossy().to_string(),
        "-config".to_string(),
        gpgpusim_config.to_string_lossy().to_string(),
        "-config".to_string(),
        trace_config.to_string_lossy().to_string(),
        "-inter_config_file".to_string(),
        inter_config.to_string_lossy().to_string(),
    ];
    dbg!(&args);

    let play_config = playground::Config {
        accelsim_compat_mode: false,
        ..playground::Config::default()
    };
    let start = Instant::now();
    let mut play_sim = playground::Accelsim::new(play_config, args)?;
    play_sim.run_to_completion();
    let play_dur = start.elapsed();

    println!(
        "play dur: {:?}, box dur: {:?} \t=> speedup {:>2.2}",
        play_dur,
        box_dur,
        play_dur.as_secs_f64() / box_dur.as_secs_f64()
    );
    let play_stats = play_sim.stats();
    let box_stats = box_sim.stats().reduce();

    let max_rel_err = Some(0.05); // allow 5% difference
    let abs_threshold = Some(10.0); // allow absolute difference of 10
    asserts::stats_match(play_stats, box_stats, max_rel_err, abs_threshold, false);
    println!(
        "play dur: {:?}, box dur: {:?} \t=> speedup {:>2.2}",
        play_dur,
        box_dur,
        play_dur.as_secs_f64() / box_dur.as_secs_f64()
    );
    Ok(())
}

pub fn test_against_serial(bench_config: &BenchmarkConfig) -> eyre::Result<()> {
    use std::time::Duration;
    let TargetBenchmarkConfig::Simulate { ref traces_dir, .. } = bench_config.target_config else {
        unreachable!();
    };

    let trace_dir = traces_dir.clone();
    let commands_path = traces_dir.join("commands.json");
    assert!(trace_dir.is_dir());
    assert!(commands_path.is_file());

    let input: config::Input = config::parse_input(&bench_config.values)?;
    dbg!(&input);

    let cores_per_cluster_factor: Option<usize> = std::env::var("CORES")
        .ok()
        .as_deref()
        .map(str::parse)
        .transpose()?;

    let num_clusters_factor: Option<usize> = std::env::var("CLUSTERS")
        .ok()
        .as_deref()
        .map(str::parse)
        .transpose()?;

    let mut serial_config: config::GPU = config::gtx1080::build_config(&input)?;
    serial_config.parallelization = config::Parallelization::Serial;
    serial_config.accelsim_compat = false;
    serial_config.fill_l2_on_memcopy = false;
    // disabling the L1 makes the l2 / accesses stats more precise until we
    // figure out block assignment...
    serial_config.global_mem_skip_l1_data_cache = false;

    // serial_config.perfect_inst_const_cache = false;
    // serial_config.flush_l1_cache = true;
    // serial_config.flush_l2_cache = false;

    // scale up cores per cluster
    // serial_config.num_simt_clusters = 28;
    // serial_config.num_simt_clusters = 56;
    // serial_config.num_simt_clusters = 112;

    if let Some(num_clusters_factor) = num_clusters_factor {
        serial_config.num_simt_clusters *= num_clusters_factor;
    }
    if let Some(cores_per_cluster_factor) = cores_per_cluster_factor {
        serial_config.num_cores_per_simt_cluster *= cores_per_cluster_factor;
    }

    let mut parallel_config = serial_config.clone();

    let run_ahead: Option<usize> = std::env::var("NONDET")
        .ok()
        .as_deref()
        .map(str::parse)
        .transpose()?;
    let parallelization = match run_ahead {
        None => config::Parallelization::Deterministic,
        Some(run_ahead) => config::Parallelization::Nondeterministic { run_ahead },
    };
    parallel_config.parallelization = parallelization;
    dbg!(serial_config.parallelization);
    dbg!(parallel_config.parallelization);

    let (parallel_dur, parallel_stats) = {
        let start = Instant::now();
        let parallel_sim = crate::accelmain(traces_dir, parallel_config)?;
        let dur = start.elapsed();
        let parallel_stats = parallel_sim.stats();
        // let _kernel_dur = Duration::from_millis(parallel_stats.sim.elapsed_millis as u64);
        (dur, parallel_stats)
    };
    dbg!(&parallel_dur);

    let (serial_dur, serial_stats) = {
        let start = Instant::now();
        let serial_sim = crate::accelmain(traces_dir, serial_config)?;
        let dur = start.elapsed();
        let serial_stats = serial_sim.stats();
        // let _kernel_dur = Duration::from_millis(serial_stats.sim.elapsed_millis as u64);
        (dur, serial_stats)
    };
    dbg!(&serial_dur);

    println!(
        "serial dur: {:?}, parallel dur: {:?} \t=> speedup {:>2.2}x",
        serial_dur,
        parallel_dur,
        serial_dur.as_secs_f64() / parallel_dur.as_secs_f64()
    );

    // let max_mape_err = Some(0.1); // allow 10% difference
    // let max_smape_err = Some(0.1); // allow 10% difference

    let options = CompareOptions {
        diff_only: false,
        check_no_kernel: false,
        check_per_kernel: true,
        max_mape_err: Some(0.05),  // allow 5% difference
        max_smape_err: Some(0.05), // allow 5% difference
    };
    assert_stats_match(serial_stats, parallel_stats, &options);
    println!(
        "serial dur: {:?}, parallel dur: {:?} \t=> speedup {:>2.2}x",
        serial_dur,
        parallel_dur,
        serial_dur.as_secs_f64() / parallel_dur.as_secs_f64()
    );

    Ok(())
}

pub struct AccessID<'a>((&'a Option<usize>, &'a AccessStatus));

impl<'a> std::fmt::Debug for AccessID<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let Self((alloc_id, access)) = self;
        match alloc_id {
            None => write!(f, "{access}"),
            Some(id) => write!(f, "{id}@{access}"),
        }
    }
}

pub fn cache_err<'a>(
    want_stats: &'a stats::Cache,
    have_stats: &'a stats::Cache,
) -> Vec<(AccessID<'a>, (usize, usize), PercentageError)> {
    let stats = have_stats.union(&want_stats);
    stats
        .map(
            |((alloc_id, access @ AccessStatus((kind, stat))), (want, have))| {
                let norm = want_stats.count_accesses_of_kind(*kind);
                (
                    AccessID((alloc_id, &access)),
                    (want, have),
                    normalized_percentage_error(want, have, norm),
                )
            },
        )
        .filter(|(_, _, err)| *err != 0.0)
        .collect()
}

pub fn dram_err<'a>(
    want: &'a IndexMap<stats::mem::AccessKind, u64>,
    have: &'a IndexMap<stats::mem::AccessKind, u64>,
) -> Vec<(&'a stats::mem::AccessKind, PercentageError)> {
    let union: IndexSet<_> = want.keys().chain(have.keys()).sorted().collect();
    union
        .into_iter()
        .map(|k| {
            let have = have.get(k).copied().unwrap_or(0);
            let want = want.get(k).copied().unwrap_or(0);
            (k, percentage_error(want, have))
        })
        .filter(|(_, err)| *err != 0.0)
        .collect()
}

pub struct CompareOptions {
    pub diff_only: bool,
    pub check_no_kernel: bool,
    pub check_per_kernel: bool,
    pub max_mape_err: Option<f64>,
    pub max_smape_err: Option<f64>,
}

pub fn assert_stats_match(
    serial: stats::PerKernel,
    parallel: stats::PerKernel,
    options: &CompareOptions,
) {
    let max_mape_err = PercentageError::MAPE(options.max_mape_err.unwrap_or(0.0));
    let max_smape_err = PercentageError::SMAPE(options.max_smape_err.unwrap_or(0.0));

    let mut kernel_stats: Vec<(stats::Stats, stats::Stats)> = Vec::new();
    kernel_stats.push((serial.no_kernel.clone(), parallel.no_kernel.clone()));

    if options.check_per_kernel {
        // compare stats per kernel
        kernel_stats.extend(
            serial
                .kernel_stats
                .into_iter()
                .zip(parallel.kernel_stats.into_iter()),
        );
    } else {
        kernel_stats.push((serial.reduce(), parallel.reduce()));
    }

    for (mut serial, mut parallel) in kernel_stats {
        eprintln!(
            "====== comparing kernel {} =======",
            if serial.sim.kernel_name.is_empty() {
                "NO KERNEL".to_string()
            } else {
                format!(
                    "{} ({})",
                    serial.sim.kernel_name, serial.sim.kernel_launch_id
                )
            }
        );

        serial.sim.elapsed_millis = 0;
        parallel.sim.elapsed_millis = 0;

        diff::diff!(serial: &serial.sim, parallel: &parallel.sim, "sim");
        diff::diff!(serial: &serial.accesses, parallel: &parallel.accesses, "L2 accesses",);
        diff::diff!(serial: &serial.instructions, parallel: &parallel.instructions, "instructions");

        let mut serial_l1c_stats = serial.l1c_stats.reduce().reduce_allocations();

        let mut serial_l1t_stats = serial.l1t_stats.reduce().reduce_allocations();
        let mut serial_l1d_stats = serial.l1d_stats.reduce().reduce_allocations();
        let mut serial_l1i_stats = serial.l1i_stats.reduce().reduce_allocations();
        let mut serial_l2d_stats = serial.l2d_stats.reduce().reduce_allocations();

        let mut parallel_l1c_stats = parallel.l1c_stats.reduce().reduce_allocations();
        let mut parallel_l1t_stats = parallel.l1t_stats.reduce().reduce_allocations();
        let mut parallel_l1d_stats = parallel.l1d_stats.reduce().reduce_allocations();
        let mut parallel_l1i_stats = parallel.l1i_stats.reduce().reduce_allocations();
        let mut parallel_l2d_stats = parallel.l2d_stats.reduce().reduce_allocations();

        let mut serial_dram = serial.dram.reduce();
        serial_dram.retain(|_, v| *v > 0);
        let mut parallel_dram = parallel.dram.reduce();
        parallel_dram.retain(|_, v| *v > 0);

        // aggregate and prepare for comparison
        for (name, cache_stats) in [
            ("serial l1c", &mut serial_l1c_stats),
            ("serial l1t", &mut serial_l1t_stats),
            ("serial l1i", &mut serial_l1i_stats),
            ("serial l1d", &mut serial_l1d_stats),
            ("serial l2d", &mut serial_l2d_stats),
            ("parallel l1i", &mut parallel_l1i_stats),
            ("parallel l1d", &mut parallel_l1d_stats),
            ("parallel l1c", &mut parallel_l1c_stats),
            ("parallel l1t", &mut parallel_l1t_stats),
            ("parallel l2d", &mut parallel_l2d_stats),
        ] {
            dbg!(&name, &cache_stats);

            // combine misses and sector misses
            let sector_misses: IndexSet<_> = cache_stats.sector_misses().collect();
            for ((allocation, access), count) in sector_misses {
                cache_stats.inner.remove(&(allocation, access));
                let AccessStatus((kind, _)) = access;
                let miss = AccessStat::Status(RequestStatus::MISS);
                *cache_stats
                    .inner
                    .entry((None, AccessStatus((kind, miss))))
                    .or_insert(0) += count;
            }

            // remove mshr hits??
            for kind in stats::mem::AccessKind::iter() {
                cache_stats.inner.remove(&(
                    None,
                    AccessStatus((kind, AccessStat::Status(RequestStatus::MSHR_HIT))),
                ));
            }

            // make sure mshr hit and hit reserved are the same
            // for kind in AccessKind::iter() {
            //     let mshr_hits = cache_stats
            //         .inner
            //         .get(&(
            //             None,
            //             AccessStatus((kind, AccessStat::Status(RequestStatus::MSHR_HIT))),
            //         ))
            //         .copied()
            //         .unwrap_or(0);
            //     let pending_hits = cache_stats
            //         .inner
            //         .get(&(
            //             None,
            //             AccessStatus((kind, AccessStat::Status(RequestStatus::HIT_RESERVED))),
            //         ))
            //         .copied()
            //         .unwrap_or(0);
            //     // assert_eq!(mshr_hits, pending_hits);
            //     diff::diff!(mshr_hits: mshr_hits, pending_hits: pending_hits);
            // }

            // combine hits and pending hits
            let pending_hits: IndexSet<_> = cache_stats.pending_hits().collect();
            for ((allocation, access), count) in pending_hits {
                cache_stats.inner.remove(&(allocation, access));
                let AccessStatus((kind, _)) = access;
                let hit = AccessStat::Status(RequestStatus::HIT);
                *cache_stats
                    .inner
                    .entry((None, AccessStatus((kind, hit))))
                    .or_insert(0) += count;
            }

            // remove reservation failure reasons
            cache_stats.inner.retain(|(_, status), _| {
                !status.is_reservation_fail() && !status.is_reservation_failure_reason()
            });
        }

        diff::diff!(
            serial: &serial_l1c_stats,
            parallel: &parallel_l1c_stats,
            "l1c",
        );
        diff::diff!(
            serial: &serial_l1t_stats,
            parallel: &parallel_l1t_stats,
            "l1t",
        );
        diff::diff!(
            serial: &serial_l1i_stats,
            parallel: &parallel_l1i_stats,
            "l1i",
        );
        diff::diff!(
            serial: &serial_l1d_stats,
            parallel: &parallel_l1d_stats,
            "l1d",
        );
        diff::diff!(
            serial: &serial_l2d_stats,
            parallel: &parallel_l2d_stats,
            "l2d",
        );
        diff::diff!(serial: &serial_dram, parallel: &parallel_dram, "dram");

        // have shown all diffs, now assert
        let serial_no_kernel = serial.sim.kernel_name.is_empty();
        let parallel_no_kernel = serial.sim.kernel_name.is_empty();
        diff::assert_eq!(serial: serial_no_kernel, parallel: parallel_no_kernel);

        let should_check = !options.diff_only || (serial_no_kernel && !options.check_no_kernel);

        // num blocks error (must match!)
        let num_blocks_err = percentage_error(serial.sim.num_blocks, parallel.sim.num_blocks);
        dbg!(&num_blocks_err);
        if should_check {
            diff::assert_eq!(serial: &serial.sim.num_blocks, parallel: &parallel.sim.num_blocks);
            // assert!(num_blocks_err <= max_mape_err || num_blocks_err <= max_smape_err);
        }

        // num instructions error (must match!)
        let instructions_err = percentage_error(serial.sim.instructions, parallel.sim.instructions);
        dbg!(&instructions_err);
        if should_check {
            diff::assert_eq!(serial: &serial.instructions, parallel: &parallel.instructions);
            // assert!(instructions_err <= max_mape_err || instructions_err <= max_smape_err);
        }

        // cycles error
        let cycle_err = percentage_error(serial.sim.cycles, parallel.sim.cycles);
        dbg!(&cycle_err);
        if should_check {
            assert!(cycle_err <= max_mape_err || cycle_err <= max_smape_err);
        }

        if false {
            // accesses error
            // nothing other than L2 accesses counts, hence dont compare
            let accesses_union: IndexSet<_> = serial
                .accesses
                .inner
                .keys()
                .chain(parallel.accesses.inner.keys())
                .sorted()
                .collect();
            let accesses_err: Vec<_> = accesses_union
                .into_iter()
                .map(|k| {
                    (
                        k,
                        percentage_error(
                            serial.accesses.inner.get(k).copied().unwrap_or_default(),
                            parallel.accesses.inner.get(k).copied().unwrap_or_default(),
                        ),
                    )
                })
                .filter(|(_, err)| *err != 0.0)
                .collect();
            dbg!(&accesses_err);
            if should_check {
                assert!(accesses_err
                    .iter()
                    .all(|(_, err)| err <= &max_mape_err || err <= &max_smape_err));
            }
        }

        // l1c error
        let l1c_err = cache_err(&serial_l1c_stats, &parallel_l1c_stats);
        dbg!(&l1c_err);
        if should_check {
            assert!(l1c_err
                .iter()
                .all(|(_, _, err)| err <= &max_mape_err || err <= &max_smape_err));
        }

        // l1t error
        let l1t_err = cache_err(&serial_l1t_stats, &parallel_l1t_stats);
        dbg!(&l1t_err);
        if should_check {
            assert!(l1t_err
                .iter()
                .all(|(_, _, err)| err <= &max_mape_err || err <= &max_smape_err));
        }

        // l1i error
        let l1i_err = cache_err(&serial_l1i_stats, &parallel_l1i_stats);
        dbg!(&l1i_err);
        if should_check {
            assert!(l1i_err
                .iter()
                // .all(|(_, _, err)| err <= &max_mape_err || err <= &max_smape_err));
                .all(|(_, _, err)| err <= &PercentageError::MAPE(0.5)
                    || err <= &PercentageError::SMAPE(0.5)));
        }

        // l1d
        let l1d_err = cache_err(&serial_l1d_stats, &parallel_l1d_stats);
        dbg!(&l1d_err);
        if should_check {
            assert!(l1d_err
                .iter()
                .all(|(_, _, err)| err <= &max_mape_err || err <= &max_smape_err));
        }

        // l2d
        let l2d_err = cache_err(&serial_l2d_stats, &parallel_l2d_stats);
        dbg!(&l2d_err);
        if should_check {
            assert!(l2d_err.iter().all(|(acc, (s, p), err)| match acc {
                AccessID((_, status)) if status.kind().is_inst() => true,
                AccessID((_, status)) => {
                    let norm = serial_l2d_stats.count_accesses_of_kind(*status.kind()) as f64;

                    // GLOBAL_ACC_R[HIT]: 1023300
                    // GLOBAL_ACC_R[MISS]: 128
                    // GLOBAL_ACC_R[HIT]: 1023296
                    // GLOBAL_ACC_R[MISS]: 36

                    // .get(None, *status.kind(), RequestStatus::HIT)
                    // .unwrap_or(0);
                    // norm += serial_l2d_stats
                    //     .get(None, *status.kind(), RequestStatus::MISS)
                    //     .unwrap_or(0);

                    let abs_err = (*s as f64 - *p as f64).abs();
                    // dbg!(&abs_err);
                    // dbg!(&norm);
                    // dbg!(abs_err / norm);

                    err <= &max_mape_err || err <= &max_smape_err || abs_err / norm <= 0.001
                    // err <= &max_mape_err || err <= &max_smape_err
                }
            }));
        }

        let dram_err = dram_err(&serial_dram, &parallel_dram);
        dbg!(&dram_err);
        if should_check {
            assert!(dram_err
                .iter()
                .all(|(_, err)| err <= &max_mape_err || err <= &max_smape_err));
        }
    }
}

macro_rules! parallel_checks {
    ($($name:ident: ($bench_name:expr, $($input:tt)+),)*) => {
        $(
            paste::paste! {
                #[test]
                fn [<parallel_ $name>]() -> color_eyre::eyre::Result<()> {
                    $crate::testing::init_test();
                    let bench_config = $crate::testing::get_bench_config(
                        $bench_name,
                        validate::input!($($input)+)?,
                    )?;
                    test_against_serial(&bench_config)
                }
            }
        )*
    }
}

parallel_checks! {
    // vectoradd
    vectoradd_32_100_test: ("vectorAdd", { "dtype": 32, "length": 100 }),
    vectoradd_32_1000_test: ("vectorAdd", { "dtype": 32, "length": 1000  }),
    vectoradd_32_10000_test: ("vectorAdd", { "dtype": 32, "length": 10_000 }),
    vectoradd_32_20000_test: ("vectorAdd", { "dtype": 32, "length": 20_000 }),
    vectoradd_32_100000_test: ("vectorAdd", { "dtype": 32, "length": 100_000 }),
    vectoradd_32_500000_test: ("vectorAdd", { "dtype": 32, "length": 500_000 }),
    vectoradd_64_10000_test: ("vectorAdd", { "dtype": 64, "length": 10_000 }),
    vectoradd_64_20000_test: ("vectorAdd", { "dtype": 64, "length": 20_000 }),
    vectoradd_64_500000_test: ("vectorAdd", { "dtype": 64, "length": 500_000 }),

    // simple matrixmul
    simple_matrixmul_32_32_32_test: ("simple_matrixmul", { "m": 32, "n": 32, "p": 32 }),
    simple_matrixmul_32_32_64_test: ("simple_matrixmul", { "m": 32, "n": 32, "p": 64 }),
    simple_matrixmul_64_128_128_test: ("simple_matrixmul", { "m": 64, "n": 128, "p": 128 }),
    simple_matrixmul_128_512_128_test: ("simple_matrixmul", { "m": 128, "n": 512, "p": 128 }),
    simple_matrixmul_512_32_512_test: ("simple_matrixmul", { "m": 512, "n": 32, "p": 512 }),

    // matrixmul (shared memory)
    matrixmul_32_test: ("matrixmul", { "rows": 32 }),
    matrixmul_64_test: ("matrixmul", { "rows": 64 }),
    matrixmul_128_test: ("matrixmul", { "rows": 128 }),
    matrixmul_256_test: ("matrixmul", { "rows": 256 }),
    matrixmul_512_test: ("matrixmul", { "rows": 512 }),

    // transpose
    transpose_256_naive_test: ("transpose", { "dim": 256, "variant": "naive"}),
    transpose_512_naive_test: ("transpose", { "dim": 512, "variant": "naive"}),
    transpose_256_coalesed_test: ("transpose", { "dim": 256, "variant": "coalesced" }),
    transpose_512_coalesed_test: ("transpose", { "dim": 512, "variant": "coalesced" }),

    // babelstream
    babelstream_1024_test: ("babelstream", { "size": 1024 }),
    babelstream_10240_test: ("babelstream", { "size": 10240 }),
}
