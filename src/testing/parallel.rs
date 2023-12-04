use super::asserts;
use super::stats::{percentage_error, PercentageError};
use crate::sync::Arc;
use crate::{config, interconn as ic};
use color_eyre::eyre;
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

    let box_interconn = Arc::new(ic::ToyInterconnect::new(
        box_config.num_simt_clusters,
        box_config.total_sub_partitions(),
    ));
    let box_config = Arc::new(box_config);

    let start = Instant::now();
    let mut box_sim = crate::MockSimulator::new(box_interconn, box_config);
    box_sim.add_commands(box_commands_path, box_trace_dir)?;

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
    let interleave_serial = false;

    // box_sim.run_to_completion_parallel_nondeterministic(1)?;
    box_sim.run_to_completion_parallel_nondeterministic(run_ahead, interleave_serial)?;
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
    Ok(())
}

pub fn test_against_serial(bench_config: &BenchmarkConfig) -> eyre::Result<()> {
    use std::time::Duration;
    let TargetBenchmarkConfig::Simulate {
        ref traces_dir,
        ..
    } = bench_config.target_config
    else {
        unreachable!();
    };

    let trace_dir = traces_dir.clone();
    let commands_path = traces_dir.join("commands.json");
    assert!(trace_dir.is_dir());
    assert!(commands_path.is_file());

    let input: config::Input = config::parse_input(&bench_config.values)?;
    dbg!(&input);

    let mut serial_config: config::GPU = config::gtx1080::build_config(&input)?;
    serial_config.parallelization = config::Parallelization::Serial;
    serial_config.accelsim_compat = false;
    serial_config.fill_l2_on_memcopy = true;
    serial_config.perfect_inst_const_cache = false;
    // serial_config.flush_l1_cache = true;
    // serial_config.flush_l2_cache = false;
    serial_config.num_simt_clusters = 80;
    // serial_config.num_simt_clusters = 28;
    // serial_config.num_cores_per_simt_cluster = 4;
    // serial_config.num_schedulers_per_core = 4;
    // serial_config.num_memory_controllers = 12;
    // serial_config.num_sub_partitions_per_memory_controller = 2;

    let mut parallel_config = serial_config.clone();

    let run_ahead: Option<usize> = std::env::var("NONDET")
        .ok()
        .as_deref()
        .map(str::parse)
        .transpose()?;
    let parallelization = match run_ahead {
        None => config::Parallelization::Deterministic,
        Some(run_ahead) => config::Parallelization::Nondeterministic {
            run_ahead,
            interleave: true,
        },
    };
    parallel_config.parallelization = parallelization;
    dbg!(serial_config.parallelization);
    dbg!(parallel_config.parallelization);

    let (serial_dur, serial_stats) = {
        let mut serial_sim = crate::accelmain(traces_dir, serial_config)?;
        // let start = Instant::now();
        serial_sim.run()?;
        // let serial_dur = start.elapsed();
        let serial_stats = serial_sim.stats().reduce();
        (
            Duration::from_millis(serial_stats.sim.elapsed_millis as u64),
            serial_stats,
        )
    };

    let (parallel_dur, parallel_stats) = {
        let mut parallel_sim = crate::accelmain(traces_dir, parallel_config)?;
        // let start = Instant::now();
        parallel_sim.run()?;
        // let parallel_dur = start.elapsed();
        let parallel_stats = parallel_sim.stats().reduce();
        (
            Duration::from_millis(parallel_stats.sim.elapsed_millis as u64),
            parallel_stats,
        )
    };

    println!(
        "serial dur: {:?}, parallel dur: {:?} \t=> speedup {:>2.2}x",
        serial_dur,
        parallel_dur,
        serial_dur.as_secs_f64() / parallel_dur.as_secs_f64()
    );

    let max_mape_err = Some(0.1); // allow 10% difference
    let max_smape_err = Some(0.1); // allow 10% difference
    let max_mape_err = Some(0.05); // allow 5% difference
    let max_smape_err = Some(0.05); // allow 5% difference

    assert_stats_match(serial_stats, parallel_stats, max_mape_err, max_smape_err);
    Ok(())
}

pub fn cache_err(left: &stats::Cache, right: &stats::Cache) -> Vec<(String, PercentageError)> {
    left.union(&right)
        .map(|((alloc_id, access), (l, r))| {
            let access_name = match alloc_id {
                None => access.to_string(),
                Some(id) => format!("{id}@{access}"),
            };
            (access_name, percentage_error(l, r))
        })
        .filter(|(_, err)| *err != 0.0)
        .collect()
}

pub fn assert_stats_match(
    mut serial: stats::Stats,
    mut parallel: stats::Stats,
    max_mape_err: Option<f64>,
    max_smape_err: Option<f64>,
) {
    // use crate::cache::{AccessStat, AccessStatus};
    // use crate::cache::{AccessStat, AccessStatus};
    let max_mape_err = PercentageError::MAPE(max_mape_err.unwrap_or(0.0));
    let max_smape_err = PercentageError::SMAPE(max_smape_err.unwrap_or(0.0));

    serial.sim.elapsed_millis = 0;
    parallel.sim.elapsed_millis = 0;

    diff::diff!(serial: &serial.sim, parallel: &parallel.sim);
    diff::diff!(serial: &serial.accesses, parallel: &parallel.accesses);
    diff::diff!(serial: &serial.instructions, parallel: &parallel.instructions);
    let serial_dram = serial.dram.reduce().retain(|_, v| *v > 0);
    let parallel_dram = parallel.dram.reduce().retain(|_, v| *v > 0);
    diff::diff!(serial: &serial_dram, parallel: &parallel_dram);

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
    for cache_stats in [
        &mut serial_l1c_stats,
        &mut serial_l1t_stats,
        &mut serial_l1i_stats,
        &mut serial_l1d_stats,
        &mut serial_l2d_stats,
        &mut parallel_l1i_stats,
        &mut parallel_l1d_stats,
        &mut parallel_l1c_stats,
        &mut parallel_l1t_stats,
        &mut parallel_l2d_stats,
    ] {
        use stats::cache::{AccessStat, AccessStatus, RequestStatus};
        // combine hits and pending hits
        let pending_hits: std::collections::HashSet<_> = cache_stats.pending_hits().collect();
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

    diff::diff!(serial: &serial_l1c_stats, parallel: &parallel_l1c_stats);
    diff::diff!(serial: &serial_l1t_stats, parallel: &parallel_l1t_stats);
    diff::diff!(serial: &serial_l1i_stats, parallel: &parallel_l1i_stats);
    diff::diff!(serial: &serial_l1d_stats, parallel: &parallel_l1d_stats);
    diff::diff!(serial: &serial_l2d_stats, parallel: &parallel_l2d_stats);

    let l1i_err = cache_err(&serial_l1i_stats, &parallel_l1i_stats);
    dbg!(&l1i_err);
    assert!(l1i_err
        .iter()
        .all(|(_, err)| err <= &max_mape_err || err <= &max_smape_err));

    let l1c_err = cache_err(&serial_l1c_stats, &parallel_l1c_stats);
    dbg!(&l1c_err);
    assert!(l1c_err
        .iter()
        .all(|(_, err)| err <= &max_mape_err || err <= &max_smape_err));

    let l1t_err = cache_err(&serial_l1t_stats, &parallel_l1t_stats);
    dbg!(&l1t_err);
    assert!(l1t_err
        .iter()
        .all(|(_, err)| err <= &max_mape_err || err <= &max_smape_err));

    let l1d_err = cache_err(&serial_l1d_stats, &parallel_l1d_stats);
    dbg!(&l1d_err);
    assert!(l1d_err
        .iter()
        .all(|(_, err)| err <= &max_mape_err || err <= &max_smape_err));

    let l2d_err = cache_err(&serial_l2d_stats, &parallel_l2d_stats);
    dbg!(&l2d_err);
    assert!(l2d_err
        .iter()
        .all(|(_, err)| err <= &max_mape_err || err <= &max_smape_err));

    // TODO: compare dram, instructions, and accesses
    // (the same way as cache stats)

    let cycle_err = percentage_error(serial.sim.cycles, parallel.sim.cycles);
    dbg!(&cycle_err);
    assert!(cycle_err <= max_mape_err || cycle_err <= max_smape_err);

    let instructions_err = percentage_error(serial.sim.instructions, parallel.sim.instructions);
    dbg!(&instructions_err);
    assert!(instructions_err <= max_mape_err || instructions_err <= max_smape_err);

    let num_blocks_err = percentage_error(serial.sim.num_blocks, parallel.sim.num_blocks);
    dbg!(&num_blocks_err);
    assert!(num_blocks_err <= max_mape_err || num_blocks_err <= max_smape_err);
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
    vectoradd_32_10000_test: ("vectorAdd", { "dtype": 32, "length": 10000 }),
    vectoradd_32_20000_test: ("vectorAdd", { "dtype": 32, "length": 20000 }),
    vectoradd_64_10000_test: ("vectorAdd", { "dtype": 64, "length": 10000 }),
    vectoradd_64_20000_test: ("vectorAdd", { "dtype": 64, "length": 20000 }),

    // simple matrixmul
    simple_matrixmul_32_32_32_test: ("simple_matrixmul", { "m": 32, "n": 32, "p": 32 }),
    simple_matrixmul_32_32_64_test: ("simple_matrixmul", { "m": 32, "n": 32, "p": 64 }),
    simple_matrixmul_64_128_128_test: ("simple_matrixmul", { "m": 64, "n": 128, "p": 128 }),

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
    transpose_512_coalesed_test: ("transpose", { "dim": 256, "variant": "coalesced" }),

    // babelstream
    babelstream_1024_test: ("babelstream", { "size": 1024 }),
    babelstream_10240_test: ("babelstream", { "size": 10240 }),

    // extra tests for large input sizes
    // vectoradd_32_500000_test: ("vectorAdd", {
    //     "dtype": 32, "length": 500000, "memory_only": false, "cores_per_cluster": 4 }),

    // // vectoradd
    // vectoradd_100_test: ("vectorAdd", { "length": 100 }),
    // vectoradd_1000_test: ("vectorAdd", { "length": 1000 }),
    // vectoradd_10000_test: ("vectorAdd", { "length": 10000 }),
    //
    // // simple matrixmul
    // simple_matrixmul_32_32_32_test: ("simple_matrixmul", { "m": 32, "n": 32, "p": 32 }),
    // simple_matrixmul_32_32_64_test: ("simple_matrixmul", { "m": 32, "n": 32, "p": 64 }),
    // simple_matrixmul_64_128_128_test: ("simple_matrixmul", { "m": 64, "n": 128, "p": 128 }),
    //
    // // matrixmul (shared memory)
    // matrixmul_32_test: ("matrixmul", { "rows": 32 }),
    // matrixmul_64_test: ("matrixmul", { "rows": 64 }),
    // matrixmul_128_test: ("matrixmul", { "rows": 128 }),
    // matrixmul_256_test: ("matrixmul", { "rows": 256 }),
    //
    // // transpose
    // transpose_256_naive_test: ("transpose", { "dim": 256, "variant": "naive"}),
    // transpose_256_coalesed_test: ("transpose", { "dim": 256, "variant": "coalesced" }),
    // transpose_256_optimized_test: ("transpose", { "dim": 256, "variant": "optimized" }),
    //
    // // babelstream
    // babelstream_1024_test: ("babelstream", { "size": 1024 }),
}
