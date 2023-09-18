use super::asserts;
use crate::sync::Arc;
use crate::{config, interconn as ic};
use color_eyre::eyre;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub fn run(
    box_trace_dir: &Path,
    box_commands_path: &Path,
    accelsim_kernelslist_path: &Path,
) -> eyre::Result<()> {
    dbg!(&box_commands_path);
    dbg!(&accelsim_kernelslist_path);

    let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
    let gpgpusim_config = manifest_dir.join("accelsim/gtx1080/gpgpusim.config");
    let trace_config = manifest_dir.join("accelsim/gtx1080/gpgpusim.trace.config");
    let inter_config = manifest_dir.join("accelsim/gtx1080/config_fermi_islip.icnt");

    assert!(box_trace_dir.is_dir());
    assert!(box_commands_path.is_file());
    assert!(accelsim_kernelslist_path.is_file());
    assert!(gpgpusim_config.is_file());
    assert!(trace_config.is_file());
    assert!(inter_config.is_file());

    // debugging config
    let box_config = Arc::new(config::GPU {
        num_simt_clusters: 20,                       // 20
        num_cores_per_simt_cluster: 4,               // 1
        num_schedulers_per_core: 2,                  // 2
        num_memory_controllers: 8,                   // 8
        num_sub_partitions_per_memory_controller: 2, // 2
        fill_l2_on_memcopy: true,                    // true
        ..config::GPU::default()
    });

    let box_interconn = Arc::new(ic::ToyInterconnect::new(
        box_config.num_simt_clusters,
        box_config.total_sub_partitions(),
    ));

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

    // box_sim.run_to_completion_parallel_nondeterministic(1)?;
    box_sim.run_to_completion_parallel_nondeterministic(run_ahead)?;
    let box_dur = start.elapsed();

    let args = vec![
        "-trace",
        accelsim_kernelslist_path.as_os_str().to_str().unwrap(),
        "-config",
        gpgpusim_config.as_os_str().to_str().unwrap(),
        "-config",
        trace_config.as_os_str().to_str().unwrap(),
        "-inter_config_file",
        inter_config.as_os_str().to_str().unwrap(),
    ];
    dbg!(&args);

    let play_config = playground::Config {
        accelsim_compat_mode: false,
        ..playground::Config::default()
    };
    let start = Instant::now();
    let mut play_sim = playground::Accelsim::new(play_config, &args)?;
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

macro_rules! parallel_checks {
    ($($name:ident: ($bench_name:expr, $($input:tt)+),)*) => {
        $(
            paste::paste! {
                #[test]
                fn [<nondeterministic_ $name>]() -> color_eyre::eyre::Result<()> {
                    use validate::{
                        Target,
                        benchmark,
                        materialized::TargetBenchmarkConfig,
                    };
                    $crate::testing::init_test();

                    let input: benchmark::Input = validate::input!($($input)+)?;
                    let bench_config = benchmark::find_exact(
                        Target::Simulate, $bench_name, &input)?;

                    let TargetBenchmarkConfig::Simulate {
                        ref traces_dir ,
                        ref accelsim_traces_dir,
                        ..
                    } = bench_config.target_config else {
                        unreachable!();
                    };

                    // let box_trace_dir = &bench_config.trace.traces_dir;
                    let commands = traces_dir.join("commands.json");
                    // let kernelslist = bench_config.accelsim_trace.traces_dir.join("box-kernelslist.g");
                    let kernelslist = accelsim_traces_dir.join("box-kernelslist.g");
                    run(traces_dir, &commands, &kernelslist)
                }
            }
        )*
    }
}

parallel_checks! {
    // vectoradd
    vectoradd_100_test: ("vectorAdd", { "length": 100 }),
    vectoradd_1000_test: ("vectorAdd", { "length": 1000 }),
    vectoradd_10000_test: ("vectorAdd", { "length": 10000 }),

    // simple matrixmul
    simple_matrixmul_32_32_32_test: ("simple_matrixmul", { "m": 32, "n": 32, "p": 32 }),
    simple_matrixmul_32_32_64_test: ("simple_matrixmul", { "m": 32, "n": 32, "p": 64 }),
    simple_matrixmul_64_128_128_test: ("simple_matrixmul", { "m": 64, "n": 128, "p": 128 }),

    // matrixmul (shared memory)
    matrixmul_32_test: ("matrixmul", { "rows": 32 }),
    matrixmul_64_test: ("matrixmul", { "rows": 64 }),
    matrixmul_128_test: ("matrixmul", { "rows": 128 }),
    matrixmul_256_test: ("matrixmul", { "rows": 256 }),

    // transpose
    transpose_256_naive_test: ("transpose", { "dim": 256, "variant": "naive"}),
    transpose_256_coalesed_test: ("transpose", { "dim": 256, "variant": "coalesced" }),
    transpose_256_optimized_test: ("transpose", { "dim": 256, "variant": "optimized" }),

    // babelstream
    babelstream_1024_test: ("babelstream", { "size": 1024 }),
}
