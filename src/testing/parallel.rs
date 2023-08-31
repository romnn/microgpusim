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
        num_simt_clusters: 20,                   // 20
        num_cores_per_simt_cluster: 4,           // 1
        num_schedulers_per_core: 2,              // 2
        num_memory_controllers: 8,               // 8
        num_sub_partition_per_memory_channel: 2, // 2
        fill_l2_on_memcopy: true,                // true
        ..config::GPU::default()
    });

    let box_interconn = Arc::new(ic::ToyInterconnect::new(
        box_config.num_simt_clusters,
        box_config.num_memory_controllers * box_config.num_sub_partition_per_memory_channel,
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

    let play_config = playground::Config::default();
    let start = Instant::now();
    let mut play_sim = playground::Accelsim::new(&play_config, &args)?;
    play_sim.run_to_completion();
    let play_dur = start.elapsed();

    println!(
        "play dur: {:?}, box dur: {:?} \t=> speedup {:>2.2}",
        play_dur,
        box_dur,
        play_dur.as_secs_f64() / box_dur.as_secs_f64()
    );
    let play_stats = play_sim.stats();
    let box_stats = box_sim.stats();

    let max_rel_err = Some(0.05); // allow 5% difference
    let abs_threshold = Some(10.0); // allow absolute difference of 10
    asserts::stats_match(play_stats, &box_stats, max_rel_err, abs_threshold, false);
    Ok(())
}

macro_rules! parallel_checks {
    ($($name:ident: $input:expr,)*) => {
        $(
            paste::paste! {
                #[test]
                fn [<nondeterministic_ $name>]() -> color_eyre::eyre::Result<()> {
                    $crate::testing::init_logging();
                    use validate::materialize::Benchmarks;

                    // load benchmark config
                    let (benchmark_name, input_idx) = $input;
                    let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));

                    let benchmarks_path = manifest_dir.join("test-apps/test-apps-materialized.yml");
                    let reader = utils::fs::open_readable(benchmarks_path)?;
                    let benchmarks = Benchmarks::from_reader(reader)?;
                    let bench_config = benchmarks.get_single_config(benchmark_name, input_idx).unwrap();

                    let box_trace_dir = &bench_config.trace.traces_dir;
                    let commands = box_trace_dir.join("commands.json");
                    let kernelslist = bench_config.accelsim_trace.traces_dir.join("box-kernelslist.g");
                    run(&box_trace_dir, &commands, &kernelslist)
                }
            }
        )*
    }
}

parallel_checks! {
    // vectoradd
    test_vectoradd_0: ("vectorAdd", 0),
    test_vectoradd_1: ("vectorAdd", 1),
    test_vectoradd_2: ("vectorAdd", 2),

    // simple matrixmul
    test_simple_matrixmul_0: ("simple_matrixmul", 0),
    test_simple_matrixmul_1: ("simple_matrixmul", 1),
    test_simple_matrixmul_17: ("simple_matrixmul", 17),

    // matrixmul (shared memory)
    test_matrixmul_0: ("matrixmul", 0),
    test_matrixmul_1: ("matrixmul", 1),
    test_matrixmul_2: ("matrixmul", 2),
    test_matrixmul_3: ("matrixmul", 3),

    // transpose
    test_transpose_0: ("transpose", 0),
    test_transpose_1: ("transpose", 1),
    test_transpose_2: ("transpose", 2),
}
