use color_eyre::eyre;
use std::io::Write;
use utils::diff;

async fn validate_playground_accelsim_compat(
    bench_config: &validate::materialize::BenchmarkConfig,
    sim_config: &accelsim::SimConfig,
) -> eyre::Result<()> {
    let traces_dir = &bench_config.accelsim_trace.traces_dir;
    let kernelslist = traces_dir.join("kernelslist.g");

    dbg!(&traces_dir);
    dbg!(&kernelslist);
    dbg!(&sim_config);

    let parse_options = accelsim::parser::Options::default();
    let timeout = None;

    // run accelsim
    let (accelsim_stdout, accelsim_stderr, accelsim_stats) = {
        let extra_sim_args: &[String] = &[];
        let stream_output = false;
        let use_upstream = false;
        let (output, accelsim_dur) = accelsim_sim::simulate_trace(
            &traces_dir,
            &kernelslist,
            sim_config,
            timeout,
            extra_sim_args,
            stream_output,
            use_upstream,
        )
        .await?;
        dbg!(&accelsim_dur);

        let stdout = utils::decode_utf8!(output.stdout);
        let stderr = utils::decode_utf8!(output.stderr);
        let log_reader = std::io::Cursor::new(&output.stdout);
        let stats = accelsim::parser::parse_stats(log_reader, &parse_options)?;
        (stdout, stderr, stats)
    };
    // dbg!(&accelsim_stats);

    // run playground in accelsim compat mode
    let (playground_log, playground_stats) = {
        let extra_args: &[String] = &[];
        let accelsim_compat_mode = true;
        let (log, _stats, playground_dur) = validate::playground::simulate_bench_config(
            bench_config,
            validate::TraceProvider::Native,
            extra_args,
            accelsim_compat_mode,
        )?;
        dbg!(&playground_dur);

        let log_reader = std::io::Cursor::new(&log);
        let parse_options = accelsim::parser::Options::default();
        let stats = accelsim::parser::parse_stats(log_reader, &parse_options)?;
        (log, stats)
    };
    // dbg!(&playground_stats);

    let filter_func =
        |((_name, _kernel, stat_name), _value): &((String, u16, String), f64)| -> bool {
            // we ignore rates and other stats that can vary per run
            !matches!(
                stat_name.as_str(),
                "gpgpu_silicon_slowdown"
                    | "gpgpu_simulation_rate"
                    | "gpgpu_simulation_time_sec"
                    | "gpu_ipc"
                    | "gpu_occupancy"
                    | "gpu_tot_ipc"
                    | "l1_inst_cache_total_miss_rate"
                    | "l2_bandwidth_gbps"
            )
        };

    let cmp_play_stats: accelsim::Stats =
        playground_stats.into_iter().filter(filter_func).collect();

    let cmp_accel_stats: accelsim::Stats = accelsim_stats
        .clone()
        .into_iter()
        .filter(filter_func)
        .collect();

    for stat in ["warp_instruction_count", "gpu_tot_sim_cycle"] {
        println!(
            "{:>15}:\t play={:.1}\t accel={:.1}",
            stat,
            cmp_play_stats.find_stat(stat).copied().unwrap_or_default(),
            cmp_accel_stats.find_stat(stat).copied().unwrap_or_default(),
        );
    }
    // diff::assert_eq!(
    //     play: cmp_play_stats.find_stat("warp_instruction_count"),
    //     accelsim: cmp_accel_stats.find_stat("warp_instruction_count"),
    // );
    // diff::assert_eq!(
    //     play: cmp_play_stats.find_stat("gpu_tot_sim_cycle"),
    //     accelsim: cmp_accel_stats.find_stat("gpu_tot_sim_cycle"),
    // );

    {
        // save the logs
        utils::fs::open_writable(traces_dir.join("debug.playground.stdout"))?
            .write_all(playground_log.as_bytes())?;
        utils::fs::open_writable(traces_dir.join("debug.accelsim.stdout"))?
            .write_all(accelsim_stdout.as_bytes())?;
        utils::fs::open_writable(traces_dir.join("debug.accelsim.stderr"))?
            .write_all(accelsim_stderr.as_bytes())?;
    }

    // diff::assert_eq!(play: playground_stdout, accelsim: accelsim_stdout);
    diff::assert_eq!(play: cmp_play_stats, accelsim: cmp_accel_stats);
    // assert!(false);
    Ok(())
}

macro_rules! accelsim_compat_tests {
    ($($name:ident: ($bench_name:expr, $($input:tt)+),)*) => {
        $(
            #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
            async fn $name() -> color_eyre::eyre::Result<()> {
                $crate::testing::init_test();
                let bench_config = super::find_bench_config($bench_name, validate::input!($($input)+))?;
                let validate::materialize::AccelsimSimConfigFiles {
                    config,
                    config_dir,
                    trace_config,
                    inter_config,
                } = bench_config.accelsim_simulate.configs.clone();

                let sim_config = accelsim::SimConfig {
                    config: Some(config),
                    config_dir: Some(config_dir),
                    trace_config: Some(trace_config),
                    inter_config: Some(inter_config),
                };

                validate_playground_accelsim_compat(&bench_config, &sim_config).await
            }
        )*
    }
}

accelsim_compat_tests! {
    // vectoradd
    accelsim_compat_vectoradd_100_test: ("vectorAdd", { "length": 100 }),
    accelsim_compat_vectoradd_1000_test: ("vectorAdd", { "length": 1000 }),
    accelsim_compat_vectoradd_10000_test: ("vectorAdd", { "length": 10000 }),

    // simple matrixmul
    accelsim_compat_simple_matrixmul_32_32_32_test:
        ("simple_matrixmul", { "m": 32, "n": 32, "p": 32 }),
    accelsim_compat_simple_matrixmul_32_32_64_test:
        ("simple_matrixmul", { "m": 32, "n": 32, "p": 64 }),
    accelsim_compat_simple_matrixmul_64_128_128_test:
        ("simple_matrixmul", { "m": 64, "n": 128, "p": 128 }),

    // matrixmul (shared memory)
    accelsim_compat_matrixmul_32_test: ("matrixmul", { "rows": 32 }),
    accelsim_compat_matrixmul_64_test: ("matrixmul", { "rows": 64 }),
    accelsim_compat_matrixmul_128_test: ("matrixmul", { "rows": 128 }),
    accelsim_compat_matrixmul_256_test: ("matrixmul", { "rows": 256 }),

    // transpose
    accelsim_compat_transpose_256_naive_test: ("transpose", { "dim": 256, "variant": "naive"}),
    accelsim_compat_transpose_256_coalesed_test: ("transpose", { "dim": 256, "variant": "coalesced" }),
    accelsim_compat_transpose_256_optimized_test: ("transpose", { "dim": 256, "variant": "optimized" }),

    // babelstream
    accelsim_compat_babelstream_1024_test: ("babelstream", { "size": 1024 }),
}
