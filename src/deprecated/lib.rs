// fn get_bench_config(benchmark_name: &str, input_idx: usize) -> eyre::Result<BenchmarkConfig> {
//     use std::path::PathBuf;
//
//     let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
//     let benchmarks_path = manifest_dir.join("test-apps/test-apps-materialized.yml");
//     let reader = utils::fs::open_readable(benchmarks_path)?;
//     let benchmarks = Benchmarks::from_reader(reader)?;
//     let bench_config = benchmarks
//         .get_single_config(benchmark_name, input_idx)
//         .ok_or_else(|| {
//             eyre::eyre!(
//                 "no benchmark {:?} or input index {}",
//                 benchmark_name,
//                 input_idx
//             )
//         })?;
//     Ok(bench_config.clone())
// }
//
// #[test]
// fn test_nondet() -> eyre::Result<()> {
//     crate::testing::init_test();
//     let (bench_name, input_num) = ("transpose", 0); // takes 34 sec (accel same)
//     println!("running {bench_name}@{input_num}");
//
//     let mut bench_config = get_bench_config(bench_name, input_num)?;
//
//     let start = Instant::now();
//     bench_config.simulate.parallel = true;
//     // bench_config.parallelization = config::Parallelization::Nondeterministic(2);
//     let sim_parallel = validate::simulate::simulate_bench_config(&bench_config)?;
//     println!("parallel took {:?}", start.elapsed());
//
//     let start = Instant::now();
//     bench_config.simulate.parallel = false;
//     // bench_config.parallelization = config::Parallelization::Nondeterministic(2);
//     let sim_serial = validate::simulate::simulate_bench_config(&bench_config)?;
//     println!("serial took {:?}", start.elapsed());
//
//     let _parallel_stats = sim_parallel.stats();
//     let _serial_stats = sim_serial.stats();
//
//     // sim_serial.states.sort_by_key(|(cycle, _)| *cycle);
//     // sim_parallel.states.sort_by_key(|(cycle, _)| *cycle);
//     // assert_eq!(sim_serial.states.len(), sim_parallel.states.len());
//
//     // diff::diff!(serial: serial_stats.sim, parallel: parallel_stats.sim);
//     // diff::diff!(
//     //     serial: &serial_stats.l2d_stats.reduce(),
//     //     parallel: &parallel_stats.l2d_stats.reduce(),
//     // );
//
//     // for ((ser_cycle, ser_state), (par_cycle, par_state)) in
//     //     sim_serial.states.iter().zip(sim_parallel.states.iter())
//     // {
//     //     assert_eq!(ser_cycle, par_cycle);
//     //     dbg!(ser_cycle);
//     //     diff::assert_eq!(serial: ser_state, parallel: par_state);
//     // }
//
//     // diff::assert_eq!(serial: sim_serial.states, parallel: sim_parallel.states);
//
//     Ok(())
// }
