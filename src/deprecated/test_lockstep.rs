#[ignore = "deprecated"]
#[test]
fn test_lockstep() -> eyre::Result<()> {
    let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
    // let trace_dir = manifest_dir.join("results/vectorAdd/vectorAdd-100-32");
    // let trace_dir = manifest_dir.join("results/vectorAdd/vectorAdd-1000-32");
    // // this has a race condition: see WIP.md
    // let trace_dir = manifest_dir.join("results/vectorAdd/vectorAdd-10000-32");
    // // how many cycles does that have?
    // let trace_dir = manifest_dir.join("results/simple_matrixmul/simple_matrixmul-32-32-32-32");
    // // this fails in cycle 4654
    // let trace_dir = manifest_dir.join("results/simple_matrixmul/simple_matrixmul-32-32-64-32");
    let trace_dir = manifest_dir.join("results/simple_matrixmul/simple_matrixmul-64-128-128-32");
    let box_trace_dir = trace_dir.join("trace");
    let box_commands_path = box_trace_dir.join("commands.json");

    let kernelslist = trace_dir.join("accelsim-trace/kernelslist.g");
    let gpgpusim_config = manifest_dir.join("accelsim/gtx1080/gpgpusim.config");
    let trace_config = manifest_dir.join("accelsim/gtx1080/gpgpusim.trace.config");
    let inter_config = manifest_dir.join("accelsim/gtx1080/config_fermi_islip.icnt");

    assert!(trace_dir.is_dir());
    assert!(kernelslist.is_file());
    assert!(gpgpusim_config.is_file());
    assert!(trace_config.is_file());
    assert!(inter_config.is_file());

    // let start = std::time::Instant::now();
    // let box_stats = super::accelmain(&vec_add_trace_dir.join("trace"), None)?;

    // debugging config
    let mut box_config = config::GPUConfig::default();
    box_config.num_simt_clusters = 1;
    box_config.num_cores_per_simt_cluster = 1;
    box_config.num_schedulers_per_core = 1;
    let box_config = Arc::new(box_config);

    let box_interconn = Arc::new(ic::ToyInterconnect::new(
        box_config.num_simt_clusters,
        box_config.num_mem_units * box_config.num_sub_partition_per_memory_channel,
        // config.num_simt_clusters * config.num_cores_per_simt_cluster,
        // config.num_mem_units,
        Some(9), // found by printf debugging gpgusim
    ));

    let mut box_sim = super::MockSimulator::new(
        box_interconn,
        box_config.clone(),
        &box_trace_dir,
        &box_commands_path,
    );
    // let box_dur = start.elapsed();

    // let start = std::time::Instant::now();
    let mut args = vec![
        "-trace",
        kernelslist.as_os_str().to_str().unwrap(),
        "-config",
        gpgpusim_config.as_os_str().to_str().unwrap(),
        "-config",
        trace_config.as_os_str().to_str().unwrap(),
        "-inter_config_file",
        inter_config.as_os_str().to_str().unwrap(),
    ];
    dbg!(&args);

    let play_config = playground::Config::default();
    let mut play_sim = playground::Accelsim::new(&play_config, &args)?;

    // accelsim.run_to_completion();
    // let ref_stats = accelsim.stats().clone();
    // let ref_stats = playground::run(&config, &args)?;
    //
    let mut start = Instant::now();
    let mut play_time_cycle = std::time::Duration::ZERO;
    let mut play_time_other = std::time::Duration::ZERO;
    let mut box_time_cycle = std::time::Duration::ZERO;
    let mut box_time_other = std::time::Duration::ZERO;

    let mut last_valid_box_sim_state = None;
    let mut last_valid_play_sim_state = None;

    let mut cycle = 0;

    while play_sim.commands_left() || play_sim.kernels_left() {
        start = Instant::now();
        play_sim.process_commands();
        play_sim.launch_kernels();
        play_time_other += start.elapsed();

        start = Instant::now();
        box_sim.process_commands();
        box_sim.lauch_kernels();
        box_time_other += start.elapsed();

        let mut finished_kernel_uid: Option<u32> = None;
        loop {
            if !play_sim.active() {
                break;
            }

            start = Instant::now();
            play_sim.cycle();
            cycle = play_sim.get_cycle();
            play_time_cycle += start.elapsed();

            start = Instant::now();
            box_sim.cycle();
            box_sim.set_cycle(cycle);
            box_time_cycle += start.elapsed();

            // todo: extract also l1i ready (least important)
            // todo: extract wb pipeline

            // iterate over sub partitions
            let total_cores = box_sim.config.total_cores();
            let num_partitions = box_sim.mem_partition_units.len();
            let num_sub_partitions = box_sim.mem_sub_partitions.len();
            let mut box_sim_state =
                testing::state::Simulation::new(total_cores, num_partitions, num_sub_partitions);

            for (cluster_id, cluster) in box_sim.clusters.iter().enumerate() {
                for (core_id, core) in cluster.coreslock().iter().enumerate() {
                    let global_core_id =
                        cluster_id * box_sim.config.num_cores_per_simt_cluster + core_id;
                    assert_eq!(core.inner.core_id, global_core_id);

                    // this is the one we will use (unless the assertion is ever false)
                    let core_id = core.inner.core_id;

                    // core: functional units
                    for (fu_id, fu) in core.functional_units.iter().enumerate() {
                        let fu = fulock();
                        let issue_port = core.issue_ports[fu_id];
                        let issue_reg: super::register_set::RegisterSet = core.inner.pipeline_reg
                            [issue_port as usize]
                            .borrow()
                            .clone();
                        assert_eq!(issue_port, issue_reg.stage);

                        box_sim_state.functional_unit_pipelines[core_id].push(issue_reg.into());
                    }
                    // core: operand collector
                    box_sim_state.operand_collectors[core_id]
                        .insert(core.inner.operand_collector.borrow().deref().into());
                    // core: schedulers
                    box_sim_state.schedulers[core_id]
                        .extend(core.schedulers.iter().map(Into::into));
                }
            }

            for (partition_id, partition) in box_sim.mem_partition_units.iter().enumerate() {
                box_sim_state.dram_latency_queue[partition_id].extend(
                    partition
                        .dram_latency_queue
                        .clone()
                        .into_iter()
                        .map(Into::into),
                );
            }
            for (sub_id, sub) in box_sim.mem_sub_partitions.iter().enumerate() {
                for (dest_queue, src_queue) in [
                    (
                        &mut box_sim_state.interconn_to_l2_queue[sub_id],
                        &sub.borrow().interconn_to_l2_queue,
                    ),
                    (
                        &mut box_sim_state.l2_to_interconn_queue[sub_id],
                        &sub.borrow().l2_to_interconn_queue,
                    ),
                    (
                        &mut box_sim_state.l2_to_dram_queue[sub_id],
                        &sub.borrow().l2_to_dram_queuelock(),
                    ),
                    (
                        &mut box_sim_state.dram_to_l2_queue[sub_id],
                        &sub.borrow().dram_to_l2_queue,
                    ),
                ] {
                    dest_queue.extend(src_queue.clone().into_iter().map(Into::into));
                }
            }

            let mut play_sim_state =
                testing::state::Simulation::new(total_cores, num_partitions, num_sub_partitions);
            for (core_id, core) in play_sim.cores().enumerate() {
                for reg in core.register_sets().into_iter() {
                    play_sim_state.functional_unit_pipelines[core_id].push(reg.into());
                }
                // core: operand collector
                let coll = core.operand_collector();
                play_sim_state.operand_collectors[core_id].insert(coll.into());
                // core: scheduler units
                let schedulers = core.schedulers();
                assert_eq!(schedulers.len(), box_sim_state.schedulers[core_id].len());
                for (sched_idx, scheduler) in schedulers.into_iter().enumerate() {
                    // let scheduler = testing::state::Scheduler::from(scheduler);
                    play_sim_state.schedulers[core_id].push(scheduler.into());

                    let num_box_warps = box_sim_state.schedulers[core_id][sched_idx]
                        .prioritized_warp_ids
                        .len();
                    let num_play_warps = play_sim_state.schedulers[core_id][sched_idx]
                        .prioritized_warp_ids
                        .len();
                    let limit = num_box_warps.min(num_play_warps);

                    // fix: make sure we only compare what can be compared
                    box_sim_state.schedulers[core_id][sched_idx]
                        .prioritized_warp_ids
                        .split_off(limit);
                    play_sim_state.schedulers[core_id][sched_idx]
                        .prioritized_warp_ids
                        .split_off(limit);

                    assert_eq!(
                        box_sim_state.schedulers[core_id][sched_idx]
                            .prioritized_warp_ids
                            .len(),
                        play_sim_state.schedulers[core_id][sched_idx]
                            .prioritized_warp_ids
                            .len(),
                    );
                }
            }

            for (partition_id, partition) in play_sim.partition_units().enumerate() {
                play_sim_state.dram_latency_queue[partition_id]
                    .extend(partition.dram_latency_queue().into_iter().map(Into::into));
            }
            for (sub_id, sub) in play_sim.sub_partitions().enumerate() {
                play_sim_state.interconn_to_l2_queue[sub_id]
                    .extend(sub.interconn_to_l2_queue().into_iter().map(Into::into));
                play_sim_state.l2_to_interconn_queue[sub_id]
                    .extend(sub.l2_to_interconn_queue().into_iter().map(Into::into));
                play_sim_state.dram_to_l2_queue[sub_id]
                    .extend(sub.dram_to_l2_queue().into_iter().map(Into::into));
                play_sim_state.l2_to_dram_queue[sub_id]
                    .extend(sub.l2_to_dram_queue().into_iter().map(Into::into));
            }

            if box_sim_state != play_sim_state {
                println!(
                    "validated play state for cycle {}: {:#?}",
                    cycle - 1,
                    &last_valid_play_sim_state
                );

                dbg!(&box_sim.allocations);
                // 140284125003776
                // 140284124987392
                // 140284125020160
                // memory cycle for instruction LDG[pc=648,warp=32] => access: Access(GLOBAL_ACC_R@1+16384)
                // memory cycle for instruction: Some(OP_LDG[pc=648,warp=32]) => access: GLOBAL_ACC_R@1+0

                for (sub_id, sub) in play_sim.sub_partitions().enumerate() {
                    let play_icnt_l2_queue = sub
                        .interconn_to_l2_queue()
                        .iter()
                        .map(|fetch| fetch.get_addr())
                        .collect::<Vec<_>>();
                    dbg!(sub_id, play_icnt_l2_queue);
                }

                for (sub_id, sub) in box_sim.mem_sub_partitions.iter().enumerate() {
                    let box_icnt_l2_queue = sub
                        .borrow()
                        .interconn_to_l2_queue
                        .iter()
                        .map(|fetch| fetch.addr())
                        .collect::<Vec<_>>();
                    dbg!(sub_id, box_icnt_l2_queue);
                }
            }
            println!("checking for diff after cycle {}", cycle);
            diff::assert_eq!(&box_sim_state, &play_sim_state);

            last_valid_box_sim_state.insert(box_sim_state);
            last_valid_play_sim_state.insert(play_sim_state);

            finished_kernel_uid = play_sim.finished_kernel_uid();
            if finished_kernel_uid.is_some() {
                break;
            }
        }

        if let Some(uid) = finished_kernel_uid {
            play_sim.cleanup_finished_kernel(uid);
        }

        if play_sim.limit_reached() {
            println!("GPGPU-Sim: ** break due to reaching the maximum cycles (or instructions) **");
            std::io::stdout().flush()?;
            break;
        }
    }

    dbg!(box_time_cycle / u32::try_from(cycle).unwrap());
    dbg!(play_time_cycle / u32::try_from(cycle).unwrap());
    dbg!(&cycle);

    let play_stats = play_sim.stats().clone();
    let box_stats = box_sim.stats();
    // let playground_dur = start.elapsed();

    // dbg!(&play_stats);
    // dbg!(&box_stats);

    // dbg!(&playground_dur);
    // dbg!(&box_dur);

    // compare stats here
    diff::assert_eq!(
        &stats::PerCache(play_stats.l1i_stats.clone().convert()),
        &box_stats.l1i_stats
    );
    diff::assert_eq!(
        &stats::PerCache(play_stats.l1d_stats.clone().convert()),
        &box_stats.l1d_stats,
    );
    diff::assert_eq!(
        &stats::PerCache(play_stats.l1t_stats.clone().convert()),
        &box_stats.l1t_stats,
    );
    diff::assert_eq!(
        &stats::PerCache(play_stats.l1c_stats.clone().convert()),
        &box_stats.l1c_stats,
    );
    diff::assert_eq!(
        &stats::PerCache(play_stats.l2d_stats.clone().convert()),
        &box_stats.l2d_stats,
    );

    diff::assert_eq!(
        play_stats.accesses,
        playground::stats::Accesses::from(box_stats.accesses.clone())
    );

    // dbg!(&play_stats.accesses);
    // dbg!(&box_stats.accesses);
    //
    // dbg!(&play_stats.instructions);
    // dbg!(&box_stats.instructions);
    //
    // dbg!(&play_stats.sim);
    // dbg!(&box_stats.sim);

    let box_dram_stats = playground::stats::DRAM::from(box_stats.dram.clone());

    // dbg!(&play_stats.dram);
    // dbg!(&box_dram_stats);

    diff::assert_eq!(&play_stats.dram, &box_dram_stats);

    let playground_instructions =
        playground::stats::InstructionCounts::from(box_stats.instructions.clone());
    diff::assert_eq!(&play_stats.instructions, &playground_instructions);

    // dbg!(&play_stats.sim, &box_stats.sim);
    diff::assert_eq!(
        &play_stats.sim,
        &playground::stats::Sim::from(box_stats.sim.clone()),
    );

    // this uses our custom PartialEq::eq implementation
    assert_eq!(&play_stats, &box_stats);

    assert!(false, "all good!");
    Ok(())
}

// #[ignore = "todo"]
// #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
// async fn test_async_vectoradd() -> eyre::Result<()> {
//     let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
//     let vec_add_trace_dir = manifest_dir.join("results/vectorAdd/vectorAdd-100-32");
//
//     let kernelslist = vec_add_trace_dir.join("accelsim-trace/kernelslist.g");
//     let gpgpusim_config = manifest_dir.join("accelsim/gtx1080/gpgpusim.config");
//     let trace_config = manifest_dir.join("accelsim/gtx1080/gpgpusim.trace.config");
//     let inter_config = manifest_dir.join("accelsim/gtx1080/config_fermi_islip.icnt");
//
//     assert!(vec_add_trace_dir.is_dir());
//     assert!(kernelslist.is_file());
//     assert!(gpgpusim_config.is_file());
//     assert!(trace_config.is_file());
//     assert!(inter_config.is_file());
//
//     for _ in 0..10 {
//         let trace_dir = vec_add_trace_dir.join("trace");
//         let stats: stats::Stats = tokio::task::spawn_blocking(move || {
//             let stats = super::accelmain(trace_dir, None)?;
//             Ok::<_, eyre::Report>(stats)
//         })
//         .await??;
//
//         let handles = (0..1).map(|_| {
//             let kernelslist = kernelslist.clone();
//             let inter_config = inter_config.clone();
//             let trace_config = trace_config.clone();
//             let gpgpusim_config = gpgpusim_config.clone();
//
//             tokio::task::spawn_blocking(move || {
//                 // let kernelslist = kernelslist.to_string_lossy().to_string();
//                 // let gpgpusim_config = gpgpusim_config.to_string_lossy().to_string();
//                 // let trace_config = trace_config.to_string_lossy().to_string();
//                 // let inter_config = inter_config.to_string_lossy().to_string();
//                 //
//                 // let mut args = vec![
//                 //     "-trace",
//                 //     &kernelslist,
//                 //     "-config",
//                 //     &gpgpusim_config,
//                 //     "-config",
//                 //     &trace_config,
//                 //     "-inter_config_file",
//                 //     &inter_config,
//                 // ];
//
//                 let mut args = vec![
//                     "-trace",
//                     kernelslist.as_os_str().to_str().unwrap(),
//                     "-config",
//                     gpgpusim_config.as_os_str().to_str().unwrap(),
//                     "-config",
//                     trace_config.as_os_str().to_str().unwrap(),
//                     "-inter_config_file",
//                     inter_config.as_os_str().to_str().unwrap(),
//                 ];
//                 dbg!(&args);
//
//                 let config = playground::Config::default();
//                 let ref_stats = playground::run(&config, &args)?;
//                 Ok::<_, eyre::Report>(ref_stats)
//             })
//         });
//
//         // wait for all to complete
//         let ref_stats: Vec<Result<Result<_, _>, _>> = futures::future::join_all(handles).await;
//         let ref_stats: Result<Vec<Result<_, _>>, _> = ref_stats.into_iter().collect();
//         let ref_stats: Result<Vec<_>, _> = ref_stats?.into_iter().collect();
//         let ref_stats: Vec<_> = ref_stats?;
//
//         let ref_stats = ref_stats[0].clone();
//         dbg!(&ref_stats);
//     }
//
//     Ok(())
// }
