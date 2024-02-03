use super::asserts;
use crate::{
    cache, config, func_unit::SimdFunctionUnit, interconn as ic, mem_fetch, register_set, testing,
};
use color_eyre::eyre;
use pretty_assertions_sorted as full_diff;
use trace_model::ToBitString;
use utils::diff;
use validate::{
    materialized::{BenchmarkConfig, TargetBenchmarkConfig},
    TraceProvider,
};

use std::collections::HashSet;
use std::io::Write;
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

type IC = ic::SimpleInterconnect<ic::Packet<mem_fetch::MemFetch>>;
type MCU = crate::mcu::PascalMemoryControllerUnit;

// #[inline]
fn gather_simulation_state(
    box_sim: &mut crate::Simulator<IC, MCU>,
    // box_sim: &mut validate::simulate::config::GTX1080,
    play_sim: &mut playground::Accelsim,
    box_cycle: u64,
    _play_cycle: u64,
) -> (testing::state::Simulation, testing::state::Simulation) {
    let num_schedulers = box_sim.config.num_schedulers_per_core;
    let num_clusters = box_sim.config.num_simt_clusters;
    let cores_per_cluster = box_sim.config.num_cores_per_simt_cluster;
    assert_eq!(
        box_sim.config.total_cores(),
        num_clusters * cores_per_cluster
    );

    let num_partitions = box_sim.mem_partition_units.len();
    // let num_sub_partitions = box_sim.mem_sub_partitions.len();
    let num_sub_partitions = box_sim.config.total_sub_partitions();

    let l1_config = box_sim.config.data_cache_l1.as_ref();
    let num_l1_banks = l1_config.map(|l1| l1.l1_banks).unwrap_or(0);
    let l1_latency = l1_config.map(|l1| l1.l1_latency).unwrap_or(0);

    assert!(num_l1_banks > 0);
    let mut box_sim_state = testing::state::Simulation::new(
        num_clusters,
        cores_per_cluster,
        num_partitions,
        num_sub_partitions,
        num_schedulers,
    );

    box_sim_state.last_cluster_issue = *box_sim.last_cluster_issue.lock();

    for (cluster_id, cluster) in box_sim.clusters.iter().enumerate() {
        // for (core_id, core) in cluster.cores.iter().enumerate() {
        for core in cluster.cores.iter() {
            // let core = core.try_read();
            // let global_core_id = cluster_id * box_sim.config.num_cores_per_simt_cluster + core_id;
            // assert_eq!(core.core_id, global_core_id);

            // this is the one we will use (unless the assertion is ever false)
            // let core_id = core.core_id;
            let global_core_id = core.global_core_id;

            let load_store_unit = &core.load_store_unit;
            let functional_units_iter = core
                .functional_units
                .iter()
                .map(|fu| fu.as_ref() as &dyn SimdFunctionUnit)
                .chain(std::iter::once(load_store_unit as &dyn SimdFunctionUnit));

            // core: functional units
            for fu in functional_units_iter {
                let issue_port = fu.issue_port();
                let issue_reg: register_set::RegisterSet =
                    core.pipeline_reg[issue_port as usize].clone();
                assert_eq!(issue_port, issue_reg.stage);

                box_sim_state.functional_unit_pipelines_per_core[global_core_id]
                    .push(issue_reg.into());
            }

            let functional_units_iter = core
                .functional_units
                .iter()
                .map(|fu| fu.as_ref() as &dyn SimdFunctionUnit)
                .chain(std::iter::once(load_store_unit as &dyn SimdFunctionUnit));

            for fu in functional_units_iter {
                box_sim_state.functional_unit_pipelines_per_core[global_core_id].push(
                    testing::state::RegisterSet {
                        name: fu.id().to_string(),
                        pipeline: fu
                            .pipeline()
                            .iter()
                            .map(|reg| reg.clone().map(Into::into))
                            .collect(),
                    },
                );
                box_sim_state.functional_unit_occupied_slots_per_core[global_core_id] =
                    fu.occupied().to_bit_string();
            }
            // core: operand collector
            let register_file =
                testing::state::OperandCollector::new(&core.register_file, &core.pipeline_reg);
            box_sim_state.operand_collector_per_core[global_core_id] = Some(register_file);
            // core: schedulers
            box_sim_state.scheduler_per_core[global_core_id] = core
                .schedulers
                .iter()
                .map(|scheduler| scheduler.deref().into())
                .collect();

            // core: pending register writes
            box_sim_state.pending_register_writes_per_core[global_core_id] = load_store_unit
                .pending_writes
                .clone()
                .into_iter()
                .flat_map(|(warp_id, pending_registers)| {
                    pending_registers
                        .into_iter()
                        .map(
                            move |(reg_num, pending)| testing::state::PendingRegisterWrites {
                                warp_id,
                                reg_num,
                                pending,
                            },
                        )
                })
                .collect();

            box_sim_state.pending_register_writes_per_core[global_core_id].sort();

            box_sim_state.l1_latency_queue_per_core[global_core_id] = load_store_unit
                .l1_latency_queue
                .iter()
                .enumerate()
                .map(|(bank_id, new_latency_queue)| {
                    // let mut latency_queue = vec![None; l1_latency];
                    // for (ready_cycle, fetch) in new_latency_queue.clone() {
                    //     //
                    //     let baseline_cycle = box_cycle.max(
                    //         new_latency_queue
                    //             .front()
                    //             .map(|(cycle, _)| *cycle)
                    //             .unwrap_or(0),
                    //     );
                    //     let remaining = ready_cycle.saturating_sub(box_cycle);
                    //     dbg!(ready_cycle, box_cycle, remaining);
                    //
                    //     assert!(remaining <= l1_latency as u64);
                    //     // if the slot is "l1 latency" from now, it should
                    //     // be all the way at the back
                    //     // let slot = (l1_latency - 1) - remaining as usize;
                    //     let slot = remaining as usize;
                    //     assert!(slot < l1_latency);
                    //
                    //     latency_queue[slot] = Some(fetch.clone().into());
                    // }
                    // (bank_id, latency_queue)

                    let latency_queue = new_latency_queue
                        .iter()
                        .map(|fetch| fetch.clone().map(Into::into))
                        .collect();
                    (bank_id, latency_queue)
                })
                .collect::<Vec<_>>();

            if let Some(l1_data_cache) = load_store_unit.data_l1.as_ref() {
                let l1_data_cache = l1_data_cache
                    .as_any()
                    .downcast_ref::<cache::Data<
                        MCU,
                        cache::controller::pascal::L1DataCacheController,
                        stats::cache::PerKernel,
                    >>()
                    .unwrap();

                let l1_data_cache = testing::state::Cache::from(&l1_data_cache.inner.tag_array);
                box_sim_state.l1_cache_per_core[global_core_id] = Some(l1_data_cache);
            }
        }
    }

    for (partition_id, partition) in box_sim.mem_partition_units.iter().enumerate() {
        // let partition = partition.try_read();
        box_sim_state.dram_latency_queue_per_partition[partition_id].extend(
            partition
                .dram_latency_queue
                .iter()
                .map(|(_, fetch)| fetch)
                .cloned()
                .map(Into::into),
        );

        let arbiter: &crate::arbitration::ArbitrationUnit =
            partition.arbiter.as_any().downcast_ref().unwrap();
        box_sim_state.dram_arbitration_per_partition[partition_id] = testing::state::Arbitration {
            last_borrower: partition.arbiter.last_borrower(),
            shared_credit: arbiter.shared_credit,
            private_credit: arbiter.private_credit.clone().into(),
        };
    }
    for partition in box_sim.mem_partition_units.iter() {
        for mem_sub in partition.sub_partitions.iter() {
            // for (sub_id, sub) in box_sim.mem_sub_partitions.iter().enumerate() {
            // let sub = sub.try_lock();

            let sub_id = mem_sub.global_id;
            let l2_cache = mem_sub.l2_cache.as_ref().unwrap();
            let l2_cache: &cache::DataL2<MCU> = l2_cache.as_any().downcast_ref().unwrap();

            box_sim_state.l2_cache_per_sub[sub_id] = Some((&l2_cache.inner.inner.tag_array).into());

            box_sim_state.rop_queue_per_sub[sub_id] = mem_sub
                .rop_queue
                .clone()
                .into_iter()
                .map(|(ready, fetch)| (ready, fetch.into()))
                .collect();

            for (dest_queue, src_queue) in [
                (
                    &mut box_sim_state.interconn_to_l2_queue_per_sub[sub_id],
                    &mem_sub.interconn_to_l2_queue,
                ),
                (
                    &mut box_sim_state.l2_to_interconn_queue_per_sub[sub_id],
                    &mem_sub.l2_to_interconn_queue,
                ),
                (
                    &mut box_sim_state.l2_to_dram_queue_per_sub[sub_id],
                    // &mem_sub.l2_to_dram_queue.lock(),
                    &mem_sub.l2_to_dram_queue,
                ),
                (
                    &mut box_sim_state.dram_to_l2_queue_per_sub[sub_id],
                    &mem_sub.dram_to_l2_queue,
                ),
            ] {
                *dest_queue = src_queue
                    .clone()
                    .into_iter()
                    .map(ic::Packet::into_inner)
                    .map(Into::into)
                    .collect();
            }
        }
    }

    let mut play_sim_state = testing::state::Simulation::new(
        num_clusters,
        cores_per_cluster,
        num_partitions,
        num_sub_partitions,
        num_schedulers,
    );

    play_sim_state.last_cluster_issue = play_sim.last_cluster_issue() as usize;

    for (core_id, core) in play_sim.cores().enumerate() {
        for regs in core.functional_unit_issue_register_sets() {
            play_sim_state.functional_unit_pipelines_per_core[core_id].push(regs.into());
        }
        let valid_units: HashSet<_> = box_sim_state.functional_unit_pipelines_per_core[core_id]
            .iter()
            .map(|fu| fu.name.clone())
            .collect();

        for regs in core
            .functional_unit_simd_pipeline_register_sets()
            .into_iter()
            .filter(|fu| valid_units.contains(&fu.name()))
        {
            play_sim_state.functional_unit_pipelines_per_core[core_id].push(regs.into());
        }

        for occupied in core.functional_unit_occupied_slots() {
            play_sim_state.functional_unit_occupied_slots_per_core[core_id] = occupied;
        }

        // core: pending register writes
        play_sim_state.pending_register_writes_per_core[core_id] = core
            .pending_register_writes()
            .into_iter()
            .map(Into::into)
            .collect();
        play_sim_state.pending_register_writes_per_core[core_id].sort();

        play_sim_state.l1_latency_queue_per_core[core_id] = core
            .l1_bank_latency_queue()
            .into_iter()
            .enumerate()
            .map(|(bank_id, latency_queue)| {
                (
                    bank_id,
                    latency_queue
                        .into_iter()
                        .map(|fetch| fetch.map(Into::into))
                        .collect(),
                )
            })
            .collect::<Vec<_>>();

        // core: operand collector
        let coll = core.operand_collector();
        play_sim_state.operand_collector_per_core[core_id] = Some(coll.into());
        // core: scheduler units
        let schedulers = core.schedulers();
        assert_eq!(
            schedulers.len(),
            box_sim_state.scheduler_per_core[core_id].len()
        );

        for (sched_idx, play_scheduler) in schedulers.into_iter().enumerate() {
            play_sim_state.scheduler_per_core[core_id][sched_idx] = play_scheduler.into();

            // let box_sched = &mut box_sim_state.scheduler_per_core[core_id][sched_idx];
            // let play_sched = &mut play_sim_state.scheduler_per_core[core_id][sched_idx];
            //
            // let num_box_warps = box_sched.prioritized_warp_ids.len();
            // let num_play_warps = play_sched.prioritized_warp_ids.len();
            // let limit = num_box_warps.min(num_play_warps);
            //
            // // make sure we only compare what can be compared
            // box_sched.prioritized_warp_ids.split_off(limit);
            // play_sched.prioritized_warp_ids.split_off(limit);
            //
            // assert_eq!(
            //     box_sched.prioritized_warp_ids.len(),
            //     play_sched.prioritized_warp_ids.len(),
            // );
        }

        // play_sim_state.l1_bank_latency_queue_per_core[core_id] = Some(testing::state::Cache {
        //     lines: core
        //         .l1_data_cache()
        //         .lines()
        //         .into_iter()
        //         .map(Into::into)
        //         .collect(),
        // });

        play_sim_state.l1_cache_per_core[core_id] = Some(testing::state::Cache {
            lines: core
                .l1_data_cache()
                .lines()
                .into_iter()
                .map(Into::into)
                .collect(),
        });
    }

    let mut partitions_added = 0;
    for (partition_id, partition) in play_sim.partition_units().enumerate() {
        assert!(partition_id < num_partitions);
        play_sim_state.dram_latency_queue_per_partition[partition_id] = partition
            .dram_latency_queue()
            .into_iter()
            .map(Into::into)
            .collect();
        partitions_added += 1;

        play_sim_state.dram_arbitration_per_partition[partition_id] = testing::state::Arbitration {
            last_borrower: partition.last_borrower(),
            shared_credit: partition.shared_credit(),
            private_credit: partition.private_credit().into(),
        };
    }
    assert_eq!(partitions_added, num_partitions);
    assert_eq!(
        play_sim.sub_partitions().count(),
        box_sim.config.total_sub_partitions()
    );

    let mut sub_partitions_added = 0;
    for (sub_id, sub) in play_sim.sub_partitions().enumerate() {
        play_sim_state.interconn_to_l2_queue_per_sub[sub_id] = sub
            .interconn_to_l2_queue()
            .into_iter()
            .map(Into::into)
            .collect();
        play_sim_state.l2_to_interconn_queue_per_sub[sub_id] = sub
            .l2_to_interconn_queue()
            .into_iter()
            .map(Into::into)
            .collect();
        play_sim_state.dram_to_l2_queue_per_sub[sub_id] =
            sub.dram_to_l2_queue().into_iter().map(Into::into).collect();
        play_sim_state.l2_to_dram_queue_per_sub[sub_id] =
            sub.l2_to_dram_queue().into_iter().map(Into::into).collect();

        play_sim_state.rop_queue_per_sub[sub_id] = sub
            .rop_delay_queue()
            .into_iter()
            .map(|(ready, fetch)| (ready, fetch.into()))
            .collect();

        play_sim_state.l2_cache_per_sub[sub_id] = Some(testing::state::Cache {
            lines: sub
                .l2_data_cache()
                .lines()
                .into_iter()
                .map(Into::into)
                .collect(),
        });

        sub_partitions_added += 1;
    }
    assert_eq!(sub_partitions_added, num_sub_partitions);
    (box_sim_state, play_sim_state)
}

pub fn run(bench_config: &BenchmarkConfig, trace_provider: TraceProvider) -> eyre::Result<()> {
    use accelsim::tracegen;
    let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));

    let TargetBenchmarkConfig::Simulate {
        ref traces_dir,
        ref accelsim_traces_dir,
        ..
    } = bench_config.target_config
    else {
        unreachable!();
    };

    let box_traces_dir = traces_dir;
    utils::fs::create_dirs(box_traces_dir)?;
    utils::fs::create_dirs(accelsim_traces_dir)?;

    let native_box_commands_path = box_traces_dir.join("commands.json");
    let native_accelsim_kernelslist_path = accelsim_traces_dir.join("kernelslist.g");

    let (box_commands_path, accelsim_kernelslist_path) = match trace_provider {
        TraceProvider::Native => {
            // use native traces
            (native_box_commands_path, native_accelsim_kernelslist_path)
        }
        TraceProvider::Accelsim => {
            let generated_box_commands_path =
                tracegen::convert_accelsim_to_box_traces(&tracegen::Conversion {
                    native_commands_path: &native_accelsim_kernelslist_path,
                    box_traces_dir,
                    accelsim_traces_dir,
                })?;
            (
                generated_box_commands_path,
                native_accelsim_kernelslist_path,
            )
        }
        TraceProvider::Box => {
            let generated_kernelslist_path =
                tracegen::convert_box_to_accelsim_traces(&tracegen::Conversion {
                    native_commands_path: &native_box_commands_path,
                    box_traces_dir,
                    accelsim_traces_dir,
                })?;
            (native_box_commands_path, generated_kernelslist_path)
        }
    };

    dbg!(&box_commands_path);
    dbg!(&accelsim_kernelslist_path);

    let gpgpusim_config = manifest_dir.join("accelsim/gtx1080/gpgpusim.config");
    let trace_config = manifest_dir.join("accelsim/gtx1080/gpgpusim.trace.config");
    let inter_config = manifest_dir.join("accelsim/gtx1080/config_pascal_islip.icnt");

    assert!(box_traces_dir.is_dir());
    assert!(box_commands_path.is_file());
    assert!(accelsim_kernelslist_path.is_file());
    assert!(gpgpusim_config.is_file());
    assert!(trace_config.is_file());
    assert!(inter_config.is_file());

    let input: config::Input = config::parse_input(&bench_config.values)?;
    dbg!(&input);

    let mut box_config: config::GPU = config::gtx1080::build_config(&input)?;
    box_config.fill_l2_on_memcopy = false;
    box_config.perfect_inst_const_cache = true;
    // box_config.flush_l1_cache = true;
    // box_config.flush_l2_cache = false;
    box_config.accelsim_compat = true;
    // box_config.shared_memory_warp_parts = 1;
    if let Some(ref mut l1_cache) = box_config.data_cache_l1 {
        // workaround: compatible with accelsim which does not differentiate
        // between l1 tag lookup and hit latency
        Arc::get_mut(l1_cache).unwrap().l1_latency = l1_cache.l1_latency + l1_cache.l1_hit_latency;
    }
    crate::init_deadlock_detector();

    let box_config = Arc::new(box_config);
    let mut box_sim = crate::config::GTX1080::new(box_config);
    assert!(!box_sim.config.is_parallel_simulation());

    box_sim
        .trace
        .add_commands(&box_commands_path, box_traces_dir)?;

    let args = vec![
        "-trace".to_string(),
        accelsim_kernelslist_path.to_string_lossy().to_string(),
        "-config".to_string(),
        gpgpusim_config.to_string_lossy().to_string(),
        "-config".to_string(),
        trace_config.to_string_lossy().to_string(),
        "-inter_config_file".to_string(),
        inter_config.to_string_lossy().to_string(),
        "-gpgpu_n_clusters".to_string(),
        input
            .num_clusters
            .unwrap_or(box_sim.config.num_simt_clusters)
            .to_string(),
        "-gpgpu_n_cores_per_cluster".to_string(),
        input
            .cores_per_cluster
            .unwrap_or(box_sim.config.num_cores_per_simt_cluster)
            .to_string(),
        "-gpgpu_perf_sim_memcpy".to_string(),
        "0".to_string(),
    ];

    dbg!(&args);

    let play_config = playground::Config {
        accelsim_compat_mode: false,
        ..playground::Config::default()
    };
    let mut play_sim = playground::Accelsim::new(play_config, args)?;

    let mut play_time_cycle = Duration::ZERO;
    let mut play_time_other = Duration::ZERO;
    let mut box_time_cycle = Duration::ZERO;
    let mut box_time_other = Duration::ZERO;

    let mut gather_state_time = Duration::ZERO;
    let mut gather_box_state_time = Duration::ZERO;
    let mut gather_play_state_time = Duration::ZERO;

    let mut last_valid_box_sim_state = None;
    let mut last_valid_play_sim_state = None;

    // let mut cycle = 0;
    let mut box_cycle = 0;
    let mut play_cycle = 0;

    let use_full_diff = std::env::var("FULL_DIFF")
        .unwrap_or_default()
        .to_lowercase()
        == "yes";
    let allow_rel_err = std::env::var("REL_ERR").unwrap_or_default().to_lowercase() == "yes";

    let check_after: u64 = std::env::var("CHECK_AFTER")
        .ok()
        .as_deref()
        .map(str::parse)
        .transpose()?
        .unwrap_or(0);

    let check_every: Option<u64> = std::env::var("CHECK_EVERY")
        .ok()
        .as_deref()
        .map(str::parse)
        .transpose()?;

    // let should_compare_states = !box_sim.config.is_parallel_simulation() || check_every.is_some();
    // let should_compare_states = check_every.is_some();

    let check_every = check_every.unwrap_or(200);
    assert!(check_every >= 1);

    // let _num_schedulers = box_sim.config.num_schedulers_per_core;
    // let num_clusters = box_sim.config.num_simt_clusters;
    // let cores_per_cluster = box_sim.config.num_cores_per_simt_cluster;
    // assert_eq!(
    //     box_sim.config.total_cores(),
    //     num_clusters * cores_per_cluster
    // );
    // let _num_partitions = box_sim.mem_partition_units.len();
    // let _num_sub_partitions = box_sim.mem_sub_partitions.len();
    //
    // let mut box_sim_state = testing::state::Simulation::new(
    //     num_clusters,
    //     cores_per_cluster,
    //     num_partitions,
    //     num_sub_partitions,
    // );
    //
    // let mut play_sim_state = testing::state::Simulation::new(
    //     num_clusters,
    //     cores_per_cluster,
    //     num_partitions,
    //     num_sub_partitions,
    // );

    // box_sim.process_commands(cycle);
    // box_sim.launch_kernels(cycle);

    while play_sim.commands_left() || play_sim.kernels_left() {
        let mut start = Instant::now();
        play_sim.process_commands();
        play_sim.launch_kernels();
        play_time_other += start.elapsed();

        box_cycle = box_sim.process_commands(box_cycle);
        box_sim.launch_kernels(box_cycle);

        // check that memcopy commands were handled correctly
        // if false && should_compare_states {
        //     start = Instant::now();
        //     let (box_sim_state, play_sim_state) =
        //         gather_simulation_state(&mut box_sim, &mut play_sim);
        //     gather_state_time += start.elapsed();
        //
        //     // start = Instant::now();
        //     // gather_box_simulation_state(&mut box_sim, &mut box_sim_state, trace_provider);
        //     // box_sim_state = gather_box_simulation_state(
        //     //     num_clusters,
        //     //     cores_per_cluster,
        //     //     num_partitions,
        //     //     num_sub_partitions,
        //     //     &mut box_sim,
        //     //     trace_provider,
        //     // );
        //     // gather_box_state_time += start.elapsed();
        //     // gather_state_time += start.elapsed();
        //
        //     // start = Instant::now();
        //     // gather_play_simulation_state(&mut play_sim, &mut play_sim_state, trace_provider);
        //     // play_sim_state = gather_play_simulation_state(
        //     //     num_clusters,
        //     //     cores_per_cluster,
        //     //     num_partitions,
        //     //     num_sub_partitions,
        //     //     &mut play_sim,
        //     //     trace_provider,
        //     // );
        //     // gather_play_state_time += start.elapsed();
        //     // gather_state_time += start.elapsed();
        //
        //     if use_full_diff {
        //         full_diff::assert_eq!(&box_sim_state, &play_sim_state);
        //     } else {
        //         // we do fail here
        //         diff::assert_eq!(box: &box_sim_state, play: &play_sim_state);
        //     }
        // }

        // start = Instant::now();
        // box_sim.process_commands();
        // box_sim.launch_kernels();
        // box_time_other += start.elapsed();

        let mut finished_kernel_uid: Option<u32> = None;
        loop {
            if !play_sim.active() {
                break;
            }

            start = Instant::now();
            play_sim.cycle();
            play_time_cycle += start.elapsed();

            start = Instant::now();
            box_cycle = box_sim.cycle(box_cycle);
            box_time_cycle += start.elapsed();

            play_cycle = play_sim.get_cycle();

            assert_eq!(box_cycle, play_cycle);
            // let cycle = box_cycle;

            // let should_check = cycle >= check_after && cycle % check_every == 0;
            let should_check = box_cycle >= check_after && box_cycle % check_every == 0;
            // if should_compare_states && should_check {
            if should_check {
                start = Instant::now();
                let (box_sim_state, play_sim_state) =
                    gather_simulation_state(&mut box_sim, &mut play_sim, box_cycle, play_cycle);
                gather_state_time += start.elapsed();

                start = Instant::now();
                // gather_box_simulation_state(&mut box_sim, &mut box_sim_state, trace_provider);
                // box_sim_state = gather_box_simulation_state(
                //     num_clusters,
                //     cores_per_cluster,
                //     num_partitions,
                //     num_sub_partitions,
                //     &mut box_sim,
                //     trace_provider,
                // );
                gather_box_state_time += start.elapsed();

                start = Instant::now();
                // gather_play_simulation_state(&mut play_sim, &mut play_sim_state, trace_provider);
                // play_sim_state = gather_play_simulation_state(
                //     num_clusters,
                //     cores_per_cluster,
                //     num_partitions,
                //     num_sub_partitions,
                //     num_schedulers,
                //     &mut play_sim,
                //     trace_provider,
                // );
                gather_play_state_time += start.elapsed();

                // sanity checks
                // assert_eq!(
                //     schedulers.len(),
                //     box_sim_state.scheduler_per_core[core_id].len()
                // );
                // let box_sched = &mut box_sim_state.scheduler_per_core[core_id][sched_idx];
                // let play_sched = &mut play_sim_state.scheduler_per_core[core_id][sched_idx];
                //
                // let num_box_warps = box_sched.prioritized_warp_ids.len();
                // let num_play_warps = play_sched.prioritized_warp_ids.len();
                // let limit = num_box_warps.min(num_play_warps);
                //
                // // make sure we only compare what can be compared
                // box_sched.prioritized_warp_ids.split_off(limit);
                // // box_sched.prioritized_dynamic_warp_ids.split_off(limit);
                // play_sched.prioritized_warp_ids.split_off(limit);
                // // play_sched.prioritized_dynamic_warp_ids.split_off(limit);
                //
                // assert_eq!(
                //     box_sched.prioritized_warp_ids.len(),
                //     play_sched.prioritized_warp_ids.len(),
                // );
                // // assert_eq!(
                // //     box_sched.prioritized_dynamic_warp_ids.len(),
                // //     play_sched.prioritized_dynamic_warp_ids.len(),
                // // );

                if box_sim_state != play_sim_state {
                    // println!(
                    //     "validated play state for cycle {}: {:#?}",
                    //     cycle - 1,
                    //     &last_valid_play_sim_state
                    // );

                    serde_json::to_writer_pretty(
                        utils::fs::open_writable(manifest_dir.join("debug.playground.state.json"))?,
                        &last_valid_play_sim_state,
                    )?;

                    // format!("{:#?}", ).as_bytes(),
                    serde_json::to_writer_pretty(
                        utils::fs::open_writable(manifest_dir.join("debug.box.state.json"))?,
                        &last_valid_box_sim_state,
                    )?;
                    // .write_all(format!("{:#?}", last_valid_box_sim_state).as_bytes())?;

                    // dbg!(&box_sim.allocations);
                    // for (sub_id, sub) in play_sim.sub_partitions().enumerate() {
                    //     let play_icnt_l2_queue = sub
                    //         .interconn_to_l2_queue()
                    //         .iter()
                    //         .map(|fetch| fetch.get_addr())
                    //         .collect::<Vec<_>>();
                    //     dbg!(sub_id, play_icnt_l2_queue);
                    // }
                    //
                    // for (sub_id, sub) in box_sim.mem_sub_partitions.iter().enumerate() {
                    //     let box_icnt_l2_queue = sub
                    //         .borrow()
                    //         .interconn_to_l2_queue
                    //         .iter()
                    //         .map(|fetch| fetch.addr())
                    //         .collect::<Vec<_>>();
                    //     dbg!(sub_id, box_icnt_l2_queue);
                    // }
                }

                println!(
                    "checking for diff after cycle {box_cycle} (box cycle={box_cycle}, play cycle={play_cycle})",
                );

                // this is useful for debugging scheduler issues
                // find the bad core and scheduler
                #[cfg(debug_assertions)]
                for (core_id, per_core) in box_sim_state.scheduler_per_core.iter().enumerate() {
                    for (scheduler_id, scheduler) in per_core.iter().enumerate() {
                        let box_prio = &scheduler.prioritized_warp_ids;
                        let play_prio = &play_sim_state.scheduler_per_core[core_id][scheduler_id]
                            .prioritized_warp_ids;
                        if box_prio != play_prio {
                            println!("CORE {core_id} SCHEDULER {scheduler_id} MISMATCH:");
                            diff::assert_eq!(box: box_prio, play: play_prio);
                        }
                    }
                }

                #[cfg(debug_assertions)]
                for (core_id, per_core) in
                    box_sim_state.l1_latency_queue_per_core.iter().enumerate()
                {
                    for (bank_id, queue) in per_core.iter() {
                        let (play_bank_id, play_queue) =
                            &play_sim_state.l1_latency_queue_per_core[core_id][*bank_id];
                        assert_eq!(play_bank_id, bank_id);
                        if queue != play_queue {
                            println!(" box core {} l1 bank {}: {:?}", core_id, bank_id, queue);
                            println!(
                                "play core {} l1 bank {}: {:?}",
                                core_id, bank_id, play_queue
                            );
                        }
                    }
                }

                if use_full_diff {
                    full_diff::assert_eq!(&box_sim_state, &play_sim_state);
                } else {
                    // assert_eq!!(&box_sim_state, &play_sim_state);
                    // assert_eq!(
                    //     &box_sim_state.last_cluster_issue,
                    //     &play_sim_state.last_cluster_issue
                    // );
                    // assert_eq!(
                    //     &box_sim_state.interconn_to_l2_queue_per_sub,
                    //     &play_sim_state.interconn_to_l2_queue_per_sub
                    // );
                    // assert_eq!(
                    //     &box_sim_state.l2_to_interconn_queue_per_sub,
                    //     &play_sim_state.l2_to_interconn_queue_per_sub
                    // );
                    // assert_eq!(
                    //     &box_sim_state.l2_to_dram_queue_per_sub,
                    //     &play_sim_state.l2_to_dram_queue_per_sub
                    // );
                    // assert_eq!(
                    //     &box_sim_state.dram_to_l2_queue_per_sub,
                    //     &play_sim_state.dram_to_l2_queue_per_sub
                    // );
                    // assert_eq!(
                    //     &box_sim_state.rop_queue_per_sub,
                    //     &play_sim_state.rop_queue_per_sub
                    // );
                    // assert_eq!(
                    //     &box_sim_state.l2_cache_per_sub,
                    //     &play_sim_state.l2_cache_per_sub
                    // );
                    // assert_eq!(
                    //     &box_sim_state.dram_latency_queue_per_partition,
                    //     &play_sim_state.dram_latency_queue_per_partition
                    // );
                    // assert_eq!(
                    //     &box_sim_state.dram_arbitration_per_partition,
                    //     &play_sim_state.dram_arbitration_per_partition
                    // );
                    // assert_eq!(
                    //     &box_sim_state.core_sim_order_per_cluster,
                    //     &play_sim_state.core_sim_order_per_cluster
                    // );
                    // assert_eq!(
                    //     &box_sim_state.functional_unit_occupied_slots_per_core,
                    //     &play_sim_state.functional_unit_occupied_slots_per_core
                    // );
                    // assert_eq!(
                    //     &box_sim_state.functional_unit_pipelines_per_core,
                    //     &play_sim_state.functional_unit_pipelines_per_core
                    // );
                    //
                    // assert_eq!(
                    //     box_sim_state.operand_collector_per_core.len(),
                    //     play_sim_state.operand_collector_per_core.len()
                    // );
                    // for collector_id in 0..box_sim_state.operand_collector_per_core.len() {
                    //     diff::assert_eq!(
                    //         box: &box_sim_state.operand_collector_per_core[collector_id]
                    //             .as_ref()
                    //             .map(|c| &c.ports),
                    //         play: &play_sim_state.operand_collector_per_core[collector_id]
                    //             .as_ref()
                    //             .map(|c| &c.ports)
                    //     );
                    //     diff::assert_eq!(
                    //         box: &box_sim_state.operand_collector_per_core[collector_id]
                    //             .as_ref()
                    //             .map(|c| &c.collector_units),
                    //         play: &play_sim_state.operand_collector_per_core[collector_id]
                    //             .as_ref()
                    //             .map(|c| &c.collector_units)
                    //     );
                    //     diff::assert_eq!(
                    //         box: &box_sim_state.operand_collector_per_core[collector_id]
                    //             .as_ref()
                    //             .map(|c| &c.dispatch_units),
                    //         plaxy: &play_sim_state.operand_collector_per_core[collector_id]
                    //             .as_ref()
                    //             .map(|c| &c.dispatch_units)
                    //     );
                    //     diff::assert_eq!(
                    //         box: &box_sim_state.operand_collector_per_core[collector_id]
                    //             .as_ref()
                    //             .map(|c| &c.arbiter),
                    //         play: &play_sim_state.operand_collector_per_core[collector_id]
                    //             .as_ref()
                    //             .map(|c| &c.arbiter),
                    //     );
                    // }
                    // diff::assert_eq!(
                    //     box: &box_sim_state.operand_collector_per_core,
                    //     play: &play_sim_state.operand_collector_per_core,
                    // );
                    // diff::assert_eq!(
                    //     box: &box_sim_state.scheduler_per_core,
                    //     play: &play_sim_state.scheduler_per_core
                    // );
                    // diff::assert_eq!(
                    //     box: &box_sim_state.pending_register_writes_per_core,
                    //     play: &play_sim_state.pending_register_writes_per_core
                    // );
                    // diff::assert_eq!(
                    //     box: &box_sim_state.l1_latency_queue_per_core,
                    //     play: &play_sim_state.l1_latency_queue_per_core
                    // );
                    // diff::assert_eq!(
                    //     box: &box_sim_state.l1_cache_per_core,
                    //     play: &play_sim_state.l1_cache_per_core
                    // );

                    diff::assert_eq!(box: &box_sim_state, play: &play_sim_state);
                }

                // this should be okay performance wise (copy, no allocation)
                last_valid_box_sim_state = Some(box_sim_state.clone());
                last_valid_play_sim_state = Some(play_sim_state.clone());
            }

            // box out of loop
            start = Instant::now();
            // if !box_sim.active() {
            //     box_sim.process_commands(cycle);
            //     box_sim.launch_kernels(cycle);
            // }

            // cycle = play_sim.get_cycle();

            // if let Some(box_sim.current_kernel.lock())
            // box_sim
            //     .stats
            //     .lock()
            //     .get_mut(box_sim.current_kernel.lock().as_ref().unwrap().id() as usize)
            //     .sim
            //     .cycles += 1;
            // box_sim.set_cycle(cycle);

            if let Some(kernel) = box_sim.kernel_manager.get_finished_kernel() {
                box_sim.cleanup_finished_kernel(&*kernel, box_cycle);
            }
            box_time_other += start.elapsed();

            finished_kernel_uid = play_sim.finished_kernel_uid();
            if finished_kernel_uid.is_some() {
                break;
            }
        }

        // dbg!(&box_sim
        //     .stats()
        //     .as_ref()
        //     .iter()
        //     .map(|kernel_stats| &kernel_stats.sim)
        //     .collect::<Vec<_>>());

        if let Some(uid) = finished_kernel_uid {
            play_sim.cleanup_finished_kernel(uid);
        }

        // if let Some(kernel) = box_sim.finished_kernel() {
        //     box_sim.cleanup_finished_kernel(&kernel, cycle);
        // }

        if play_sim.limit_reached() {
            println!("GPGPU-Sim: ** break due to reaching the maximum cycles (or instructions) **");
            std::io::stdout().flush()?;
            break;
        }
    }

    // dbg!(&box_sim
    //     .stats()
    //     .as_ref()
    //     .iter()
    //     .map(|kernel_stats| &kernel_stats.sim)
    //     .collect::<Vec<_>>());

    // let play_cycle = cycle;
    // #[allow(unused_mut)]
    // let mut box_cycle = cycle;

    // if box_sim.parallel_simulation {
    //     // allow parallel simulation to complete
    //     box_sim.run_to_completion()?;
    //     box_cycle = box_sim.cycle.get();
    // }

    if box_cycle > 0 {
        let box_time_cycle = box_time_cycle / u32::try_from(box_cycle).unwrap();
        let box_time_other = box_time_other / u32::try_from(box_cycle).unwrap();
        println!(
            "box time  (cycle):\t {:>3.6} ms",
            box_time_cycle.as_secs_f64() * 1000.0
        );
        println!(
            "box time  (other):\t {:>3.6} ms",
            box_time_other.as_secs_f64() * 1000.0
        );
    }
    if play_cycle > 0 {
        let play_time_cycle = play_time_cycle / u32::try_from(play_cycle).unwrap();
        let play_time_other = play_time_other / u32::try_from(play_cycle).unwrap();
        println!(
            "play time (cycle):\t {:>3.6} ms",
            play_time_cycle.as_secs_f64() * 1000.0
        );
        println!(
            "play time (other):\t {:>3.6} ms",
            play_time_other.as_secs_f64() * 1000.0
        );
    }

    let cycle = box_cycle;
    let num_checks = u32::try_from(cycle.saturating_sub(check_after) / check_every).unwrap();
    // if !box_sim.config.is_parallel_simulation() && num_checks > 0 {
    if num_checks > 0 {
        let gather_box_state_time = gather_box_state_time / num_checks;
        let gather_play_state_time = gather_play_state_time / num_checks;
        let gather_state_time = gather_state_time / num_checks;

        dbg!(gather_box_state_time);
        dbg!(gather_play_state_time);
        dbg!(gather_box_state_time + gather_play_state_time);
        dbg!(gather_state_time);
    }

    dbg!(&play_cycle);
    dbg!(&box_cycle);
    // dbg!(&cycle);

    // if box_sim.config.is_parallel_simulation() {
    //     dbg!(&play_cycle);
    //     dbg!(&box_cycle);
    // }

    let play_stats = play_sim.stats();

    // dbg!(&play_stats.sim);
    // dbg!(&box_sim
    //     .stats()
    //     .as_ref()
    //     .iter()
    //     .map(|kernel_stats| &kernel_stats.sim)
    //     .collect::<Vec<_>>());
    // assert!(false);

    let mut box_stats = box_sim.stats().reduce();
    box_stats.l1i_stats = box_stats.l1i_stats.merge_allocations();
    box_stats.l1c_stats = box_stats.l1c_stats.merge_allocations();
    box_stats.l1d_stats = box_stats.l1d_stats.merge_allocations();
    box_stats.l1t_stats = box_stats.l1t_stats.merge_allocations();
    box_stats.l2d_stats = box_stats.l2d_stats.merge_allocations();

    // dbg!(&box_stats.l1d_stats);
    // dbg!(&play_stats);
    // dbg!(&box_stats);

    // dbg!(&playground_dur);
    // dbg!(&box_dur);

    // allow 5% difference
    // let max_rel_err = if allow_rel_err { Some(0.05) } else { None };
    // allow absolute difference of 10
    // let abs_threshold = if allow_rel_err { Some(10.0) } else { None };
    // asserts::stats_match(play_stats, box_stats, max_rel_err, abs_threshold, true);

    diff::assert_eq!(play: &play_cycle, box: &box_cycle);
    asserts::stats_match(play_stats, box_stats, None, None, false);

    Ok(())
}

#[test]
fn lockstep_accelsim_pchase() -> eyre::Result<()> {
    crate::testing::init_test();
    // let benchmarks = validate::materialized::Benchmarks::default()?;
    // let configs = benchmarks.config.accelsim_simulate.configs;
    let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
    let target_config = TargetBenchmarkConfig::Simulate {
        parallel: Some(false),
        l2_prefill: Some(false),
        accelsim_traces_dir: manifest_dir.join("./pchase-debug-trace"),
        traces_dir: manifest_dir.join("./pchase-debug-trace/box"),
        stats_dir: manifest_dir.join("./pchase-debug-trace/stats"),
    };
    let bench_config = BenchmarkConfig {
        ..BenchmarkConfig::custom(target_config)
    };
    dbg!(&bench_config);
    run(&bench_config, TraceProvider::Accelsim)
}

macro_rules! lockstep_checks {
    ($($name:ident: ($bench_name:expr, $($input:tt)+),)*) => {
        $(
            paste::paste! {
                #[ignore = "native traces cannot be compared"]
                #[test]
                fn [<lockstep_native_ $name>]() -> color_eyre::eyre::Result<()> {
                    $crate::testing::init_test();
                    let bench_config = $crate::testing::get_bench_config(
                        $bench_name,
                        validate::input!($($input)+)?,
                    )?;
                    run(&bench_config, TraceProvider::Native)
                }

                #[test]
                fn [<lockstep_accelsim_ $name>]() -> color_eyre::eyre::Result<()> {
                    $crate::testing::init_test();
                    let bench_config = $crate::testing::get_bench_config(
                        $bench_name,
                        validate::input!($($input)+)?,
                    )?;
                    run(&bench_config, TraceProvider::Accelsim)
                }

                #[test]
                fn [<lockstep_box_ $name>]() -> color_eyre::eyre::Result<()> {
                    $crate::testing::init_test();
                    let bench_config = $crate::testing::get_bench_config(
                        $bench_name,
                        validate::input!($($input)+)?,
                    )?;
                    run(&bench_config, TraceProvider::Box)
                }
            }
        )*
    }
}

lockstep_checks! {
    // vectoradd
    vectoradd_32_100_test: ("vectorAdd", { "dtype": 32, "length": 100 }),
    vectoradd_32_1000_test: ("vectorAdd", { "dtype": 32, "length": 1000  }),
    vectoradd_32_10000_test: ("vectorAdd", { "dtype": 32, "length": 10000 }),
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

    // transpose
    transpose_256_naive_test: ("transpose", { "dim": 256, "variant": "naive"}),
    transpose_256_coalesed_test: ("transpose", { "dim": 256, "variant": "coalesced" }),
    // transpose_256_optimized_test: ("transpose", { "dim": 256, "variant": "optimized" }),

    // babelstream
    babelstream_1024_test: ("babelstream", { "size": 1024 }),
    babelstream_10240_test: ("babelstream", { "size": 10240 }),

    // extra tests for large input sizes
    // vectoradd_32_500000_test: ("vectorAdd", {
    //     "dtype": 32, "length": 500000, "memory_only": false, "cores_per_cluster": 4 }),
}
