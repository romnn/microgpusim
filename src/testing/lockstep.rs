use super::diff;
use crate::{cache, config, fifo, interconn as ic, mem_fetch, register_set, testing};
use color_eyre::eyre;
use itertools::Itertools;
use pretty_assertions_sorted as full_diff;
use serde::Serialize;

use std::collections::HashSet;
use std::io::Write;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use trace_model::Command;

#[derive(Debug, Clone, Copy)]
pub enum TraceProvider {
    Native,
    Accelsim,
    Box,
}

#[inline]
fn gather_simulation_state(
    box_sim: &mut crate::MockSimulator<ic::ToyInterconnect<crate::Packet>>,
    play_sim: &mut playground::Accelsim,
    _trace_provider: TraceProvider,
) -> (testing::state::Simulation, testing::state::Simulation) {
    // iterate over sub partitions
    let num_schedulers = box_sim.config.num_schedulers_per_core;
    let num_clusters = box_sim.config.num_simt_clusters;
    let cores_per_cluster = box_sim.config.num_cores_per_simt_cluster;
    assert_eq!(
        box_sim.config.total_cores(),
        num_clusters * cores_per_cluster
    );

    let num_partitions = box_sim.mem_partition_units.len();
    let num_sub_partitions = box_sim.mem_sub_partitions.len();
    let mut box_sim_state = testing::state::Simulation::new(
        num_clusters,
        cores_per_cluster,
        num_partitions,
        num_sub_partitions,
        num_schedulers,
    );

    box_sim_state.last_cluster_issue = box_sim.last_cluster_issue;

    for (cluster_id, cluster) in box_sim.clusters.iter().enumerate() {
        // for (core_id, core) in cluster.cores.lock().unwrap().iter().enumerate() {
        for (core_id, core) in cluster.cores.iter().enumerate() {
            let core = core.lock().unwrap();
            let global_core_id = cluster_id * box_sim.config.num_cores_per_simt_cluster + core_id;
            assert_eq!(core.core_id, global_core_id);

            // this is the one we will use (unless the assertion is ever false)
            let core_id = core.core_id;

            // core: functional units
            for (fu_id, fu) in core.functional_units.iter().enumerate() {
                let _fu = fu.lock().unwrap();
                let issue_port = core.issue_ports[fu_id];
                let issue_reg: register_set::RegisterSet = core.pipeline_reg[issue_port as usize]
                    // .borrow()
                    .try_lock()
                    .unwrap()
                    .clone();
                assert_eq!(issue_port, issue_reg.stage);

                box_sim_state.functional_unit_pipelines_per_core[core_id].push(issue_reg.into());
            }
            for (_fu_id, fu) in core.functional_units.iter().enumerate() {
                let fu = fu.lock().unwrap();
                box_sim_state.functional_unit_pipelines_per_core[core_id].push(
                    testing::state::RegisterSet {
                        name: fu.id().to_string(),
                        pipeline: fu
                            .pipeline()
                            .iter()
                            .map(|reg| reg.clone().map(Into::into))
                            .collect(),
                    },
                );
            }
            // core: operand collector
            box_sim_state.operand_collector_per_core[core_id] =
                Some(core.operand_collector.try_lock().unwrap().deref().into());
            // Some(core.inner.operand_collector.borrow().deref().into());
            // core: schedulers
            box_sim_state.scheduler_per_core[core_id] = core
                .schedulers
                .iter()
                .map(|scheduler| scheduler.lock().unwrap().deref().into())
                .collect();
            // .extend(core.schedulers.iter().map(Into::into));
            // core: l2 cache
            let ldst_unit = core.load_store_unit.lock().unwrap();

            // core: pending register writes
            box_sim_state.pending_register_writes_per_core[core_id] = ldst_unit
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

            box_sim_state.pending_register_writes_per_core[core_id].sort();
            // let l1d_tag_array = ldst_unit.data_l1.unwrap().tag_array;
            // dbg!(&l1d_tag_array);
            // let l1d_tag_array = ldst_unit.data_l1.unwrap().tag_array;
        }
    }

    for (partition_id, partition) in box_sim.mem_partition_units.iter().enumerate() {
        box_sim_state.dram_latency_queue_per_partition[partition_id].extend(
            partition
                .dram_latency_queue
                .clone()
                .into_iter()
                .map(Into::into),
        );
        box_sim_state.dram_arbitration_per_partition[partition_id] = testing::state::Arbitration {
            last_borrower: partition.arbitration_metadata.last_borrower,
            shared_credit: partition.arbitration_metadata.shared_credit,
            private_credit: partition.arbitration_metadata.private_credit.clone().into(),
        };
    }
    for (sub_id, sub) in box_sim.mem_sub_partitions.iter().enumerate() {
        // let sub = sub.borrow();
        let sub = sub.try_lock().unwrap();
        let l2_cache = sub.l2_cache.as_ref().unwrap();
        let l2_cache: &cache::DataL2<ic::L2Interface<fifo::FifoQueue<mem_fetch::MemFetch>>> =
            l2_cache.as_any().downcast_ref().unwrap();

        box_sim_state.l2_cache_per_sub[sub_id] =
            Some(l2_cache.inner.inner.tag_array.clone().into());

        for (dest_queue, src_queue) in [
            (
                &mut box_sim_state.interconn_to_l2_queue_per_sub[sub_id],
                &sub.interconn_to_l2_queue,
            ),
            (
                &mut box_sim_state.l2_to_interconn_queue_per_sub[sub_id],
                &sub.l2_to_interconn_queue,
            ),
            (
                &mut box_sim_state.l2_to_dram_queue_per_sub[sub_id],
                &sub.l2_to_dram_queue.lock().unwrap(),
            ),
            (
                &mut box_sim_state.dram_to_l2_queue_per_sub[sub_id],
                &sub.dram_to_l2_queue,
            ),
        ] {
            // dest_queue.extend(src_queue.clone().into_iter().map(Into::into));
            *dest_queue = src_queue.clone().into_iter().map(Into::into).collect();
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

        // core: pending register writes
        play_sim_state.pending_register_writes_per_core[core_id] = core
            .pending_register_writes()
            .into_iter()
            .map(Into::into)
            .collect();
        play_sim_state.pending_register_writes_per_core[core_id].sort();

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
    }

    let mut partitions_added = 0;
    for (partition_id, partition) in play_sim.partition_units().enumerate() {
        assert!(partition_id < num_partitions);
        play_sim_state.dram_latency_queue_per_partition[partition_id] = partition
            .dram_latency_queue()
            .into_iter()
            .map(Into::into)
            .collect();
        // .extend(partition.dram_latency_queue().into_iter().map(Into::into));
        partitions_added += 1;

        play_sim_state.dram_arbitration_per_partition[partition_id] = testing::state::Arbitration {
            last_borrower: partition.last_borrower(),
            shared_credit: partition.shared_credit(),
            private_credit: partition.private_credit().into(),
        };
    }
    assert_eq!(partitions_added, num_partitions);

    let mut sub_partitions_added = 0;
    for (sub_id, sub) in play_sim.sub_partitions().enumerate() {
        play_sim_state.interconn_to_l2_queue_per_sub[sub_id] = sub
            .interconn_to_l2_queue()
            .into_iter()
            .map(Into::into)
            .collect();
        // .extend(sub.interconn_to_l2_queue().into_iter().map(Into::into));
        play_sim_state.l2_to_interconn_queue_per_sub[sub_id] = sub
            .l2_to_interconn_queue()
            .into_iter()
            .map(Into::into)
            .collect();
        // .extend(sub.l2_to_interconn_queue().into_iter().map(Into::into));
        play_sim_state.dram_to_l2_queue_per_sub[sub_id] =
            sub.dram_to_l2_queue().into_iter().map(Into::into).collect();
        // .extend(sub.dram_to_l2_queue().into_iter().map(Into::into));
        play_sim_state.l2_to_dram_queue_per_sub[sub_id] =
            sub.l2_to_dram_queue().into_iter().map(Into::into).collect();
        // .extend(sub.l2_to_dram_queue().into_iter().map(Into::into));

        play_sim_state.l2_cache_per_sub[sub_id] = Some(testing::state::Cache {
            lines: sub.l2_cache().lines().into_iter().map(Into::into).collect(),
        });
        sub_partitions_added += 1;
    }
    assert_eq!(sub_partitions_added, num_sub_partitions);
    (box_sim_state, play_sim_state)
}

// #[deprecated]
// #[inline]
// fn gather_box_simulation_state(
//     // num_clusters: usize,
//     // cores_per_cluster: usize,
//     // num_partitions: usize,
//     // num_sub_partitions: usize,
//     box_sim: &mut crate::MockSimulator<ic::ToyInterconnect<super::Packet>>,
//     box_sim_state: &mut testing::state::Simulation,
//     _trace_provider: TraceProvider,
// ) {
//     // ) -> testing::state::Simulation {
//     // let mut box_sim_state = testing::state::Simulation::new(
//     //     num_clusters,
//     //     cores_per_cluster,
//     //     num_partitions,
//     //     num_sub_partitions,
//     // );
//
//     box_sim_state.last_cluster_issue = box_sim.last_cluster_issue;
//
//     for (cluster_id, cluster) in box_sim.clusters.iter().enumerate() {
//         // cluster: core sim order
//         box_sim_state.core_sim_order_per_cluster[cluster_id] =
//             cluster.core_sim_order.iter().copied().collect();
//
//         for (core_id, core) in cluster.cores.lock().unwrap().iter().enumerate() {
//             let global_core_id =
//                 cluster_id * box_sim.config.num_cores_per_simt_cluster + core_id;
//             assert_eq!(core.inner.core_id, global_core_id);
//
//             // this is the one we will use (unless the assertion is ever false)
//             let core_id = core.inner.core_id;
//
//             // core: functional units
//             let num_fus = core.functional_units.len();
//             box_sim_state.functional_unit_pipelines_per_core[core_id]
//                 .resize(2 * num_fus, testing::state::RegisterSet::default());
//
//             for (fu_id, fu) in core.functional_units.iter().enumerate() {
//                 let _fu = fu.lock().unwrap();
//                 let issue_port = core.issue_ports[fu_id];
//                 let issue_reg: super::register_set::RegisterSet = core.inner.pipeline_reg
//                     [issue_port as usize]
//                     .borrow()
//                     .clone();
//                 assert_eq!(issue_port, issue_reg.stage);
//
//                 box_sim_state.functional_unit_pipelines_per_core[core_id][fu_id] =
//                     issue_reg.into();
//             }
//
//             // box_sim_state.functional_unit_pipelines_per_core[core_id]
//             //     .resize(num_fus, testing::state::RegisterSet::default());
//             for (fu_id, fu) in core.functional_units.iter().enumerate() {
//                 let fu = fu.lock().unwrap();
//                 box_sim_state.functional_unit_pipelines_per_core[core_id][num_fus + fu_id] =
//                     testing::state::RegisterSet {
//                         name: fu.id().to_string(),
//                         pipeline: fu
//                             .pipeline()
//                             .iter()
//                             .map(|reg| reg.clone().map(Into::into))
//                             .collect(),
//                     };
//                 // .push();
//             }
//             // core: operand collector
//             box_sim_state.operand_collector_per_core[core_id] =
//                 Some(core.inner.operand_collector.borrow().deref().into());
//             // core: schedulers
//             box_sim_state.scheduler_per_core[core_id] =
//                 core.schedulers.iter().map(Into::into).collect();
//             // .extend(core.schedulers.iter().map(Into::into));
//             // core: l2 cache
//             let ldst_unit = core.inner.load_store_unit.lock().unwrap();
//
//             // core: pending register writes
//             box_sim_state.pending_register_writes_per_core[core_id] = ldst_unit
//                 .pending_writes
//                 .clone()
//                 .into_iter()
//                 .flat_map(|(warp_id, pending_registers)| {
//                     pending_registers
//                         .into_iter()
//                         .map(
//                             move |(reg_num, pending)| testing::state::PendingRegisterWrites {
//                                 warp_id,
//                                 reg_num,
//                                 pending,
//                             },
//                         )
//                 })
//                 .collect();
//
//             box_sim_state.pending_register_writes_per_core[core_id].sort();
//             // let l1d_tag_array = ldst_unit.data_l1.unwrap().tag_array;
//             // dbg!(&l1d_tag_array);
//             // let l1d_tag_array = ldst_unit.data_l1.unwrap().tag_array;
//         }
//     }
//
//     for (partition_id, partition) in box_sim.mem_partition_units.iter().enumerate() {
//         box_sim_state.dram_latency_queue_per_partition[partition_id] = partition
//             .dram_latency_queue
//             .clone()
//             .into_iter()
//             .map(Into::into)
//             .collect();
//     }
//     for (sub_id, sub) in box_sim.mem_sub_partitions.iter().enumerate() {
//         let sub = sub.borrow();
//         let l2_cache = sub.l2_cache.as_ref().unwrap();
//         let l2_cache: &ported::l2::DataL2<
//             ic::L2Interface<fifo::FifoQueue<ported::mem_fetch::MemFetch>>,
//         > = l2_cache.as_any().downcast_ref().unwrap();
//
//         box_sim_state.l2_cache_per_sub[sub_id] =
//             Some(l2_cache.inner.inner.tag_array.clone().into());
//
//         for (dest_queue, src_queue) in [
//             (
//                 &mut box_sim_state.interconn_to_l2_queue_per_sub[sub_id],
//                 &sub.interconn_to_l2_queue,
//             ),
//             (
//                 &mut box_sim_state.l2_to_interconn_queue_per_sub[sub_id],
//                 &sub.l2_to_interconn_queue,
//             ),
//             (
//                 &mut box_sim_state.l2_to_dram_queue_per_sub[sub_id],
//                 &sub.l2_to_dram_queue.lock().unwrap(),
//             ),
//             (
//                 &mut box_sim_state.dram_to_l2_queue_per_sub[sub_id],
//                 &sub.dram_to_l2_queue,
//             ),
//         ] {
//             *dest_queue = src_queue.clone().into_iter().map(Into::into).collect();
//             // dest_queue.extend(src_queue.clone().into_iter().map(Into::into));
//         }
//     }
//     // box_sim_state
// }
//
// #[deprecated]
// #[inline]
// fn gather_play_simulation_state(
//     // num_clusters: usize,
//     // cores_per_cluster: usize,
//     // num_partitions: usize,
//     // num_sub_partitions: usize,
//     // num_schedulers: usize,
//     play_sim: &mut playground::Accelsim,
//     play_sim_state: &mut testing::state::Simulation,
//     _trace_provider: TraceProvider,
// ) {
//     // ) -> testing::state::Simulation {
//     // let mut play_sim_state = testing::state::Simulation::new(
//     //     num_clusters,
//     //     cores_per_cluster,
//     //     num_partitions,
//     //     num_sub_partitions,
//     //     num_schedulers,
//     // );
//
//     play_sim_state.last_cluster_issue = play_sim.last_cluster_issue() as usize;
//
//     for (cluster_id, cluster) in play_sim.clusters().enumerate() {
//         // cluster: core sim order
//         play_sim_state.core_sim_order_per_cluster[cluster_id] = cluster.core_sim_order().into();
//     }
//     for (core_id, core) in play_sim.cores().enumerate() {
//         let fu_issue_regs: Vec<_> = core
//             .functional_unit_issue_register_sets()
//             .into_iter()
//             .map(testing::state::RegisterSet::from)
//             .collect();
//         let num_fu_issue_regs = fu_issue_regs.len();
//         // let fu_names: HashSet<_> = fu_issue_regs.iter().map(|fu| fu.get_name()).collect();
//         // let valid_units: HashSet<_> = fu_issue_regs.iter().map(|fu| fu.name.clone()).collect();
//
//         play_sim_state.functional_unit_pipelines_per_core[core_id]
//             .resize(num_fu_issue_regs, testing::state::RegisterSet::default());
//
//         for (reg_id, regs) in fu_issue_regs.into_iter().enumerate() {
//             play_sim_state.functional_unit_pipelines_per_core[core_id][reg_id] = regs.into();
//         }
//         // let valid_units: HashSet<_> = box_sim_state.functional_unit_pipelines_per_core[core_id]
//         // let valid_units: HashSet<_> = [
//
//         let fu_simd_pipelines: Vec<testing::state::RegisterSet> = core
//             .functional_unit_simd_pipeline_register_sets()
//             .into_iter()
//             .map(testing::state::RegisterSet::from)
//             // .filter(|fu| valid_units.contains(&fu.name))
//             .filter(|fu| match fu.name.as_str() {
//                 "SPUnit" | "LdstUnit" => true,
//                 _ => false,
//             })
//             .collect();
//         let num_fu_simd_pipelines = fu_simd_pipelines.len();
//
//         play_sim_state.functional_unit_pipelines_per_core[core_id].resize(
//             num_fu_issue_regs + num_fu_simd_pipelines,
//             testing::state::RegisterSet::default(),
//         );
//         // assert_eq!(num_fu_issue_regs, num_fu_simd_pipelines);
//         for (reg_id, regs) in fu_simd_pipelines
//             .into_iter()
//             // .filter(|fu| valid_units.contains(&fu.name()))
//             .enumerate()
//         {
//             play_sim_state.functional_unit_pipelines_per_core[core_id]
//                 [num_fu_issue_regs + reg_id] = regs.into();
//         }
//
//         // core: pending register writes
//         play_sim_state.pending_register_writes_per_core[core_id] = core
//             .pending_register_writes()
//             .into_iter()
//             .map(Into::into)
//             .collect();
//         play_sim_state.pending_register_writes_per_core[core_id].sort();
//
//         // core: operand collector
//         let coll = core.operand_collector();
//         play_sim_state.operand_collector_per_core[core_id] = Some(coll.into());
//         // core: scheduler units
//         // let schedulers = core.schedulers();
//         // play_sim_state.scheduler_per_core[core_id]
//         // .resize(schedulers.len(), testing::state::Scheduler::default());
//
//         for (sched_idx, play_scheduler) in core.schedulers().into_iter().enumerate() {
//             play_sim_state.scheduler_per_core[core_id][sched_idx] = play_scheduler.into();
//         }
//     }
//
//     for (partition_id, partition) in play_sim.partition_units().enumerate() {
//         play_sim_state.dram_latency_queue_per_partition[partition_id] = partition
//             .dram_latency_queue()
//             .into_iter()
//             .map(Into::into)
//             .collect();
//     }
//     for (sub_id, sub) in play_sim.sub_partitions().enumerate() {
//         play_sim_state.interconn_to_l2_queue_per_sub[sub_id] = sub
//             .interconn_to_l2_queue()
//             .into_iter()
//             .map(Into::into)
//             .collect();
//         // .extend(sub.interconn_to_l2_queue().into_iter().map(Into::into));
//         play_sim_state.l2_to_interconn_queue_per_sub[sub_id] = sub
//             .l2_to_interconn_queue()
//             .into_iter()
//             .map(Into::into)
//             .collect();
//         // .extend(sub.l2_to_interconn_queue().into_iter().map(Into::into));
//         play_sim_state.dram_to_l2_queue_per_sub[sub_id] =
//             sub.dram_to_l2_queue().into_iter().map(Into::into).collect();
//         // .extend(sub.dram_to_l2_queue().into_iter().map(Into::into));
//         play_sim_state.l2_to_dram_queue_per_sub[sub_id] =
//             sub.l2_to_dram_queue().into_iter().map(Into::into).collect();
//         // .extend(sub.l2_to_dram_queue().into_iter().map(Into::into));
//
//         play_sim_state.l2_cache_per_sub[sub_id] = Some(testing::state::Cache {
//             lines: sub.l2_cache().lines().into_iter().map(Into::into).collect(),
//         });
//     }
//
//     // play_sim_state
// }

pub fn run_lockstep(trace_dir: &Path, trace_provider: TraceProvider) -> eyre::Result<()> {
    use accelsim::tracegen::reader::Command as AccelsimCommand;
    // let _ = color_eyre::install();

    let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));

    let box_trace_dir = trace_dir.join("trace");
    let accelsim_trace_dir = trace_dir.join("accelsim-trace");
    utils::fs::create_dirs(&box_trace_dir)?;
    utils::fs::create_dirs(&accelsim_trace_dir)?;

    let native_box_commands_path = box_trace_dir.join("commands.json");
    let native_accelsim_kernelslist_path = accelsim_trace_dir.join("kernelslist.g");

    let (box_commands_path, accelsim_kernelslist_path) = match trace_provider {
        TraceProvider::Native => {
            // use native traces
            (native_box_commands_path, native_accelsim_kernelslist_path)
        }
        TraceProvider::Accelsim => {
            assert!(native_accelsim_kernelslist_path.is_file());
            let generated_box_commands_path = box_trace_dir.join("accelsim.commands.json");
            println!(
                "generating commands {}",
                generated_box_commands_path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
            );

            let reader = utils::fs::open_readable(&native_accelsim_kernelslist_path)?;
            let accelsim_commands =
                accelsim::tracegen::reader::read_commands(&accelsim_trace_dir, reader)?;

            let commands: Vec<_> = accelsim_commands
                .into_iter()
                .map(|cmd| match cmd {
                    AccelsimCommand::MemcpyHtoD(memcopy) => {
                        Ok::<_, eyre::Report>(trace_model::Command::MemcpyHtoD(memcopy))
                    }
                    AccelsimCommand::KernelLaunch((mut kernel, metadata)) => {
                        // transform kernel instruction trace
                        let kernel_trace_path = accelsim_trace_dir.join(&kernel.trace_file);
                        let reader = utils::fs::open_readable(kernel_trace_path)?;
                        let parsed_trace = accelsim::tracegen::reader::read_trace_instructions(
                            reader,
                            metadata.trace_version,
                            metadata.line_info,
                            &kernel,
                        )?;

                        let generated_kernel_trace_name =
                            format!("accelsim-kernel-{}.msgpack", kernel.id);
                        let generated_kernel_trace_path =
                            box_trace_dir.join(&generated_kernel_trace_name);

                        let mut writer = utils::fs::open_writable(&generated_kernel_trace_path)?;
                        rmp_serde::encode::write(&mut writer, &parsed_trace)?;

                        // also save as json for inspection
                        let mut writer = utils::fs::open_writable(
                            generated_kernel_trace_path.with_extension("json"),
                        )?;
                        serde_json::to_writer_pretty(&mut writer, &parsed_trace)?;

                        // update the kernel trace path
                        kernel.trace_file = generated_kernel_trace_name;

                        Ok::<_, eyre::Report>(trace_model::Command::KernelLaunch(kernel))
                    }
                })
                .try_collect()?;

            let mut json_serializer = serde_json::Serializer::with_formatter(
                utils::fs::open_writable(&generated_box_commands_path)?,
                serde_json::ser::PrettyFormatter::with_indent(b"    "),
            );
            commands.serialize(&mut json_serializer)?;

            (
                generated_box_commands_path,
                native_accelsim_kernelslist_path,
            )
        }
        TraceProvider::Box => {
            assert!(native_box_commands_path.is_file());
            let generated_kernelslist_path = accelsim_trace_dir.join("box-kernelslist.g");
            println!(
                "generating commands {}",
                generated_kernelslist_path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
            );
            let mut commands_writer = utils::fs::open_writable(&generated_kernelslist_path)?;
            accelsim::tracegen::writer::generate_commands(
                &native_box_commands_path,
                &mut commands_writer,
            )?;
            drop(commands_writer);

            let reader = utils::fs::open_readable(&native_box_commands_path)?;
            let commands: Vec<Command> = serde_json::from_reader(reader)?;

            for cmd in commands {
                if let Command::KernelLaunch(kernel) = cmd {
                    // generate trace for kernel
                    let generated_kernel_trace_path = trace_dir.join(format!(
                        "accelsim-trace/kernel-{}.box.traceg",
                        kernel.id + 1
                    ));
                    println!(
                        "generating trace {} for kernel {}",
                        generated_kernel_trace_path
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy(),
                        kernel.id
                    );
                    let mut trace_writer = utils::fs::open_writable(generated_kernel_trace_path)?;
                    accelsim::tracegen::writer::generate_trace(
                        &box_trace_dir,
                        &kernel,
                        &mut trace_writer,
                    )?;
                }
            }
            (native_box_commands_path, generated_kernelslist_path)
        }
    };

    dbg!(&box_commands_path);
    dbg!(&accelsim_kernelslist_path);

    let gpgpusim_config = manifest_dir.join("accelsim/gtx1080/gpgpusim.config");
    let trace_config = manifest_dir.join("accelsim/gtx1080/gpgpusim.trace.config");
    let inter_config = manifest_dir.join("accelsim/gtx1080/config_fermi_islip.icnt");

    assert!(trace_dir.is_dir());
    assert!(box_trace_dir.is_dir());
    assert!(box_commands_path.is_file());
    assert!(accelsim_kernelslist_path.is_file());
    assert!(gpgpusim_config.is_file());
    assert!(trace_config.is_file());
    assert!(inter_config.is_file());

    // debugging config
    let box_config = Arc::new(config::GPUConfig {
        num_simt_clusters: 20,                   // 20
        num_cores_per_simt_cluster: 4,           // 1
        num_schedulers_per_core: 2,              // 2
        num_memory_controllers: 8,               // 8
        num_sub_partition_per_memory_channel: 2, // 2
        fill_l2_on_memcopy: true,                // true
        ..config::GPUConfig::default()
    });

    let box_interconn = Arc::new(ic::ToyInterconnect::new(
        box_config.num_simt_clusters,
        box_config.num_memory_controllers * box_config.num_sub_partition_per_memory_channel,
    ));

    let mut box_sim = crate::MockSimulator::new(box_interconn, box_config);
    box_sim.add_commands(&box_commands_path, &box_trace_dir)?;
    box_sim.parallel_simulation =
        std::env::var("PARALLEL").unwrap_or_default().to_lowercase() == "yes";

    let should_compare_states = !box_sim.parallel_simulation;

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
    let mut play_sim = playground::Accelsim::new(&play_config, &args)?;

    let mut play_time_cycle = Duration::ZERO;
    let mut play_time_other = Duration::ZERO;
    let mut box_time_cycle = Duration::ZERO;
    let mut box_time_other = Duration::ZERO;

    let mut gather_state_time = Duration::ZERO;
    let mut gather_box_state_time = Duration::ZERO;
    let mut gather_play_state_time = Duration::ZERO;

    let mut last_valid_box_sim_state = None;
    let mut last_valid_play_sim_state = None;

    let mut cycle = 0;

    let use_full_diff = std::env::var("FULL_DIFF")
        .unwrap_or_default()
        .to_lowercase()
        == "yes";
    let check_after: u64 = std::env::var("CHECK_AFTER")
        .ok()
        .as_deref()
        .map(str::parse)
        .transpose()?
        .unwrap_or(0);
    let check_every: u64 = std::env::var("CHECK_EVERY")
        .ok()
        .as_deref()
        .map(str::parse)
        .transpose()?
        .unwrap_or(200);
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

    box_sim.process_commands();
    box_sim.launch_kernels();

    while play_sim.commands_left() || play_sim.kernels_left() {
        let mut start = Instant::now();
        play_sim.process_commands();
        play_sim.launch_kernels();
        play_time_other += start.elapsed();

        // check that memcopy commands were handled correctly
        if false && should_compare_states {
            start = Instant::now();
            let (box_sim_state, play_sim_state) =
                gather_simulation_state(&mut box_sim, &mut play_sim, trace_provider);
            gather_state_time += start.elapsed();

            // start = Instant::now();
            // gather_box_simulation_state(&mut box_sim, &mut box_sim_state, trace_provider);
            // box_sim_state = gather_box_simulation_state(
            //     num_clusters,
            //     cores_per_cluster,
            //     num_partitions,
            //     num_sub_partitions,
            //     &mut box_sim,
            //     trace_provider,
            // );
            // gather_box_state_time += start.elapsed();
            // gather_state_time += start.elapsed();

            // start = Instant::now();
            // gather_play_simulation_state(&mut play_sim, &mut play_sim_state, trace_provider);
            // play_sim_state = gather_play_simulation_state(
            //     num_clusters,
            //     cores_per_cluster,
            //     num_partitions,
            //     num_sub_partitions,
            //     &mut play_sim,
            //     trace_provider,
            // );
            // gather_play_state_time += start.elapsed();
            // gather_state_time += start.elapsed();

            if use_full_diff {
                full_diff::assert_eq!(&box_sim_state, &play_sim_state);
            } else {
                diff::assert_eq!(box: &box_sim_state, play: &play_sim_state);
            }
        }

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
            cycle = play_sim.get_cycle();
            play_time_cycle += start.elapsed();

            start = Instant::now();
            box_sim.cycle();
            box_sim.set_cycle(cycle);
            box_time_cycle += start.elapsed();

            let should_check = cycle >= check_after && cycle % check_every == 0;
            if should_compare_states && should_check {
                start = Instant::now();
                let (box_sim_state, play_sim_state) =
                    gather_simulation_state(&mut box_sim, &mut play_sim, trace_provider);
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

                    {
                        serde_json::to_writer_pretty(
                            utils::fs::open_writable(
                                manifest_dir.join("debug.playground.state.json"),
                            )?,
                            &last_valid_play_sim_state,
                        )?;

                        // format!("{:#?}", ).as_bytes(),
                        serde_json::to_writer_pretty(
                            utils::fs::open_writable(manifest_dir.join("debug.box.state.json"))?,
                            &last_valid_box_sim_state,
                        )?;
                        // .write_all(format!("{:#?}", last_valid_box_sim_state).as_bytes())?;
                    };

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
                println!("checking for diff after cycle {cycle}");

                if use_full_diff {
                    full_diff::assert_eq!(&box_sim_state, &play_sim_state);
                } else {
                    diff::assert_eq!(box: &box_sim_state, play: &play_sim_state);
                }

                // this should be okay performance wise (copy, no allocation)
                last_valid_box_sim_state = Some(box_sim_state.clone());
                last_valid_play_sim_state = Some(play_sim_state.clone());
            }

            // box out of loop
            start = Instant::now();
            if !box_sim.active() {
                box_sim.process_commands();
                box_sim.launch_kernels();
            }

            if let Some(kernel) = box_sim.finished_kernel() {
                box_sim.cleanup_finished_kernel(&kernel);
            }
            box_time_other += start.elapsed();

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

    let play_cycle = cycle;
    let mut box_cycle = cycle;

    if box_sim.parallel_simulation {
        // allow parallel simulation to complete
        box_sim.run_to_completion()?;
        box_cycle = box_sim.cycle.get();
    }

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

    let num_checks = u32::try_from(cycle.saturating_sub(check_after) / check_every).unwrap();
    if !box_sim.parallel_simulation && num_checks > 0 {
        let gather_box_state_time = gather_box_state_time / num_checks;
        let gather_play_state_time = gather_play_state_time / num_checks;
        let gather_state_time = gather_state_time / num_checks;

        dbg!(gather_box_state_time);
        dbg!(gather_play_state_time);
        dbg!(gather_box_state_time + gather_play_state_time);
        dbg!(gather_state_time);
    }

    dbg!(&cycle);
    if box_sim.parallel_simulation {
        dbg!(&play_cycle);
        dbg!(&box_cycle);
    }

    let play_stats = play_sim.stats().clone();
    let box_stats = box_sim.stats();
    // let playground_dur = start.elapsed();

    // dbg!(&play_stats);
    // dbg!(&box_stats);

    // dbg!(&playground_dur);
    // dbg!(&box_dur);

    // compare stats here

    let play_l1i_stats = stats::PerCache::from_iter(play_stats.l1i_stats.to_vec());
    let play_l1d_stats = stats::PerCache::from_iter(play_stats.l1d_stats.to_vec());
    let play_l1t_stats = stats::PerCache::from_iter(play_stats.l1t_stats.to_vec());
    let play_l1c_stats = stats::PerCache::from_iter(play_stats.l1c_stats.to_vec());
    let play_l2d_stats = stats::PerCache::from_iter(play_stats.l2d_stats.to_vec());

    if box_sim.parallel_simulation {
        // compare reduced cache stats
        let max_rel_err = 0.05; // allow 5% difference
        {
            let play_l1i_stats = play_l1i_stats.reduce();
            let box_l1i_stats = box_stats.l1i_stats.reduce();
            dbg!(&play_l1i_stats, &box_l1i_stats);
            if play_l1i_stats != box_l1i_stats {
                diff::diff!(play: &play_l1i_stats, box: &box_l1i_stats);
            }
            let rel_err = super::stats::cache_rel_err(&play_l1i_stats, &box_l1i_stats);
            dbg!(&rel_err);
            assert!(rel_err.into_iter().all(|(_, err)| err <= max_rel_err));
        }
        {
            let play_l1d_stats = play_l1d_stats.reduce();
            let box_l1d_stats = box_stats.l1d_stats.reduce();
            dbg!(&play_l1d_stats, &box_l1d_stats);
            if play_l1d_stats != box_l1d_stats {
                diff::diff!(play: &play_l1d_stats, box: &box_l1d_stats);
            }
            let rel_err = super::stats::cache_rel_err(&play_l1d_stats, &box_l1d_stats);
            dbg!(&rel_err);
            assert!(rel_err.into_iter().all(|(_, err)| err <= max_rel_err));
        }
        {
            let play_l1t_stats = play_l1t_stats.reduce();
            let box_l1t_stats = box_stats.l1t_stats.reduce();
            dbg!(&play_l1t_stats, &box_l1t_stats);
            if play_l1t_stats != box_l1t_stats {
                diff::diff!(play: &play_l1t_stats, box: &box_l1t_stats);
            }
            let rel_err = super::stats::cache_rel_err(&play_l1t_stats, &box_l1t_stats);
            dbg!(&rel_err);
            assert!(rel_err.into_iter().all(|(_, err)| err <= max_rel_err));
        }
        {
            let play_l1c_stats = play_l1c_stats.reduce();
            let box_l1c_stats = box_stats.l1c_stats.reduce();
            dbg!(&play_l1c_stats, &box_l1c_stats);
            if play_l1c_stats != box_l1c_stats {
                diff::diff!(play: &play_l1c_stats, box: &box_l1c_stats);
            }
            let rel_err = super::stats::cache_rel_err(&play_l1c_stats, &box_l1c_stats);
            dbg!(&rel_err);
            assert!(rel_err.into_iter().all(|(_, err)| err <= max_rel_err));
        }
        {
            let play_l2d_stats = play_l2d_stats.reduce();
            let box_l2d_stats = box_stats.l2d_stats.reduce();
            dbg!(&play_l2d_stats, &box_l2d_stats);
            if play_l2d_stats != box_l2d_stats {
                diff::diff!(play: &play_l2d_stats, box: &box_l2d_stats);
            }
            let rel_err = super::stats::cache_rel_err(&play_l2d_stats, &box_l2d_stats);
            dbg!(&rel_err);
            assert!(rel_err.into_iter().all(|(_, err)| err <= max_rel_err));
        }

        {
            // compare DRAM stats
            let box_dram_stats = playground::stats::DRAM::from(box_stats.dram.clone());
            dbg!(&play_stats.dram, &box_dram_stats);
            if play_stats.dram != box_dram_stats {
                diff::diff!(play: &play_stats.dram, box: &box_dram_stats);
            }
            let rel_err = super::stats::dram_rel_err(&play_stats.dram, &box_dram_stats);
            dbg!(&rel_err);
            assert!(rel_err.into_iter().all(|(_, err)| err <= max_rel_err));
        }
    } else {
        dbg!(&play_l1i_stats, &box_stats.l1i_stats);
        diff::assert_eq!(play: &play_l1i_stats, box: &box_stats.l1i_stats);
        dbg!(&play_l1d_stats, &box_stats.l1d_stats);
        diff::assert_eq!( play: &play_l1d_stats, box: &box_stats.l1d_stats);
        dbg!(&play_l1t_stats, &box_stats.l1t_stats);
        diff::assert_eq!( play: &play_l1t_stats, box: &box_stats.l1t_stats);
        dbg!(&play_l1c_stats, &box_stats.l1c_stats);
        diff::assert_eq!( play: &play_l1c_stats, box: &box_stats.l1c_stats);
        dbg!(&play_l2d_stats, &box_stats.l2d_stats);
        diff::assert_eq!( play: &play_l2d_stats, box: &box_stats.l2d_stats);

        // compare DRAM stats
        let box_dram_stats = playground::stats::DRAM::from(box_stats.dram.clone());
        dbg!(&play_stats.dram, &box_dram_stats);
        diff::assert_eq!(play: &play_stats.dram, box: &box_dram_stats);
    }

    // compare accesses
    let box_accesses = playground::stats::Accesses::from(box_stats.accesses.clone());
    dbg!(&play_stats.accesses, &box_stats.accesses);
    diff::assert_eq!(play: play_stats.accesses, box: box_accesses);

    // compare instruction stats
    let box_instructions =
        playground::stats::InstructionCounts::from(box_stats.instructions.clone());
    dbg!(&play_stats.instructions, &box_instructions);
    diff::assert_eq!(play: &play_stats.instructions, box: &box_instructions);

    // compate simulation stats
    let box_sim_stats = playground::stats::Sim::from(box_stats.sim);
    dbg!(&play_stats.sim, &box_sim_stats);
    diff::assert_eq!(play: &play_stats.sim, box: &box_sim_stats);

    // this uses our custom PartialEq::eq implementation
    // assert_eq!(&play_stats, &box_stats);
    // assert!(false);
    Ok(())
}

macro_rules! lockstep_checks {
        ($($name:ident: $path:expr,)*) => {
            $(
                paste::paste! {
                    #[ignore = "native traces cannot be compared"]
                    #[test]
                    fn [<lockstep_native_ $name _test>]() -> color_eyre::eyre::Result<()> {
                        let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
                        let trace_dir = manifest_dir.join($path);
                        run_lockstep(&trace_dir, TraceProvider::Native)
                    }

                    #[test]
                    fn [<lockstep_accelsim_ $name _test>]() -> color_eyre::eyre::Result<()> {
                        let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
                        let trace_dir = manifest_dir.join($path);
                        run_lockstep(&trace_dir, TraceProvider::Accelsim)
                    }

                    #[test]
                    fn [<lockstep_box_ $name _test>]() -> color_eyre::eyre::Result<()> {
                        let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
                        let trace_dir = manifest_dir.join($path);
                        run_lockstep(&trace_dir, TraceProvider::Box)
                    }
                }
            )*
        }
    }

lockstep_checks! {
    // vectoradd
    vectoradd_32_100: "results/vectorAdd/vectorAdd-dtype-32-length-100",
    vectoradd_32_1000: "results/vectorAdd/vectorAdd-dtype-32-length-1000",
    vectoradd_32_10000: "results/vectorAdd/vectorAdd-dtype-32-length-10000",
    // simple matrixmul
    simple_matrixmul_32_32_32_32:
        "results/simple_matrixmul/simple_matrixmul-dtype-32-m-32-n-32-p-32",
    simple_matrixmul_32_32_32_64:
        "results/simple_matrixmul/simple_matrixmul-dtype-32-m-32-n-32-p-64",
    simple_matrixmul_32_64_128_128:
        "results/simple_matrixmul/simple_matrixmul-dtype-32-m-64-n-128-p-128",
    // matrixmul (shared memory)
    matrixmul_32_32: "results/matrixmul/matrixmul-dtype-32-rows-32",
    matrixmul_32_64: "results/matrixmul/matrixmul-dtype-32-rows-64",
    matrixmul_32_128: "results/matrixmul/matrixmul-dtype-32-rows-128",
    matrixmul_32_256: "results/matrixmul/matrixmul-dtype-32-rows-256",
    // transpose
    transpose_256_naive: "results/transpose/transpose-dim-256-variant-naive-repeat-1",
    transpose_256_coalesced: "results/transpose/transpose-dim-256-variant-coalesced-repeat-1",
    transpose_256_optimized: "results/transpose/transpose-dim-256-variant-optimized-repeat-1",
}
