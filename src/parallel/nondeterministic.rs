#![allow(warnings, clippy::all)]

use crate::ic::ToyInterconnect;
use crate::sync::{Arc, Mutex, RwLock};
use crate::{
    config, core, engine::cycle::Component, ic, kernel::Kernel, mem_fetch, mem_sub_partition,
    MockSimulator,
};
use color_eyre::eyre;
use ndarray::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

#[tracing::instrument]
// #[inline]
fn interleaved_serial_cycle<I, C>(
    cycle: u64,
    active_clusters: &Vec<bool>,
    cores: &Arc<Vec<Vec<Arc<RwLock<crate::core::Core<I>>>>>>,
    sim_orders: &Arc<Vec<Arc<Mutex<VecDeque<usize>>>>>,
    mem_ports: &Arc<Vec<Vec<Arc<Mutex<crate::core::CoreMemoryConnection<C>>>>>>,
    interconn: &Arc<I>,
    clusters: &Vec<Arc<crate::Cluster<I>>>,
    config: &config::GPU,
) where
    C: ic::BufferedConnection<ic::Packet<(usize, mem_fetch::MemFetch, u32)>>,
    I: ic::Interconnect<ic::Packet<mem_fetch::MemFetch>>,
{
    let use_round_robin = config.simt_core_sim_order == config::SchedulingOrder::RoundRobin;

    for (cluster_id, _cluster) in clusters.iter().enumerate() {
        let cluster_active = active_clusters[cluster_id];
        let mut core_sim_order = sim_orders[cluster_id].try_lock();
        for core_id in &*core_sim_order {
            let mut port = mem_ports[cluster_id][*core_id].lock();
            if cluster_active {
                if !port.buffer.is_empty() {}
                // assert!(port.buffer.is_empty());
            }

            for ic::Packet {
                data: (dest, fetch, size),
                time,
            } in port.buffer.drain()
            {
                interconn.push(cluster_id, dest, ic::Packet { data: fetch, time }, size);
            }
        }

        if cluster_active {
            if use_round_robin {
                core_sim_order.rotate_left(1);
            }
        } else {
            // println!(
            //     "SERIAL: cluster {} not updated in cycle {}",
            //     cluster.cluster_id,
            //     cycle + i as u64
            // );
        }
    }
}

#[tracing::instrument]
// #[inline]
fn new_serial_cycle<I>(
    cycle: u64,
    stats: &Arc<Mutex<stats::PerKernel>>,
    need_issue_lock: &Arc<RwLock<Vec<Vec<(bool, bool)>>>>,
    last_issued_kernel: &Arc<Mutex<usize>>,
    block_issue_next_core: &Arc<Vec<Mutex<usize>>>,
    running_kernels: &Arc<RwLock<Vec<Option<(usize, Arc<dyn Kernel>)>>>>,
    executed_kernels: &Arc<Mutex<HashMap<u64, Arc<dyn Kernel>>>>,
    mem_sub_partitions: &Arc<Vec<Arc<Mutex<crate::mem_sub_partition::MemorySubPartition>>>>,
    mem_partition_units: &Arc<Vec<Arc<RwLock<crate::mem_partition_unit::MemoryPartitionUnit>>>>,
    interconn: &Arc<I>,
    clusters: &Arc<Vec<Arc<crate::Cluster<I>>>>,
    cores: &Arc<Vec<Vec<Arc<RwLock<crate::Core<I>>>>>>,
    last_cluster_issue: &Arc<Mutex<usize>>,
    config: &config::GPU,
) where
    I: ic::Interconnect<ic::Packet<mem_fetch::MemFetch>>,
{
    // it could happen that two serial cycles overlap when using spawn fifo, so we need
    for cluster in clusters.iter() {
        crate::timeit!("serial::interconn", cluster.interconn_cycle(cycle));
    }

    for (i, mem_sub) in mem_sub_partitions.iter().enumerate() {
        let mut mem_sub = mem_sub.try_lock();
        if let Some(fetch) = mem_sub.top() {
            let response_packet_size = if fetch.is_write() {
                fetch.control_size()
            } else {
                fetch.size()
            };
            let device = config.mem_id_to_device_id(i);
            if interconn.has_buffer(device, response_packet_size) {
                let mut fetch = mem_sub.pop().unwrap();
                if let Some(cluster_id) = fetch.cluster_id {
                    fetch.set_status(mem_fetch::Status::IN_ICNT_TO_SHADER, 0);
                    interconn.push(
                        device,
                        cluster_id,
                        ic::Packet {
                            data: fetch,
                            time: cycle,
                        },
                        response_packet_size,
                    );
                }
            }
        }
    }

    for (_i, unit) in mem_partition_units.iter().enumerate() {
        crate::timeit!("serial::dram", unit.try_write().simple_dram_cycle(cycle));
    }

    for (i, mem_sub) in mem_sub_partitions.iter().enumerate() {
        let mut mem_sub = mem_sub.try_lock();
        // move memory request from interconnect into memory partition
        // (if not backed up)
        //
        // Note:This needs to be called in DRAM clock domain if there
        // is no L2 cache in the system In the worst case, we may need
        // to push SECTOR_CHUNCK_SIZE requests, so ensure you have enough
        // buffer for them
        let device = config.mem_id_to_device_id(i);

        // same as full with parameter overload
        if mem_sub
            .interconn_to_l2_queue
            .can_fit(mem_sub_partition::SECTOR_CHUNK_SIZE as usize)
        {
            if let Some(packet) = interconn.pop(device) {
                log::debug!(
                    "got new fetch {} for mem sub partition {} ({})",
                    packet.data,
                    i,
                    device
                );

                // changed from packet.time to cycle here
                // mem_sub.push(packet.data, packet.time);
                mem_sub.push(packet.data, cycle);
            }
        } else {
            log::debug!("SKIP sub partition {} ({}): DRAM full stall", i, device);
            // TODO
            // if let Some(kernel) = &*mem_sub.current_kernel.lock() {
            //     let mut stats = stats.lock();
            //     let kernel_stats = stats.get_mut(kernel.id() as usize);
            //     kernel_stats.stall_dram_full += 1;
            // }
        }
        // we borrow all of sub here, which is a problem for the cyclic reference in l2
        // interface
        crate::timeit!("serial::subpartitions", mem_sub.cycle(cycle));
    }
}

impl<I> MockSimulator<I>
where
    I: ic::Interconnect<ic::Packet<mem_fetch::MemFetch>>,
{
    #[tracing::instrument]
    pub fn run_to_completion_parallel_nondeterministic(
        &mut self,
        mut run_ahead: usize,
    ) -> eyre::Result<()> {
        run_ahead = run_ahead.max(1);

        let interleave_serial = true;
        // interleave_serial |= std::env::var("INTERLEAVE")
        //     .unwrap_or_default()
        //     .to_lowercase()
        //     == "yes";

        let num_threads = super::get_num_threads()?
            .or(self.config.simulation_threads)
            .unwrap_or_else(num_cpus::get_physical);

        super::rayon_pool(num_threads)?.install(|| {
            println!("nondeterministic interleaved [{run_ahead} run ahead] using RAYON");
            println!(
                "\t => launching {num_threads} worker threads for {} cores",
                self.config.total_cores()
            );
            // println!("\t => interleave serial={interleave_serial}");
            println!("");

            let sim_orders: Vec<Arc<_>> = self
                .clusters
                .iter()
                .map(|cluster| Arc::clone(&cluster.core_sim_order))
                .collect();
            let mem_ports: Vec<Vec<Arc<_>>> = self
                .clusters
                .iter()
                .map(|cluster| {
                    cluster
                        .cores
                        .iter()
                        .map(|core| Arc::clone(&core.try_read().mem_port))
                        .collect()
                })
                .collect();
            let cores: Vec<Vec<Arc<_>>> = self
                .clusters
                .iter()
                .map(|cluster| cluster.cores.clone())
                .collect();

            let last_cycle: Arc<Mutex<Vec<Vec<u64>>>> = Arc::new(Mutex::new(
                self.clusters
                    .iter()
                    .map(|cluster| {
                        let num_cores = cluster.cores.len();
                        vec![0; num_cores]
                    })
                    .collect(),
            ));

            let num_clusters = self.clusters.len();
            let num_cores_per_cluster = self.clusters[0].cores.len();
            let shape = (run_ahead, num_clusters, num_cores_per_cluster);
            let progress = Array3::<Option<bool>>::from_elem(shape, None);

            let progress = Arc::new(Mutex::new(progress));
            let clusters = Arc::new(self.clusters.clone());
            let cores = Arc::new(cores);
            let sim_orders = Arc::new(sim_orders);
            let mem_ports = Arc::new(mem_ports);
            let mem_sub_partitions = Arc::new(self.mem_sub_partitions.clone());
            let mem_partition_units = Arc::new(self.mem_partition_units.clone());

            let use_round_robin =
                self.config.simt_core_sim_order == config::SchedulingOrder::RoundRobin;

            let last_issued_kernel = Arc::new(Mutex::new(0));
            let block_issue_next_core = Arc::new(
                (0..num_clusters)
                    .into_iter()
                    .map(|_| Mutex::new(num_cores_per_cluster - 1))
                    .collect(),
            );
            let need_issue: Arc<RwLock<Vec<Vec<_>>>> = Arc::new(RwLock::new(vec![
                vec![(false, false); num_cores_per_cluster];
                num_clusters
            ]));

            let serial_lock = Arc::new(Mutex::new(()));
            let issue_guard = Arc::new(Mutex::new(()));

            let mut cycle: u64 = 0;
            let log_every = 10_000;
            let mut last_time = std::time::Instant::now();

            let mut active_clusters = vec![false; num_clusters];

            while (self.commands_left() || self.kernels_left()) && !self.reached_limit(cycle) {
                cycle = self.process_commands(cycle);
                self.launch_kernels(cycle);

                let start_cycle = cycle;

                let mut finished_kernel = None;
                loop {
                    log::info!("======== cycle {cycle} ========");
                    log::info!("");
                    if (cycle - start_cycle) % log_every == 0 && (cycle - start_cycle) > 0 {
                        eprintln!(
                            "cycle {cycle:<10} ({:>8.4} cycle/sec)",
                            log_every as f64 / last_time.elapsed().as_secs_f64()
                        );
                        last_time = std::time::Instant::now()
                    }

                    // if self.reached_limit(cycle) || !self.active() {
                    if self.reached_limit(cycle) {
                        break;
                    }

                    let span = tracing::span!(tracing::Level::INFO, "wave", cycle, run_ahead);
                    let enter = span.enter();

                    if interleave_serial {
                        rayon::scope_fifo(|wave| {
                            for i in 0..run_ahead {
                                for (cluster_id, _cluster_arc) in
                                    clusters.iter().cloned().enumerate()
                                {
                                    for (core_id, core) in
                                        cores[cluster_id].iter().cloned().enumerate()
                                    {
                                        let progress = Arc::clone(&progress);

                                        let sim_orders = Arc::clone(&sim_orders);
                                        let mem_ports = Arc::clone(&mem_ports);
                                        let cores = Arc::clone(&cores);

                                        let interconn = Arc::clone(&self.interconn);
                                        let clusters = Arc::clone(&clusters);

                                        let stats = Arc::clone(&self.stats);
                                        let mem_sub_partitions = Arc::clone(&mem_sub_partitions);
                                        let mem_partition_units = Arc::clone(&mem_partition_units);
                                        let config = Arc::clone(&self.config);

                                        let running_kernels = Arc::clone(&self.running_kernels);
                                        let executed_kernels = Arc::clone(&self.executed_kernels);

                                        let last_cluster_issue =
                                            Arc::clone(&self.last_cluster_issue);
                                        let last_issued_kernel = Arc::clone(&last_issued_kernel);
                                        let block_issue_next_core =
                                            Arc::clone(&block_issue_next_core);
                                        let need_issue = Arc::clone(&need_issue);
                                        let issue_guard = Arc::clone(&issue_guard);
                                        let serial_lock = Arc::clone(&serial_lock);

                                        wave.spawn_fifo(move |_| {
                                            let mut core = core.write();

                                            let kernels_completed = running_kernels
                                                .try_read()
                                                .iter()
                                                .filter_map(Option::as_ref)
                                                .all(|(_, k)| k.no_more_blocks_to_run());

                                            let core_active =
                                                !(kernels_completed && core.not_completed() == 0);

                                            if core_active {
                                                crate::timeit!(
                                                    "core::cycle",
                                                    core.cycle(cycle + i as u64)
                                                );
                                            }

                                            drop(core);

                                            let last_to_finish = {
                                                let mut progress = progress.lock();
                                                progress[[i, cluster_id, core_id]] =
                                                    Some(core_active);

                                                let res = progress
                                                    .slice(s![i, .., ..])
                                                    .iter()
                                                    .all(|&c| c.is_some());
                                                res
                                            };

                                            if last_to_finish {
                                                let guard = serial_lock.lock();
                                                let ready_serial_i = {
                                                    let mut progress = progress.lock();
                                                    let ready: Vec<_> = (0..run_ahead)
                                                        .into_iter()
                                                        .skip_while(|&si| {
                                                            !progress
                                                                .slice(s![si, .., ..])
                                                                .iter()
                                                                .all(|&c| c.is_some())
                                                        })
                                                        .take_while(|&si| {
                                                            progress
                                                                .slice(s![si, .., ..])
                                                                .iter()
                                                                .all(|&c| c.is_some())
                                                        })
                                                        .map(|si| {
                                                            let active_clusters: Vec<_> = progress
                                                                .slice(s![si, .., ..])
                                                                .axis_iter(Axis(0))
                                                                .map(|cluster_cores| {
                                                                    cluster_cores
                                                                        .iter()
                                                                        .any(|&c| c == Some(true))
                                                                })
                                                                .collect();

                                                            assert_eq!(
                                                                active_clusters.len(),
                                                                num_clusters
                                                            );

                                                            (si, active_clusters)
                                                        })
                                                        .collect();

                                                    for (ri, _) in &ready {
                                                        progress
                                                            .slice_mut(s![*ri, .., ..])
                                                            .fill(None);
                                                    }
                                                    ready
                                                };

                                                for (i, active_clusters) in ready_serial_i {
                                                    crate::timeit!(
                                                        "serial::postcore",
                                                        interleaved_serial_cycle(
                                                            cycle + i as u64,
                                                            &active_clusters,
                                                            &cores,
                                                            &sim_orders,
                                                            &mem_ports,
                                                            &interconn,
                                                            &clusters,
                                                            &config,
                                                        )
                                                    );
                                                    // if (cycle + i as u64) % 4 == 0 {
                                                    crate::timeit!(
                                                        "serial::cycle",
                                                        new_serial_cycle(
                                                            cycle + i as u64,
                                                            &stats,
                                                            &need_issue,
                                                            &last_issued_kernel,
                                                            &block_issue_next_core,
                                                            &running_kernels,
                                                            &executed_kernels,
                                                            &mem_sub_partitions,
                                                            &mem_partition_units,
                                                            &interconn,
                                                            &clusters,
                                                            &cores,
                                                            &last_cluster_issue,
                                                            &config,
                                                        )
                                                    );
                                                    // }
                                                }

                                                drop(guard);
                                            }
                                        });
                                    }
                                }
                            }
                        });
                        // all run_ahead cycles completed
                        progress.lock().fill(None);
                        crate::timeit!("cycle::issue_blocks", self.issue_block_to_core(cycle));
                    }

                    if !interleave_serial {
                        rayon::scope_fifo(|wave| {
                            for i in 0..run_ahead {
                                for (cluster_id, _cluster_arc) in
                                    clusters.iter().cloned().enumerate()
                                {
                                    for (core_id, core) in
                                        cores[cluster_id].iter().cloned().enumerate()
                                    {
                                        let running_kernels = Arc::clone(&self.running_kernels);
                                        wave.spawn_fifo(move |_| {
                                            let mut core = core.write();

                                            // let kernels_completed = running_kernels
                                            //     .try_read()
                                            //     .iter()
                                            //     .filter_map(Option::as_ref)
                                            //     .all(|(_, k)| k.no_more_blocks_to_run());
                                            //
                                            // // let core_active = core.not_completed() != 0;
                                            // let core_active =
                                            //     !(kernels_completed && core.not_completed() == 0);
                                            //
                                            // if core_active {
                                            //     core.cycle(cycle + i as u64);
                                            //     core.last_cycle =
                                            //         core.last_cycle.max(cycle + i as u64);
                                            //     core.last_active_cycle =
                                            //         core.last_active_cycle.max(cycle + i as u64);
                                            // } else {
                                            //     core.last_cycle =
                                            //         core.last_cycle.max(cycle + i as u64);
                                            // }
                                            // if core_active {
                                            //     core.cycle(cycle + i as u64);
                                            // }
                                            crate::timeit!(
                                                "core::cycle",
                                                core.cycle(cycle + i as u64)
                                            );
                                        });
                                    }
                                }

                                // let progress = Arc::clone(&progress);
                                //
                                // let sim_orders = Arc::clone(&sim_orders);
                                // let mem_ports = Arc::clone(&mem_ports);
                                // let cores = Arc::clone(&cores);
                                //
                                // let interconn = Arc::clone(&self.interconn);
                                // let clusters = Arc::clone(&clusters);
                                //
                                // let stats = Arc::clone(&self.stats);
                                // let mem_sub_partitions = Arc::clone(&mem_sub_partitions);
                                // let mem_partition_units = Arc::clone(&mem_partition_units);
                                // let config = Arc::clone(&self.config);
                                //
                                // let executed_kernels = Arc::clone(&self.executed_kernels);
                                // let running_kernels = Arc::clone(&self.running_kernels);
                                //
                                // let last_cluster_issue = Arc::clone(&self.last_cluster_issue);
                                // let last_issued_kernel = Arc::clone(&last_issued_kernel);
                                // let block_issue_next_core = Arc::clone(&block_issue_next_core);
                                // let need_issue = Arc::clone(&need_issue);
                                // let issue_guard = Arc::clone(&issue_guard);
                                // let serial_lock = Arc::clone(&serial_lock);
                                //
                                // let active_clusters = vec![true; self.config.num_simt_clusters];
                                //
                                // wave.spawn_fifo(move |_| {
                                //     let guard = serial_lock.lock();
                                //     crate::timeit!(
                                //         "serial::postcore",
                                //         interleaved_serial_cycle(
                                //             cycle + i as u64,
                                //             &active_clusters,
                                //             &cores,
                                //             &sim_orders,
                                //             &mem_ports,
                                //             &interconn,
                                //             &clusters,
                                //             &config,
                                //         )
                                //     );
                                // });
                            }
                        });

                        let progress = Arc::clone(&progress);

                        let sim_orders = Arc::clone(&sim_orders);
                        let mem_ports = Arc::clone(&mem_ports);
                        let cores = Arc::clone(&cores);

                        let interconn = Arc::clone(&self.interconn);
                        let clusters = Arc::clone(&clusters);

                        let stats = Arc::clone(&self.stats);
                        let mem_sub_partitions = Arc::clone(&mem_sub_partitions);
                        let mem_partition_units = Arc::clone(&mem_partition_units);
                        let config = Arc::clone(&self.config);

                        let executed_kernels = Arc::clone(&self.executed_kernels);
                        let running_kernels = Arc::clone(&self.running_kernels);

                        let last_cluster_issue = Arc::clone(&self.last_cluster_issue);
                        let last_issued_kernel = Arc::clone(&last_issued_kernel);
                        let block_issue_next_core = Arc::clone(&block_issue_next_core);
                        let need_issue = Arc::clone(&need_issue);
                        let issue_guard = Arc::clone(&issue_guard);
                        let serial_lock = Arc::clone(&serial_lock);

                        let active_clusters = vec![true; self.config.num_simt_clusters];
                        //
                        // wave.spawn_fifo(move |_| {
                        //     let guard = serial_lock.lock();
                        //     crate::timeit!(
                        //         "serial::postcore",
                        //         interleaved_serial_cycle(
                        //             cycle + i as u64,
                        //             &active_clusters,
                        //             &cores,
                        //             &sim_orders,
                        //             &mem_ports,
                        //             &interconn,
                        //             &clusters,
                        //             &config,
                        //         )
                        //     );
                        //     if (cycle + i as u64) % 2 == 0 {
                        for i in 0..run_ahead {
                            crate::timeit!(
                                "serial::postcore",
                                interleaved_serial_cycle(
                                    cycle + i as u64,
                                    &active_clusters,
                                    &cores,
                                    &sim_orders,
                                    &mem_ports,
                                    &interconn,
                                    &clusters,
                                    &config,
                                )
                            );

                            crate::timeit!(
                                "serial::cycle",
                                new_serial_cycle(
                                    cycle + i as u64,
                                    &stats,
                                    &need_issue,
                                    &last_issued_kernel,
                                    &block_issue_next_core,
                                    &running_kernels,
                                    &executed_kernels,
                                    &mem_sub_partitions,
                                    &mem_partition_units,
                                    &interconn,
                                    &clusters,
                                    &cores,
                                    &last_cluster_issue,
                                    &config,
                                )
                            );
                            //         }
                            //         drop(guard);
                            //     });
                            // }
                            // });
                            crate::timeit!("issue blocks", self.issue_block_to_core(cycle));
                        }
                        //     for i in 0..run_ahead {
                        //         // run cores in any order
                        //         rayon::scope(|core_scope| {
                        //             for cluster_arc in &self.clusters {
                        //                 let cluster = cluster_arc.try_read();
                        //                 let kernels_completed = self
                        //                     .running_kernels
                        //                     .try_read()
                        //                     .iter()
                        //                     .filter_map(std::option::Option::as_ref)
                        //                     .all(|(_, k)| k.no_more_blocks_to_run());
                        //
                        //                 let cores_completed = cluster.not_completed() == 0;
                        //                 active_clusters[cluster.cluster_id] =
                        //                     !(cores_completed && kernels_completed);
                        //
                        //                 if cores_completed && kernels_completed {
                        //                     continue;
                        //                 }
                        //                 for core in cluster.cores.iter().cloned() {
                        //                     core_scope.spawn(move |_| {
                        //                         let mut core = core.write();
                        //                         core.cycle(cycle + i as u64);
                        //                     });
                        //                 }
                        //             }
                        //         });
                        //
                        //         // push memory request packets generated by cores to the interconnection network.
                        //         for (cluster_id, cluster) in self.clusters.iter().enumerate() {
                        //             let cluster = cluster.try_read();
                        //             assert_eq!(cluster.cluster_id, cluster_id);
                        //
                        //             let mut core_sim_order = cluster.core_sim_order.try_lock();
                        //             for core_id in &*core_sim_order {
                        //                 let core = cluster.cores[*core_id].try_read();
                        //                 // was_updated |= core.last_active_cycle >= (cycle + i as u64);
                        //
                        //                 let mut port = core.mem_port.lock();
                        //                 if !active_clusters[cluster_id] {
                        //                     assert!(port.buffer.is_empty());
                        //                 }
                        //                 for ic::Packet {
                        //                     data: (dest, fetch, size),
                        //                     time,
                        //                 } in port.buffer.drain(..)
                        //                 {
                        //                     self.interconn.push(
                        //                         cluster_id,
                        //                         dest,
                        //                         ic::Packet { data: fetch, time },
                        //                         size,
                        //                     );
                        //                 }
                        //             }
                        //
                        //             if active_clusters[cluster_id] {
                        //                 if use_round_robin {
                        //                     core_sim_order.rotate_left(1);
                        //                 }
                        //             } else {
                        //                 // println!(
                        //                 //     "cluster {} not updated in cycle {}",
                        //                 //     cluster.cluster_id,
                        //                 //     cycle + i as u64
                        //                 // );
                        //             }
                        //         }
                        //
                        //         // after cores complete, run serial cycle
                        //         self.serial_cycle(cycle + i as u64);
                        //     }
                        //     crate::timeit!("issue blocks", self.issue_block_to_core(cycle));
                    }

                    cycle += run_ahead as u64;

                    drop(enter);

                    if !self.active() {
                        finished_kernel = self.finished_kernel();
                        if finished_kernel.is_some() {
                            break;
                        }
                    }
                }

                if let Some(kernel) = finished_kernel {
                    self.cleanup_finished_kernel(&*kernel, cycle);
                }

                log::trace!(
                    "commands left={} kernels left={}",
                    self.commands_left(),
                    self.kernels_left()
                );
            }

            self.stats.lock().no_kernel.sim.cycles = cycle;

            if let Some(log_after_cycle) = self.log_after_cycle {
                if log_after_cycle >= cycle {
                    eprintln!("WARNING: log after {log_after_cycle} cycles but simulation ended after {cycle} cycles");
                }
            }

            log::info!("exit after {cycle} cycles");
            dbg!(&cycle);
            Ok::<_, eyre::Report>(())
        })
    }

    #[tracing::instrument]
    pub fn serial_cycle(&mut self, cycle: u64) {
        for cluster in &self.clusters {
            // Receive memory responses addressed to each cluster and forward to cores
            crate::timeit!("serial::interconn", cluster.interconn_cycle(cycle));
        }

        // send memory responses from memory sub partitions to the requestor clusters via
        // interconnect
        for (i, mem_sub) in self.mem_sub_partitions.iter().enumerate() {
            let mut mem_sub = mem_sub.try_lock();
            if let Some(fetch) = mem_sub.top() {
                let response_packet_size = if fetch.is_write() {
                    fetch.control_size()
                } else {
                    fetch.size()
                };
                let device = self.config.mem_id_to_device_id(i);
                if self.interconn.has_buffer(device, response_packet_size) {
                    let mut fetch = mem_sub.pop().unwrap();
                    if let Some(cluster_id) = fetch.cluster_id {
                        fetch.set_status(mem_fetch::Status::IN_ICNT_TO_SHADER, 0);
                        self.interconn.push(
                            device,
                            cluster_id,
                            ic::Packet {
                                data: fetch,
                                time: cycle,
                            },
                            response_packet_size,
                        );
                    }
                }
            }
        }

        // dram cycle
        for (_i, unit) in self.mem_partition_units.iter().enumerate() {
            crate::timeit!("serial::dram", unit.try_write().simple_dram_cycle(cycle));
        }

        // receive requests sent to L2 from interconnect and push them to the
        // targeted memory sub partition.
        // Run cycle for each sub partition
        for (i, mem_sub) in self.mem_sub_partitions.iter().enumerate() {
            let mut mem_sub = mem_sub.try_lock();
            // move memory request from interconnect into memory partition
            // (if not backed up)
            //
            // Note:This needs to be called in DRAM clock domain if there
            // is no L2 cache in the system In the worst case, we may need
            // to push SECTOR_CHUNCK_SIZE requests, so ensure you have enough
            // buffer for them
            let device = self.config.mem_id_to_device_id(i);

            if mem_sub
                .interconn_to_l2_queue
                .can_fit(mem_sub_partition::SECTOR_CHUNK_SIZE as usize)
            {
                if let Some(packet) = self.interconn.pop(device) {
                    log::debug!(
                        "got new fetch {} for mem sub partition {} ({})",
                        packet.data,
                        i,
                        device
                    );

                    // assert_eq!(cycle, packet.time);
                    // TODO: changed form packet.time to cycle
                    mem_sub.push(packet.data, cycle);
                }
            } else {
                log::debug!("SKIP sub partition {} ({}): DRAM full stall", i, device);
                let kernel_id = self
                    .current_kernel
                    .lock()
                    .as_ref()
                    .map(|kernel| kernel.id() as usize);
                let mut stats = self.stats.lock();
                let kernel_stats = stats.get_mut(kernel_id);
                kernel_stats.stall_dram_full += 1;
            }
            // we borrow all of sub here, which is a problem for the cyclic reference in l2
            // interface
            crate::timeit!("serial::subpartitions", mem_sub.cycle(cycle));
        }
    }
}
