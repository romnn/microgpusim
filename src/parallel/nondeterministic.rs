#![allow(warnings, clippy::all)]

use crate::ic::ToyInterconnect;
use crate::sync::{Arc, Mutex, RwLock};
use crate::{
    config, core, engine::cycle::Component, ic, mem_fetch, mem_sub_partition, MockSimulator,
};
use color_eyre::eyre;
use ndarray::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

#[tracing::instrument]
#[inline]
fn interleaved_serial_cycle<I, C>(
    cycle: u64,
    // i: usize,
    active_clusters: &Vec<bool>,
    // progress: &Arc<Mutex<Array3<Option<bool>>>>,
    // progress: &Array3<Option<bool>>,
    cores: &Arc<Vec<Vec<Arc<RwLock<crate::core::Core<I>>>>>>,
    sim_orders: &Arc<Vec<Arc<Mutex<VecDeque<usize>>>>>,
    mem_ports: &Arc<Vec<Vec<Arc<Mutex<crate::core::CoreMemoryConnection<C>>>>>>,
    interconn: &Arc<I>,
    clusters: &Vec<Arc<RwLock<crate::Cluster<I>>>>,
    config: &config::GPU,
) where
    C: ic::BufferedConnection<ic::Packet<(usize, mem_fetch::MemFetch, u32)>>,
    I: ic::Interconnect<ic::Packet<mem_fetch::MemFetch>>,
{
    let use_round_robin = config.simt_core_sim_order == config::SchedulingOrder::RoundRobin;

    for (cluster_id, _cluster) in clusters.iter().enumerate() {
        let cluster_active = active_clusters[cluster_id];
        // let cluster_active = {
        //     let progress = progress.lock();
        //     progress
        //         .slice(s![i, cluster_id, ..])
        //         .iter()
        //         .any(|&c| c == Some(true))
        // };
        // if cluster_active {
        // println!(
        //     "PARALLEL: cluster {} is active in cycle {} ({})",
        //     cluster_id, i, cycle
        // );
        // }
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
#[inline]
fn new_serial_cycle<I>(
    cycle: u64,
    stats: &Arc<Mutex<stats::Stats>>,
    // need_issue: &Arc<Mutex<Vec<Vec<(bool, bool)>>>>,
    need_issue_lock: &Arc<RwLock<Vec<Vec<(bool, bool)>>>>,
    last_issued_kernel: &Arc<Mutex<usize>>,
    block_issue_next_core: &Arc<Vec<Mutex<usize>>>,
    running_kernels: &Arc<RwLock<Vec<Option<Arc<crate::Kernel>>>>>,
    executed_kernels: &Arc<Mutex<HashMap<u64, String>>>,
    mem_sub_partitions: &Arc<Vec<Arc<Mutex<crate::mem_sub_partition::MemorySubPartition>>>>,
    mem_partition_units: &Arc<Vec<Arc<RwLock<crate::mem_partition_unit::MemoryPartitionUnit>>>>,
    interconn: &Arc<I>,
    clusters: &Arc<Vec<Arc<RwLock<crate::Cluster<I>>>>>,
    cores: &Arc<Vec<Vec<Arc<RwLock<crate::Core<I>>>>>>,
    last_cluster_issue: &Arc<Mutex<usize>>,
    config: &config::GPU,
) where
    I: ic::Interconnect<ic::Packet<mem_fetch::MemFetch>>,
{
    // it could happen that two serial cycles overlap when using spawn fifo, so we need
    for cluster in clusters.iter() {
        cluster.write().interconn_cycle(cycle);
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
                let cluster_id = fetch.cluster_id;
                fetch.set_status(mem_fetch::Status::IN_ICNT_TO_SHADER, 0);
                // fetch.set_return_timestamp(gpu_sim_cycle + gpu_tot_sim_cycle);
                // , gpu_sim_cycle + gpu_tot_sim_cycle);
                // drop(fetch);
                interconn.push(
                    device,
                    cluster_id,
                    ic::Packet {
                        data: fetch,
                        time: cycle,
                    },
                    response_packet_size,
                );
                // self.partition_replies_in_parallel += 1;
            } else {
                // self.gpu_stall_icnt2sh += 1;
            }
        }
    }

    for (_i, unit) in mem_partition_units.iter().enumerate() {
        unit.try_write().simple_dram_cycle(cycle);
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
            .can_fit(mem_sub_partition::SECTOR_CHUNCK_SIZE as usize)
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
                // self.parallel_mem_partition_reqs += 1;
            }
        } else {
            log::debug!("SKIP sub partition {} ({}): DRAM full stall", i, device);
            stats.lock().stall_dram_full += 1;
        }
        // we borrow all of sub here, which is a problem for the cyclic reference in l2
        // interface
        mem_sub.cache_cycle(cycle);
    }

    if false {
        log::debug!("===> issue block to core");
        let mut last_cluster_issue = last_cluster_issue.try_lock();
        let last_issued = *last_cluster_issue;
        let num_clusters = config.num_simt_clusters;
        let need_issue = need_issue_lock.read();

        let mut issued = Vec::new();

        for cluster_id in 0..num_clusters {
            let cluster_id = (cluster_id + last_issued + 1) % num_clusters;
            let num_cores = need_issue[cluster_id].len();

            log::debug!(
                "cluster {}: issue block to core for {} cores",
                cluster_id,
                num_cores
            );
            let mut num_blocks_issued = 0;

            let mut block_issue_next_core = block_issue_next_core[cluster_id].try_lock();

            for core_id in 0..num_cores {
                let core_id = (core_id + *block_issue_next_core + 1) % num_cores;
                let (want_more_blocks, should_select_new_kernel) = need_issue[cluster_id][core_id];

                if !(want_more_blocks || should_select_new_kernel) {
                    continue;
                }

                let mut core = cores[cluster_id][core_id].write();
                let mut current_kernel = core.current_kernel.lock().clone();

                if should_select_new_kernel {
                    current_kernel = (|| {
                        let mut last_issued_kernel = last_issued_kernel.lock();
                        let mut executed_kernels = executed_kernels.lock();
                        let running_kernels = running_kernels.read();

                        // issue same kernel again
                        match running_kernels[*last_issued_kernel] {
                            // && !kernel.kernel_TB_latency)
                            Some(ref last_kernel) if !last_kernel.no_more_blocks_to_run() => {
                                let launch_id = last_kernel.id();
                                executed_kernels
                                    .entry(launch_id)
                                    .or_insert(last_kernel.name().to_string());
                                return Some(last_kernel.clone());
                            }
                            _ => {}
                        };

                        // issue new kernel
                        let num_kernels = running_kernels.len();
                        let max_concurrent = config.max_concurrent_kernels;
                        for n in 0..num_kernels {
                            let idx = (n + *last_issued_kernel + 1) % max_concurrent;
                            match running_kernels[idx] {
                                // &&!kernel.kernel_TB_latency)
                                Some(ref kernel) if !kernel.no_more_blocks_to_run() => {
                                    *last_issued_kernel = idx;
                                    let launch_id = kernel.id();
                                    assert!(!executed_kernels.contains_key(&launch_id));
                                    executed_kernels.insert(launch_id, kernel.name().to_string());
                                    return Some(kernel.clone());
                                }
                                _ => {}
                            }
                        }
                        None
                    })();
                };

                if let Some(kernel) = current_kernel {
                    log::debug!(
                        "core {}-{}: selected kernel {} more blocks={} can issue={}",
                        cluster_id,
                        core_id,
                        kernel,
                        !kernel.no_more_blocks_to_run(),
                        core.can_issue_block(&kernel),
                    );

                    let can_issue =
                        !kernel.no_more_blocks_to_run() && core.can_issue_block(&kernel);
                    // drop(core);
                    if can_issue {
                        // let mut core = cores[cluster_id][core_id].write();
                        core.issue_block(&kernel, cycle);
                        num_blocks_issued += 1;
                        issued.push((cluster_id, core_id));
                        // *want_more_blocks = false;
                        // *should_select_new_kernel = false;
                        *block_issue_next_core = core_id;
                        break;
                    }
                } else {
                    log::debug!("core {}-{}: selected kernel NULL", cluster_id, core.core_id);
                }
            }

            log::trace!(
                "cluster[{}] issued {} blocks",
                cluster_id,
                num_blocks_issued
            );

            if num_blocks_issued > 0 {
                *last_cluster_issue = cluster_id;
                // self.total_blocks_launched += num_blocks_issued;
            }
        }

        drop(need_issue);

        for (cluster_id, core_id) in issued {
            let mut need_issue = need_issue_lock.write();
            need_issue[cluster_id][core_id] = (false, false);
        }
    }

    if false {
        log::debug!("===> issue block to core");
        let mut last_cluster_issue = last_cluster_issue.try_lock();
        let last_issued = *last_cluster_issue;
        let num_clusters = config.num_simt_clusters;
        for cluster_id in 0..num_clusters {
            // debug_assert_eq!(
            //     cluster_idx,
            //     self.clusters[cluster_idx].try_read().cluster_id
            // );
            let cluster_id = (cluster_id + last_issued + 1) % num_clusters;
            let cluster = clusters[cluster_id].read();
            // dbg!((idx, cluster.num_active_sms()));
            // let num_blocks_issued = cluster.issue_block_to_core(self, cycle);
            // let num_cores = cluster.cores.len();
            let num_cores = cores[cluster_id].len();

            log::debug!(
                "cluster {}: issue block to core for {} cores",
                cluster_id,
                num_cores
            );
            let mut num_blocks_issued = 0;

            let mut block_issue_next_core = cluster.block_issue_next_core.try_lock();

            for core_id in 0..num_cores {
                let core_id = (core_id + *block_issue_next_core + 1) % num_cores;
                let core = cores[cluster_id][core_id].read();

                let mut current_kernel = core.current_kernel.try_lock().clone();
                let should_select_new_kernel = if let Some(ref current) = current_kernel {
                    // if no more blocks left, get new kernel once current block completes
                    current.no_more_blocks_to_run() && core.not_completed() == 0
                } else {
                    // core was not assigned a kernel yet
                    true
                };

                if should_select_new_kernel {
                    current_kernel = (|| {
                        let mut last_issued_kernel = last_issued_kernel.lock();
                        let mut executed_kernels = executed_kernels.try_lock();
                        let running_kernels = running_kernels.try_read();

                        // issue same kernel again
                        match running_kernels[*last_issued_kernel] {
                            // && !kernel.kernel_TB_latency)
                            Some(ref last_kernel) if !last_kernel.no_more_blocks_to_run() => {
                                let launch_id = last_kernel.id();
                                executed_kernels
                                    .entry(launch_id)
                                    .or_insert(last_kernel.name().to_string());
                                return Some(last_kernel.clone());
                            }
                            _ => {}
                        };

                        // issue new kernel
                        let num_kernels = running_kernels.len();
                        let max_concurrent = config.max_concurrent_kernels;
                        for n in 0..num_kernels {
                            let idx = (n + *last_issued_kernel + 1) % max_concurrent;
                            match running_kernels[idx] {
                                // &&!kernel.kernel_TB_latency)
                                Some(ref kernel) if !kernel.no_more_blocks_to_run() => {
                                    *last_issued_kernel = idx;
                                    let launch_id = kernel.id();
                                    assert!(!executed_kernels.contains_key(&launch_id));
                                    executed_kernels.insert(launch_id, kernel.name().to_string());
                                    return Some(kernel.clone());
                                }
                                _ => {}
                            }
                        }
                        None
                    })();
                };

                if let Some(kernel) = current_kernel {
                    log::debug!(
                        "core {}-{}: selected kernel {} more blocks={} can issue={}",
                        cluster_id,
                        core_id,
                        kernel,
                        !kernel.no_more_blocks_to_run(),
                        core.can_issue_block(&kernel),
                    );

                    let can_issue =
                        !kernel.no_more_blocks_to_run() && core.can_issue_block(&kernel);
                    drop(core);
                    if can_issue {
                        let mut core = cores[cluster_id][core_id].write();
                        // println!("PARALLEL issue to {:?}", core.id());
                        core.issue_block(&kernel, cycle);
                        num_blocks_issued += 1;
                        *block_issue_next_core = core_id;
                        break;
                    }
                } else {
                    log::debug!("core {}-{}: selected kernel NULL", cluster_id, core.core_id);
                }
            }

            log::trace!(
                "cluster[{}] issued {} blocks",
                cluster_id,
                num_blocks_issued
            );

            if num_blocks_issued > 0 {
                *last_cluster_issue = cluster_id;
                // self.total_blocks_launched += num_blocks_issued;
            }
        }
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

        let interleave_serial = std::env::var("INTERLEAVE")
            .unwrap_or_default()
            .to_lowercase()
            == "yes";

        let num_threads = super::get_num_threads()?;
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global();

        println!("nondeterministic [{run_ahead} run ahead] using RAYON");
        println!(
            "\t => launching {num_threads} worker threads for {} cores",
            self.config.total_cores()
        );
        println!("\t => interleave serial={interleave_serial}");
        println!("");

        let sim_orders: Vec<Arc<_>> = self
            .clusters
            .iter()
            .map(|cluster| Arc::clone(&cluster.try_read().core_sim_order))
            .collect();
        let mem_ports: Vec<Vec<Arc<_>>> = self
            .clusters
            .iter()
            .map(|cluster| {
                cluster
                    .try_read()
                    .cores
                    .iter()
                    .map(|core| Arc::clone(&core.try_read().mem_port))
                    .collect()
            })
            .collect();
        let cores: Vec<Vec<Arc<_>>> = self
            .clusters
            .iter()
            .map(|cluster| cluster.try_read().cores.clone())
            .collect();

        // let last_cycle: Vec<Arc<Mutex<Vec<(u64, u64)>>>> = self
        //     .clusters
        //     .iter()
        //     .map(|cluster| {
        //         Arc::new(Mutex::new(
        //             cluster.try_read().cores.iter().map(|_| (0, 0)).collect(),
        //         ))
        //     })
        //     .collect();

        let last_cycle: Arc<Mutex<Vec<Vec<u64>>>> = Arc::new(Mutex::new(
            self.clusters
                .iter()
                .map(|cluster| {
                    let num_cores = cluster.try_read().cores.len();
                    vec![0; num_cores]
                })
                .collect(),
        ));

        let num_clusters = self.clusters.len();
        let num_cores_per_cluster = self.clusters[0].try_read().cores.len();
        let shape = (run_ahead, num_clusters, num_cores_per_cluster);
        let progress = Array3::<Option<bool>>::from_elem(shape, None);

        let progress = Arc::new(Mutex::new(progress));
        let clusters = Arc::new(self.clusters.clone());
        let cores = Arc::new(cores);
        let sim_orders = Arc::new(sim_orders);
        let mem_ports = Arc::new(mem_ports);
        let mem_sub_partitions = Arc::new(self.mem_sub_partitions.clone());
        let mem_partition_units = Arc::new(self.mem_partition_units.clone());
        // let states: Arc<Mutex<Vec<(u64, crate::DebugState)>>> = Arc::new(Mutex::new(Vec::new()));

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

        let mut active_clusters = vec![false; num_clusters];

        while (self.commands_left() || self.kernels_left()) && !self.reached_limit(cycle) {
            self.process_commands(cycle);
            self.launch_kernels(cycle);

            let mut finished_kernel = None;
            loop {
                log::info!("======== cycle {cycle} ========");
                // println!("======== cycle {cycle} ========");

                if self.reached_limit(cycle) || !self.active() {
                    break;
                }

                let span = tracing::span!(tracing::Level::INFO, "wave", cycle, run_ahead);
                let enter = span.enter();
                if false && interleave_serial {
                    // let kernels_completed = self
                    //     .running_kernels
                    //     .try_read()
                    //     .iter()
                    //     .filter_map(std::option::Option::as_ref)
                    //     .all(|k| k.no_more_blocks_to_run());

                    for (cluster_id, _cluster) in clusters.iter().enumerate() {
                        active_clusters[cluster_id] = true;
                        // active_clusters[cluster_id] =
                        //     !(kernels_completed && cluster.read().not_completed() == 0);
                    }
                    // let active_clusters: Vec<_> = clusters
                    //     .iter()
                    //     .map(|cluster| !(kernels_completed && cluster.read().not_completed() == 0))
                    //     .collect();

                    rayon::scope_fifo(|wave| {
                        for i in 0..run_ahead {
                            for (cluster_id, _cluster_arc) in clusters.iter().cloned().enumerate() {
                                if !active_clusters[cluster_id] {
                                    continue;
                                }
                                // let cores = Arc::clone(&cores);
                                // wave.spawn_fifo(move |_| {
                                //     for (core_id, core) in
                                //         cores[cluster_id].iter().cloned().enumerate()
                                //     {
                                //         let mut core = core.write();
                                //         crate::timeit!(
                                //             "parallel core cycle",
                                //             core.cycle(cycle + i as u64)
                                //         );
                                //     }
                                // });

                                for (_core_id, core) in
                                    cores[cluster_id].iter().cloned().enumerate()
                                {
                                    wave.spawn_fifo(move |_| {
                                        let mut core = core.write();
                                        crate::timeit!(
                                            "parallel core cycle",
                                            core.cycle(cycle + i as u64)
                                        );
                                    });
                                }
                            }

                            // crate::timeit!(
                            //     "prepare serial",
                            //     interleaved_serial_cycle(
                            //         cycle,
                            //         // i,
                            //         &active_clusters,
                            //         // &progress,
                            //         &cores,
                            //         &sim_orders,
                            //         &mem_ports,
                            //         &self.interconn,
                            //         &clusters,
                            //         &self.config,
                            //     )
                            // );
                            //
                            // crate::timeit!("SERIAL PART", self.serial_cycle(cycle));
                            // crate::timeit!("issue blocks", self.issue_block_to_core(cycle));

                            // let last_cycle = Arc::clone(&last_cycle);
                            // let progress = Arc::clone(&progress);
                            //
                            let sim_orders = Arc::clone(&sim_orders);
                            let mem_ports = Arc::clone(&mem_ports);
                            let cores = Arc::clone(&cores);
                            // let states = Arc::clone(&states);

                            let interconn = Arc::clone(&self.interconn);
                            let clusters = Arc::clone(&clusters);

                            let stats = Arc::clone(&self.stats);
                            let mem_sub_partitions = Arc::clone(&mem_sub_partitions);
                            let mem_partition_units = Arc::clone(&mem_partition_units);
                            let config = Arc::clone(&self.config);

                            let running_kernels = Arc::clone(&self.running_kernels);
                            let executed_kernels = Arc::clone(&self.executed_kernels);

                            let last_cluster_issue = Arc::clone(&self.last_cluster_issue);
                            let last_issued_kernel = Arc::clone(&last_issued_kernel);
                            let block_issue_next_core = Arc::clone(&block_issue_next_core);
                            let need_issue = Arc::clone(&need_issue);
                            let issue_guard = Arc::clone(&issue_guard);
                            let serial_lock = Arc::clone(&serial_lock);
                            let active_clusters = active_clusters.clone();

                            wave.spawn_fifo(move |_| {
                                let _guard = serial_lock.lock();
                                interleaved_serial_cycle(
                                    cycle + i as u64,
                                    // i,
                                    &active_clusters,
                                    // &progress,
                                    &cores,
                                    &sim_orders,
                                    &mem_ports,
                                    &interconn,
                                    &clusters,
                                    &config,
                                );
                                crate::timeit!(
                                    "SERIAL PART",
                                    new_serial_cycle(
                                        cycle,
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
                            });
                        }
                    });

                    // crate::timeit!(
                    //     "prepare serial",
                    //     interleaved_serial_cycle(
                    //         cycle + run_ahead as u64,
                    //         // i,
                    //         &active_clusters,
                    //         // &progress,
                    //         &cores,
                    //         &sim_orders,
                    //         &mem_ports,
                    //         &self.interconn,
                    //         &clusters,
                    //         &self.config,
                    //     )
                    // );
                    //
                    // crate::timeit!("SERIAL PART", self.serial_cycle(cycle + run_ahead as u64));
                    crate::timeit!(
                        "issue blocks",
                        self.issue_block_to_core(cycle + run_ahead as u64)
                    );
                }

                if interleave_serial {
                    // rayon::spawn(|| {
                    //     // do the serial cycles here
                    //     let guard = serial_lock.lock();
                    //     // println!(
                    //     //     "PARALLEL: serial cycle for {} ({})",
                    //     //     i,
                    //     //     cycle + i as u64
                    //     // );
                    //
                    //     // wait for signal
                    //     let active_clusters: Vec<_> = {
                    //         let progress = progress.lock();
                    //         clusters
                    //             .iter()
                    //             .enumerate()
                    //             .map(|(cluster_id, _)| {
                    //                 progress
                    //                     .slice(s![i, cluster_id, ..])
                    //                     .iter()
                    //                     .any(|&c| c == Some(true))
                    //             })
                    //             .collect()
                    //     };
                    //
                    //     interleaved_serial_cycle(
                    //         cycle + i as u64,
                    //         // i,
                    //         &active_clusters,
                    //         // &progress,
                    //         &cores,
                    //         &sim_orders,
                    //         &mem_ports,
                    //         &interconn,
                    //         &clusters,
                    //         &config,
                    //     );
                    //     crate::timeit!(
                    //         "SERIAL PART",
                    //         new_serial_cycle(
                    //             cycle + i as u64,
                    //             &stats,
                    //             &need_issue,
                    //             &last_issued_kernel,
                    //             &block_issue_next_core,
                    //             &running_kernels,
                    //             &executed_kernels,
                    //             &mem_sub_partitions,
                    //             &mem_partition_units,
                    //             &interconn,
                    //             &clusters,
                    //             &cores,
                    //             &last_cluster_issue,
                    //             &config,
                    //         )
                    //     );
                    // });

                    // println!("START OF WAVE");
                    rayon::scope_fifo(|wave| {
                        for i in 0..run_ahead {
                            // let last_cycle = last_cycle.clone();
                            // let last_cycle = last_cycle.clone();
                            // let progress = progress.clone();
                            //
                            // // small optimizations
                            // let sim_orders = sim_orders.clone();
                            // let mem_ports = mem_ports.clone();
                            // let cores = cores.clone();
                            //
                            // let interconn = self.interconn.clone();
                            // let clusters = Arc::clone(&clusters);
                            //
                            // let stats = Arc::clone(&self.stats);
                            // let mem_sub_partitions = Arc::clone(&mem_sub_partitions);
                            // let mem_partition_units = Arc::clone(&mem_partition_units);
                            // let config = self.config.clone();

                            for (cluster_id, _cluster_arc) in clusters.iter().cloned().enumerate() {
                                for (core_id, core) in cores[cluster_id].iter().cloned().enumerate()
                                {
                                    // let last_cycle = Arc::clone(&last_cycle);
                                    let progress = Arc::clone(&progress);

                                    let sim_orders = Arc::clone(&sim_orders);
                                    let mem_ports = Arc::clone(&mem_ports);
                                    let cores = Arc::clone(&cores);
                                    // let states = Arc::clone(&states);

                                    let interconn = Arc::clone(&self.interconn);
                                    let clusters = Arc::clone(&clusters);

                                    let stats = Arc::clone(&self.stats);
                                    let mem_sub_partitions = Arc::clone(&mem_sub_partitions);
                                    let mem_partition_units = Arc::clone(&mem_partition_units);
                                    let config = Arc::clone(&self.config);

                                    let running_kernels = Arc::clone(&self.running_kernels);
                                    let executed_kernels = Arc::clone(&self.executed_kernels);

                                    let last_cluster_issue = Arc::clone(&self.last_cluster_issue);
                                    let last_issued_kernel = Arc::clone(&last_issued_kernel);
                                    let block_issue_next_core = Arc::clone(&block_issue_next_core);
                                    let need_issue = Arc::clone(&need_issue);
                                    let issue_guard = Arc::clone(&issue_guard);
                                    let serial_lock = Arc::clone(&serial_lock);

                                    wave.spawn_fifo(move |_| {
                                        // println!(
                                        //     "core ({:>2}-{:>2}) cycle {:>2} ({:>4})",
                                        //     cluster_id,
                                        //     core_id,
                                        //     i,
                                        //     cycle + i as u64
                                        // );

                                        // this can be used for enforcing serial core execution
                                        // for debugging
                                        // let mut progress = progress.lock();

                                        let mut core = core.write();

                                        let kernels_completed = running_kernels
                                            .try_read()
                                            .iter()
                                            .filter_map(std::option::Option::as_ref)
                                            .all(|k| k.no_more_blocks_to_run());

                                        let core_active =
                                            !(kernels_completed && core.not_completed() == 0);

                                        if core_active {
                                            core.cycle(cycle + i as u64);
                                            core.last_cycle = core.last_cycle.max(cycle + i as u64);
                                            core.last_active_cycle =
                                                core.last_active_cycle.max(cycle + i as u64);
                                        } else {
                                            core.last_cycle = core.last_cycle.max(cycle + i as u64);
                                        }

                                        // let mut current_kernel =
                                        //     core.current_kernel.try_lock().clone();
                                        // let want_more_blocks = current_kernel
                                        //     .as_ref()
                                        //     .is_some_and(|k| core.can_issue_block(&*k));
                                        // let should_select_new_kernel =
                                        //     if let Some(ref current) = current_kernel {
                                        //         // if no more blocks left, get new kernel once current block completes
                                        //         current.no_more_blocks_to_run()
                                        //             && core.not_completed() == 0
                                        //     } else {
                                        //         // core was not assigned a kernel yet
                                        //         true
                                        //     };
                                        //
                                        // drop(core);
                                        //
                                        // if want_more_blocks || should_select_new_kernel {
                                        //     let mut need_issue = need_issue.write();
                                        //     need_issue[cluster_id][core_id] =
                                        //         (want_more_blocks, should_select_new_kernel)
                                        // }

                                        // let issue_lock =
                                        //     if want_more_blocks || should_select_new_kernel {
                                        //         issue_guard.0.try_lock()
                                        //     } else {
                                        //         None
                                        //     };
                                        //
                                        // if issue_lock.is_some() {
                                        //     if should_select_new_kernel {
                                        //         current_kernel = (|| {
                                        //             let mut last_issued_kernel =
                                        //                 last_issued_kernel.lock();
                                        //             let mut executed_kernels =
                                        //                 executed_kernels.try_lock();
                                        //             let running_kernels =
                                        //                 running_kernels.try_read();
                                        //
                                        //             // issue same kernel again
                                        //             match running_kernels[*last_issued_kernel] {
                                        //                 // && !kernel.kernel_TB_latency)
                                        //                 Some(ref last_kernel)
                                        //                     if !last_kernel
                                        //                         .no_more_blocks_to_run() =>
                                        //                 {
                                        //                     let launch_id = last_kernel.id();
                                        //                     executed_kernels
                                        //                         .entry(launch_id)
                                        //                         .or_insert(
                                        //                             last_kernel.name().to_string(),
                                        //                         );
                                        //                     return Some(last_kernel.clone());
                                        //                 }
                                        //                 _ => {}
                                        //             };
                                        //
                                        //             // issue new kernel
                                        //             let num_kernels = running_kernels.len();
                                        //             let max_concurrent =
                                        //                 config.max_concurrent_kernels;
                                        //             for n in 0..num_kernels {
                                        //                 let idx = (n + *last_issued_kernel + 1)
                                        //                     % max_concurrent;
                                        //                 match running_kernels[idx] {
                                        //                     // &&!kernel.kernel_TB_latency)
                                        //                     Some(ref kernel)
                                        //                         if !kernel
                                        //                             .no_more_blocks_to_run() =>
                                        //                     {
                                        //                         // *last_issued_kernel = idx;
                                        //                         let launch_id = kernel.id();
                                        //                         assert!(!executed_kernels
                                        //                             .contains_key(&launch_id));
                                        //                         executed_kernels.insert(
                                        //                             launch_id,
                                        //                             kernel.name().to_string(),
                                        //                         );
                                        //                         return Some(kernel.clone());
                                        //                     }
                                        //                     _ => {}
                                        //                 }
                                        //             }
                                        //             None
                                        //         })(
                                        //         );
                                        //     };
                                        //
                                        //     if let Some(kernel) = current_kernel {
                                        //         let can_issue = !kernel.no_more_blocks_to_run()
                                        //             && core.can_issue_block(&kernel);
                                        //         if can_issue {
                                        //             core.issue_block(&kernel, cycle + i as u64);
                                        //         }
                                        //     } else {
                                        //         log::debug!(
                                        //             "core {}-{}: selected kernel NULL",
                                        //             cluster_id,
                                        //             core.core_id
                                        //         );
                                        //     }
                                        // }
                                        //
                                        // drop(issue_lock);
                                        // drop(core);

                                        // check if core is last to finish cycle + i
                                        drop(core);

                                        // for serial core execution for debugging
                                        // progress[[i, cluster_id, core_id]] = Some(core_active);
                                        // let last_to_finish = progress
                                        //     .slice(s![i, .., ..])
                                        //     .iter()
                                        //     .all(|&c| c.is_some());

                                        let last_to_finish = {
                                            let mut progress = progress.lock();
                                            progress[[i, cluster_id, core_id]] = Some(core_active);

                                            let res = progress
                                                .slice(s![i, .., ..])
                                                .iter()
                                                .all(|&c| c.is_some());
                                            // drop(progress);
                                            res
                                        };

                                        if last_to_finish {
                                            // println!(
                                            //     "core {:?} is last to finish cycle {}",
                                            //     (cluster_id, core_id),
                                            //     cycle + i as u64
                                            // );
                                            // println!(
                                            //     "serial cycle {:>2} ({:>4})",
                                            //     i,
                                            //     cycle + i as u64
                                            // );
                                            let guard = serial_lock.lock();
                                            // println!(
                                            //     "PARALLEL: serial cycle for {} ({})",
                                            //     i,
                                            //     cycle + i as u64
                                            // );

                                            let ready_serial_i = {
                                                let mut progress = progress.lock();
                                                // run serial for the first completed sub-cycle
                                                // let mut serial_i = 0;
                                                // while !progress
                                                //     .slice(s![i, .., ..])
                                                //     .iter()
                                                //     .all(|&c| c.is_some())
                                                // {
                                                //     serial_i += 1;
                                                // }

                                                // for ri in 0..run_ahead {
                                                //     dbg!(
                                                //         ri,
                                                //         &progress
                                                //             .slice(s![ri, .., ..])
                                                //             .iter()
                                                //             .all(|&c| c.is_some())
                                                //     );
                                                // }
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

                                                        // let active_clusters = progress
                                                        //     .slice(s![si, cluster_id, ..])
                                                        //     .iter()
                                                        //     .any(|&c| c == Some(true));
                                                        (si, active_clusters)
                                                    })
                                                    .collect();

                                                for (ri, _) in &ready {
                                                    progress.slice_mut(s![*ri, .., ..]).fill(None);
                                                }
                                                // drop(progress);
                                                ready
                                            };

                                            // dbg!(&ready_serial_i);

                                            // for i in 0..run_ahead {
                                            //     let ready = progress
                                            //         .slice(s![i, .., ..])
                                            //         .iter()
                                            //         .all(|&c| c.is_some());
                                            //     //
                                            // }

                                            // let active_clusters: Vec<_> = {
                                            //     let progress = progress.lock();
                                            //     clusters
                                            //         .iter()
                                            //         .enumerate()
                                            //         .map(|(cluster_id, _)| {
                                            //             progress
                                            //                 .slice(s![i, cluster_id, ..])
                                            //                 .iter()
                                            //                 .any(|&c| c == Some(true))
                                            //         })
                                            //         .collect()
                                            // };

                                            for (i, active_clusters) in ready_serial_i {
                                                interleaved_serial_cycle(
                                                    cycle + i as u64,
                                                    // i,
                                                    &active_clusters,
                                                    // &progress,
                                                    &cores,
                                                    &sim_orders,
                                                    &mem_ports,
                                                    &interconn,
                                                    &clusters,
                                                    &config,
                                                );
                                                crate::timeit!(
                                                    "SERIAL PART",
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
                                            }

                                            drop(guard);
                                            if false {
                                                // let state = crate::DebugState {
                                                //     core_orders_per_cluster: sim_orders
                                                //         .iter()
                                                //         .map(|order| order.lock().clone())
                                                //         .collect(),
                                                //     last_cluster_issue: *last_cluster_issue.lock(),
                                                //     last_issued_kernel: *last_issued_kernel.lock(),
                                                //     block_issue_next_core_per_cluster: clusters
                                                //         .iter()
                                                //         .map(|cluster| {
                                                //             *cluster
                                                //                 .read()
                                                //                 .block_issue_next_core
                                                //                 .lock()
                                                //         })
                                                //         .collect(),
                                                // };
                                                // states.lock().push((cycle + i as u64, state));
                                            }
                                        }
                                    });
                                }
                            }

                            // wave.spawn_fifo(move |_| {
                            //     // println!(
                            //     //     "core {:?} is last to finish cycle {}",
                            //     //     (cluster_id, core_id),
                            //     //     cycle + i as u64
                            //     // );
                            //     // println!(
                            //     //     "serial cycle {:>2} ({:>4})",
                            //     //     i,
                            //     //     cycle + i as u64
                            //     // );
                            //     interleaved_serial_cycle(
                            //         cycle + i as u64,
                            //         i,
                            //         // core_sim_order_before,
                            //         &progress,
                            //         &cores,
                            //         &sim_orders,
                            //         &mem_ports,
                            //         &interconn,
                            //         &clusters,
                            //         &config,
                            //     );
                            //     new_serial_cycle(
                            //         cycle + i as u64,
                            //         &stats,
                            //         &mem_sub_partitions,
                            //         &mem_partition_units,
                            //         &interconn,
                            //         &clusters,
                            //         &config,
                            //     );
                            // });
                        }
                    });
                    // println!("END OF WAVE");
                    // all run_ahead cycles completed
                    progress.lock().fill(None);
                    crate::timeit!("issue blocks", self.issue_block_to_core(cycle));
                }

                if !interleave_serial {
                    for i in 0..run_ahead {
                        // run cores in any order
                        // rayon::scope(|core_scope| {
                        //     for (cluster, core, core_id) in cores.iter() {
                        //         core_scope.spawn(move |_| {
                        //             // if *core_id == 0 {
                        //             //     cluster.write().interconn_cycle(cycle + i as
                        //             u64);
                        //             // }
                        //
                        //             core.write().cycle(cycle);
                        //         });
                        //     }
                        // });

                        rayon::scope(|core_scope| {
                            for cluster_arc in &self.clusters {
                                let cluster = cluster_arc.try_read();
                                let kernels_completed = self
                                    .running_kernels
                                    .try_read()
                                    .iter()
                                    .filter_map(std::option::Option::as_ref)
                                    .all(|k| k.no_more_blocks_to_run());

                                let cores_completed = cluster.not_completed() == 0;
                                active_clusters[cluster.cluster_id] =
                                    !(cores_completed && kernels_completed);

                                if cores_completed && kernels_completed {
                                    continue;
                                }
                                for core in cluster.cores.iter().cloned() {
                                    core_scope.spawn(move |_| {
                                        let mut core = core.write();
                                        core.cycle(cycle + i as u64);
                                        // core.last_cycle = core.last_cycle.max(cycle + i as u64);
                                        // core.last_active_cycle =
                                        //     core.last_active_cycle.max(cycle + i as u64);
                                    });
                                }
                            }
                        });

                        for (cluster_id, cluster) in self.clusters.iter().enumerate() {
                            let cluster = cluster.try_read();
                            assert_eq!(cluster.cluster_id, cluster_id);

                            let mut core_sim_order = cluster.core_sim_order.try_lock();
                            for core_id in &*core_sim_order {
                                let core = cluster.cores[*core_id].try_read();
                                // was_updated |= core.last_active_cycle >= (cycle + i as u64);

                                let mut port = core.mem_port.lock();
                                if !active_clusters[cluster_id] {
                                    assert!(port.buffer.is_empty());
                                }
                                for ic::Packet {
                                    data: (dest, fetch, size),
                                    time,
                                } in port.buffer.drain(..)
                                {
                                    self.interconn.push(
                                        cluster_id,
                                        dest,
                                        ic::Packet { data: fetch, time },
                                        size,
                                    );
                                }
                            }

                            if active_clusters[cluster_id] {
                                if use_round_robin {
                                    core_sim_order.rotate_left(1);
                                }
                            } else {
                                // println!(
                                //     "cluster {} not updated in cycle {}",
                                //     cluster.cluster_id,
                                //     cycle + i as u64
                                // );
                            }
                        }

                        // after cores complete, run serial cycle
                        self.serial_cycle(cycle + i as u64);
                    }
                    crate::timeit!("issue blocks", self.issue_block_to_core(cycle));
                }

                cycle += run_ahead as u64;
                self.set_cycle(cycle);

                drop(enter);

                if !self.active() {
                    finished_kernel = self.finished_kernel();
                    if finished_kernel.is_some() {
                        break;
                    }
                }
            }

            if let Some(kernel) = finished_kernel {
                self.cleanup_finished_kernel(&kernel);
            }

            log::trace!(
                "commands left={} kernels left={}",
                self.commands_left(),
                self.kernels_left()
            );
        }

        // self.states = Arc::into_inner(states).unwrap().into_inner();
        log::info!("exit after {cycle} cycles");
        dbg!(&cycle);

        Ok(())
    }

    // #[tracing::instrument]
    // pub fn run_to_completion_parallel_nondeterministic_old(
    //     &mut self,
    //     mut run_ahead: usize,
    // ) -> eyre::Result<()> {
    //     run_ahead = run_ahead.max(1);
    //
    //     let num_threads: usize = std::env::var("NUM_THREADS")
    //         .ok()
    //         .as_deref()
    //         .map(str::parse)
    //         .transpose()?
    //         .unwrap_or_else(num_cpus::get_physical);
    //
    //     let cores_per_thread = self.clusters.len() as f64 / num_threads as f64;
    //     // prefer less cores
    //     let cores_per_thread = cores_per_thread.ceil() as usize;
    //     // todo: tune this
    //     let core_chunks: Vec<Vec<(_, usize, Vec<_>)>> = self
    //         .clusters
    //         .chunks(cores_per_thread)
    //         .map(|clusters| {
    //             clusters
    //                 .iter()
    //                 .map(|cluster| {
    //                     let cluster = cluster.try_read();
    //                     (
    //                         Arc::clone(&cluster.core_sim_order),
    //                         cluster.cluster_id,
    //                         cluster
    //                             .cores
    //                             .iter()
    //                             .map(|core| (core.clone(), core.try_read().mem_port.clone()))
    //                             .collect(),
    //                     )
    //                 })
    //                 .collect()
    //         })
    //         .collect();
    //     let num_chunks = core_chunks.len();
    //
    //     println!("non deterministic [{run_ahead} run ahead]");
    //     println!("\t => launching {num_chunks} threads with {cores_per_thread} cores per thread");
    //
    //     let core_reached: Vec<_> = vec![crossbeam::channel::bounded(1); run_ahead];
    //
    //     let start_core: Vec<_> = vec![crossbeam::channel::bounded(1); num_chunks];
    //
    //     let core_done: Vec<_> = vec![crossbeam::channel::bounded(1); num_chunks];
    //
    //     let lockstep = true;
    //
    //     let use_round_robin =
    //         self.config.simt_core_sim_order == config::SchedulingOrder::RoundRobin;
    //
    //     // spawn worker threads for core cycles
    //     let core_worker_handles: Vec<_> = core_chunks
    //         .into_iter()
    //         .enumerate()
    //         .map(|(cluster_idx, clusters)| {
    //             let (_, start_core_rx) = start_core[cluster_idx].clone();
    //             let (core_done_tx, _) = core_done[cluster_idx].clone();
    //             let core_reached_tx: Vec<_> =
    //                 core_reached.iter().map(|(tx, _)| tx).cloned().collect();
    //             // let running_kernels = self.running_kernels.clone();
    //             let interconn = Arc::clone(&self.interconn);
    //
    //             std::thread::spawn(move || loop {
    //                 let Ok(cycle) = start_core_rx.recv() else {
    //                     // println!("cluster {} exited", cluster.try_read().cluster_id);
    //                     break;
    //                 };
    //
    //                 for i in 0..run_ahead {
    //                     // let kernels_completed = running_kernels
    //                     //     .try_read()
    //                     //     .unwrap()
    //                     //     .iter()
    //                     //     .filter_map(std::option::Option::as_ref)
    //                     //     .all(|k| k.no_more_blocks_to_run());
    //                     tracing::info!("cycle {cycle} + run ahead {i}");
    //
    //                     for (core_sim_order, cluster_id, cores) in &clusters {
    //                         // let mut cluster = cluster.read();
    //                         // let cores_completed = cluster.not_completed() == 0;
    //                         // let cluster_done = cores_completed && kernels_completed;
    //                         // let start = Instant::now();
    //                         let cluster_done = false;
    //                         if !cluster_done {
    //                             for (core, _) in cores {
    //                                 let mut core = core.write();
    //                                 // println!("start core {:?} ({} clusters)", core.id(), num_cores);
    //                                 core.cycle(cycle);
    //                                 // crate::timeit!("parallel::core", core.cycle(cycle));
    //                                 // println!("done core {:?} ({} clusters)", core.id(), num_cores);
    //                             }
    //                         }
    //
    //                         let mut core_sim_order = core_sim_order.try_lock();
    //                         for core_id in core_sim_order.iter() {
    //                             let (_core, mem_port) = &cores[*core_id];
    //                             let mut port = mem_port.try_lock();
    //                             for ic::Packet {
    //                                 data: (dest, fetch, size),
    //                                 time,
    //                             } in port.buffer.drain(..)
    //                             {
    //                                 interconn.push(
    //                                     *cluster_id,
    //                                     dest,
    //                                     ic::Packet { data: fetch, time },
    //                                     size,
    //                                 );
    //                             }
    //                         }
    //
    //                         if use_round_robin {
    //                             core_sim_order.rotate_left(1);
    //                         }
    //
    //                         // #[cfg(feature = "timings")]
    //                         // {
    //                         //     crate::TIMINGS
    //                         //         .lock()
    //                         //         .entry("parallel::cluster")
    //                         //         .or_default()
    //                         //         .add(start.elapsed());
    //                         // }
    //
    //                         // issue new blocks
    //                         // issue_block_to_core
    //                     }
    //
    //                     // collect the core packets pushed to the interconn
    //                     // for cluster in &clusters {
    //                     //     let mut cluster = cluster.write();
    //                     // }
    //
    //                     if lockstep {
    //                         core_reached_tx[i].send(()).unwrap();
    //                     }
    //                     // std::thread::yield_now();
    //                 }
    //
    //                 core_done_tx.send(()).unwrap();
    //             })
    //         })
    //         .collect();
    //
    //     assert_eq!(core_worker_handles.len(), num_chunks);
    //
    //     let mut cycle: u64 = 0;
    //     while (self.commands_left() || self.kernels_left()) && !self.reached_limit(cycle) {
    //         self.process_commands(cycle);
    //         self.launch_kernels(cycle);
    //
    //         let mut finished_kernel = None;
    //         loop {
    //             if self.reached_limit(cycle) || !self.active() {
    //                 break;
    //             }
    //
    //             let span = tracing::span!(tracing::Level::INFO, "wave", cycle, run_ahead);
    //             let enter = span.enter();
    //
    //             // start all cores
    //             tracing::warn!("WAVE START");
    //             for core_idx in 0..num_chunks {
    //                 start_core[core_idx].0.send(cycle).unwrap();
    //             }
    //
    //             for i in 0..run_ahead {
    //                 tracing::info!("cycle {cycle} + run ahead {i}");
    //                 if lockstep {
    //                     // wait until all cores are ready for this
    //                     // println!("waiting for cores to reach barrier {i}");
    //                     for _ in 0..num_chunks {
    //                         core_reached[i].1.recv().unwrap();
    //                     }
    //                     // println!("all cores reached reached barrier {i}");
    //                 }
    //
    //                 log::info!("======== cycle {cycle} ========");
    //                 // log::info!("");
    //
    //                 // could enforce round robin here
    //
    //                 self.serial_cycle(cycle + i as u64);
    //                 // crate::timeit!("SERIAL CYCLE", self.serial_cycle(cycle + i as u64));
    //
    //                 // issue new blocks
    //                 // let start = Instant::now();
    //                 // self.issue_block_to_core();
    //                 // #[cfg(feature = "timings")]
    //                 // {
    //                 //     crate::TIMINGS
    //                 //         .lock()
    //                 //         .entry("serial::issue_block_to_core")
    //                 //         .or_default()
    //                 //         .add(start.elapsed());
    //                 // }
    //             }
    //
    //             // wait for all cores to finish
    //             for core_idx in 0..num_chunks {
    //                 core_done[core_idx].1.recv().unwrap();
    //             }
    //
    //             // locks are uncontended now
    //             self.issue_block_to_core(cycle);
    //             // crate::timeit!("SERIAL ISSUE", self.issue_block_to_core(cycle));
    //             drop(enter);
    //
    //             cycle += run_ahead as u64;
    //             self.set_cycle(cycle);
    //
    //             // dbg!(self.active());
    //
    //             if !self.active() {
    //                 finished_kernel = self.finished_kernel();
    //                 if finished_kernel.is_some() {
    //                     break;
    //                 }
    //             }
    //         }
    //
    //         if let Some(kernel) = finished_kernel {
    //             self.cleanup_finished_kernel(&kernel);
    //         }
    //
    //         log::trace!(
    //             "commands left={} kernels left={}",
    //             self.commands_left(),
    //             self.kernels_left()
    //         );
    //     }
    //
    //     log::info!("exit after {cycle} cycles");
    //     dbg!(&cycle);
    //     Ok(())
    // }

    #[tracing::instrument]
    pub fn serial_cycle(&mut self, cycle: u64) {
        for cluster in &self.clusters {
            cluster.write().interconn_cycle(cycle);
        }

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
                    let cluster_id = fetch.cluster_id;
                    fetch.set_status(mem_fetch::Status::IN_ICNT_TO_SHADER, 0);
                    // fetch.set_return_timestamp(gpu_sim_cycle + gpu_tot_sim_cycle);
                    // , gpu_sim_cycle + gpu_tot_sim_cycle);
                    // drop(fetch);
                    self.interconn.push(
                        device,
                        cluster_id,
                        ic::Packet {
                            data: fetch,
                            time: cycle,
                        },
                        response_packet_size,
                    );
                    // self.partition_replies_in_parallel += 1;
                } else {
                    // self.gpu_stall_icnt2sh += 1;
                }
            }
        }

        for (_i, unit) in self.mem_partition_units.iter().enumerate() {
            unit.try_write().simple_dram_cycle(cycle);
        }

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

            // same as full with parameter overload
            if mem_sub
                .interconn_to_l2_queue
                .can_fit(mem_sub_partition::SECTOR_CHUNCK_SIZE as usize)
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
                    // mem_sub.push(packet.data, packet.time);
                    // self.parallel_mem_partition_reqs += 1;
                }
            } else {
                log::debug!("SKIP sub partition {} ({}): DRAM full stall", i, device);
                self.stats.lock().stall_dram_full += 1;
            }
            // we borrow all of sub here, which is a problem for the cyclic reference in l2
            // interface
            mem_sub.cache_cycle(cycle);
        }

        // let mut all_threads_complete = true;
        // if self.config.flush_l1_cache {
        //     for cluster in &self.clusters {
        //         let mut cluster = cluster.try_write();
        //         if cluster.not_completed() == 0 {
        //             cluster.cache_invalidate();
        //         } else {
        //             all_threads_complete = false;
        //         }
        //     }
        // }
        //
        // if self.config.flush_l2_cache {
        //     if !self.config.flush_l1_cache {
        //         for cluster in &self.clusters {
        //             let cluster = cluster.try_read();
        //             if cluster.not_completed() > 0 {
        //                 all_threads_complete = false;
        //                 break;
        //             }
        //         }
        //     }
        //
        //     if let Some(l2_config) = &self.config.data_cache_l2 {
        //         if all_threads_complete {
        //             log::debug!("flushed L2 caches...");
        //             if l2_config.inner.total_lines() > 0 {
        //                 for (i, mem_sub) in self.mem_sub_partitions.iter().enumerate() {
        //                     // let mut mem_sub = mem_sub.try_borrow_mut().unwrap();
        //                     let mut mem_sub = mem_sub.try_lock();
        //                     let num_dirty_lines_flushed = mem_sub.flush_l2();
        //                     log::debug!(
        //                         "dirty lines flushed from L2 {} is {:?}",
        //                         i,
        //                         num_dirty_lines_flushed
        //                     );
        //                 }
        //             }
        //         }
        //     }
        // }
    }
}
