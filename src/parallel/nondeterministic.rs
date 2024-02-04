#![allow(warnings, clippy::all)]

use crate::kernel_manager::SelectKernel;
use crate::sync::{Arc, Mutex, RwLock};
use crate::{config, core, ic, kernel::Kernel, mem_fetch, mem_sub_partition, Simulator};
use color_eyre::eyre;
use ndarray::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

// #[tracing::instrument]
// #[inline]
// fn interleaved_serial_cycle<I, C, MC>(
//     cycle: u64,
//     active_clusters: &Vec<bool>,
//     cores: &Arc<Vec<Vec<Arc<RwLock<crate::core::Core<I, MC>>>>>>,
//     sim_orders: &Arc<Vec<Arc<Mutex<VecDeque<usize>>>>>,
//     mem_ports: &Arc<Vec<Vec<Arc<Mutex<crate::core::CoreMemoryConnection<C>>>>>>,
//     interconn: &Arc<I>,
//     clusters: &Vec<Arc<crate::Cluster<I, MC>>>,
//     config: &config::GPU,
// ) where
//     C: ic::BufferedConnection<ic::Packet<(usize, mem_fetch::MemFetch, u32)>>,
//     I: ic::Interconnect<ic::Packet<mem_fetch::MemFetch>>,
// {
//     let use_round_robin = config.simt_core_sim_order == config::SchedulingOrder::RoundRobin;
//
//     for (cluster_id, _cluster) in clusters.iter().enumerate() {
//         let cluster_active = active_clusters[cluster_id];
//         let mut core_sim_order = sim_orders[cluster_id].try_lock();
//         for core_id in &*core_sim_order {
//             let mut port = mem_ports[cluster_id][*core_id].lock();
//             if cluster_active {
//                 if !port.buffer.is_empty() {}
//                 // assert!(port.buffer.is_empty());
//             }
//
//             for ic::Packet {
//                 fetch: (dest, fetch, size),
//                 time,
//             } in port.buffer.drain()
//             {
//                 interconn.push(cluster_id, dest, ic::Packet { fetch, time }, size);
//             }
//         }
//
//         if cluster_active {
//             if use_round_robin {
//                 core_sim_order.rotate_left(1);
//             }
//         } else {
//             // println!(
//             //     "SERIAL: cluster {} not updated in cycle {}",
//             //     cluster.cluster_id,
//             //     cycle + i as u64
//             // );
//         }
//     }
// }

// fn newer_serial_cycle<I, MC>(
fn newer_serial_cycle(
    cycle: u64,
    // stats: &Arc<Mutex<stats::PerKernel>>,
    // need_issue_lock: &Arc<RwLock<Vec<Vec<(bool, bool)>>>>,
    // last_issued_kernel: &Arc<Mutex<usize>>,
    // block_issue_next_core: &Arc<Vec<Mutex<usize>>>,
    // running_kernels: &Arc<RwLock<Vec<Option<(usize, Arc<dyn Kernel>)>>>>,
    // executed_kernels: &Arc<Mutex<HashMap<u64, Arc<dyn Kernel>>>>,
    // mem_sub_partitions: &Arc<Vec<Arc<Mutex<crate::mem_sub_partition::MemorySubPartition<MC>>>>>,
    // mem_partition_units: &Arc<Vec<Arc<RwLock<crate::mem_partition_unit::MemoryPartitionUnit<MC>>>>>,
    // interconn: &Arc<I>,
    // clusters: &Arc<Vec<Arc<crate::Cluster<I, MC>>>>,
    // cores: &Arc<Vec<Vec<Arc<RwLock<crate::Core<I, MC>>>>>>,
    // last_cluster_issue: &Arc<Mutex<usize>>,
    // config: &config::GPU,
)
// where
//     I: ic::Interconnect<ic::Packet<mem_fetch::MemFetch>>,
//     MC: crate::mcu::MemoryController,
{
    todo!()
}

// #[tracing::instrument]
// // #[inline]
// fn new_serial_cycle<I, MC>(
//     cycle: u64,
//     // stats: &Arc<Mutex<stats::PerKernel>>,
//     need_issue_lock: &Arc<RwLock<Vec<Vec<(bool, bool)>>>>,
//     last_issued_kernel: &Arc<Mutex<usize>>,
//     block_issue_next_core: &Arc<Vec<Mutex<usize>>>,
//     running_kernels: &Arc<RwLock<Vec<Option<(usize, Arc<dyn Kernel>)>>>>,
//     executed_kernels: &Arc<Mutex<HashMap<u64, Arc<dyn Kernel>>>>,
//     mem_sub_partitions: &Arc<Vec<Arc<Mutex<crate::mem_sub_partition::MemorySubPartition<MC>>>>>,
//     mem_partition_units: &Arc<Vec<Arc<RwLock<crate::mem_partition_unit::MemoryPartitionUnit<MC>>>>>,
//     interconn: &Arc<I>,
//     clusters: &Arc<Vec<Arc<crate::Cluster<I, MC>>>>,
//     cores: &Arc<Vec<Vec<Arc<RwLock<crate::Core<I, MC>>>>>>,
//     last_cluster_issue: &Arc<Mutex<usize>>,
//     config: &config::GPU,
// ) where
//     I: ic::Interconnect<ic::Packet<mem_fetch::MemFetch>>,
//     MC: crate::mcu::MemoryController,
// {
//     // it could happen that two serial cycles overlap when using spawn fifo, so we need
//
//     todo!();
//     // for cluster in clusters.iter_mut() {
//     //     crate::timeit!("serial::interconn", cluster.interconn_cycle(cycle));
//     // }
//
//     for (i, mem_sub) in mem_sub_partitions.iter().enumerate() {
//         let mut mem_sub = mem_sub.try_lock();
//         if let Some(fetch) = mem_sub.top() {
//             let response_packet_size = if fetch.is_write() {
//                 fetch.control_size()
//             } else {
//                 fetch.size()
//             };
//             let device = config.mem_id_to_device_id(i);
//             if interconn.has_buffer(device, response_packet_size) {
//                 let mut fetch = mem_sub.pop().unwrap();
//                 if let Some(cluster_id) = fetch.cluster_id {
//                     fetch.set_status(mem_fetch::Status::IN_ICNT_TO_SHADER, 0);
//                     interconn.push(
//                         device,
//                         cluster_id,
//                         ic::Packet { fetch, time: cycle },
//                         response_packet_size,
//                     );
//                 }
//             }
//         }
//     }
//
//     for (_i, unit) in mem_partition_units.iter().enumerate() {
//         crate::timeit!("serial::dram", unit.try_write().simple_dram_cycle(cycle));
//     }
//
//     for (i, mem_sub) in mem_sub_partitions.iter().enumerate() {
//         let mut mem_sub = mem_sub.try_lock();
//         // move memory request from interconnect into memory partition
//         // (if not backed up)
//         //
//         // Note:This needs to be called in DRAM clock domain if there
//         // is no L2 cache in the system In the worst case, we may need
//         // to push SECTOR_CHUNCK_SIZE requests, so ensure you have enough
//         // buffer for them
//         let device = config.mem_id_to_device_id(i);
//
//         // same as full with parameter overload
//         if mem_sub
//             .interconn_to_l2_queue
//             .can_fit(mem_sub_partition::NUM_SECTORS as usize)
//         {
//             if let Some(ic::Packet { fetch, .. }) = interconn.pop(device) {
//                 log::debug!(
//                     "got new fetch {} for mem sub partition {} ({})",
//                     fetch,
//                     i,
//                     device
//                 );
//
//                 // changed from packet.time to cycle here
//                 // mem_sub.push(packet.data, packet.time);
//                 mem_sub.push(fetch, cycle);
//             }
//         } else {
//             log::debug!("SKIP sub partition {} ({}): DRAM full stall", i, device);
//             // TODO
//             // if let Some(kernel) = &*mem_sub.current_kernel.lock() {
//             //     let mut stats = stats.lock();
//             //     let kernel_stats = stats.get_mut(kernel.id() as usize);
//             //     kernel_stats.stall_dram_full += 1;
//             // }
//         }
//         // we borrow all of sub here, which is a problem for the cyclic reference in l2
//         // interface
//         crate::timeit!("serial::subpartitions", mem_sub.cycle(cycle));
//     }
// }

use crate::{cluster::Cluster, mem_partition_unit::MemoryPartitionUnit};

struct SerialCycle<'a, I, MC> {
    clusters: &'a mut [Cluster<I, MC>],
    mem_partition_units: &'a mut [MemoryPartitionUnit<MC>],
    interconn: &'a I,
    config: &'a config::GPU,
}

// impl SerialCycle {
impl<'a, I, MC> SerialCycle<'a, I, MC>
where
    I: ic::Interconnect<ic::Packet<mem_fetch::MemFetch>>,
    MC: crate::mcu::MemoryController,
{
    pub fn cycle(&mut self, cycle: u64) {
        for cluster in self.clusters.iter_mut() {
            // Receive memory responses addressed to each cluster and forward to cores
            crate::timeit!("serial::interconn", cluster.interconn_cycle(cycle));
        }

        for partition in self.mem_partition_units.iter_mut() {
            for mem_sub in partition.sub_partitions.iter_mut() {
                // let mut mem_sub = mem_sub.try_lock();
                if let Some(fetch) = mem_sub.top() {
                    let response_packet_size = if fetch.is_write() {
                        fetch.control_size()
                    } else {
                        fetch.size()
                    };
                    let device = self.config.mem_id_to_device_id(mem_sub.global_id);
                    if self.interconn.has_buffer(device, response_packet_size) {
                        let mut fetch = mem_sub.pop().unwrap();
                        if let Some(cluster_id) = fetch.cluster_id {
                            fetch.set_status(mem_fetch::Status::IN_ICNT_TO_SHADER, 0);
                            self.interconn.push(
                                device,
                                cluster_id,
                                ic::Packet { fetch, time: cycle },
                                response_packet_size,
                            );
                        }
                    }
                }
            }
        }

        // dram cycle
        for (_i, unit) in self.mem_partition_units.iter_mut().enumerate() {
            crate::timeit!("serial::dram", unit.simple_dram_cycle(cycle));
            // crate::timeit!("serial::dram", unit.try_write().simple_dram_cycle(cycle));
        }

        for partition in self.mem_partition_units.iter_mut() {
            for mem_sub in partition.sub_partitions.iter_mut() {
                // let mut mem_sub = mem_sub.try_lock();
                // move memory request from interconnect into memory partition
                // (if not backed up)
                //
                // Note:This needs to be called in DRAM clock domain if there
                // is no L2 cache in the system In the worst case, we may need
                // to push SECTOR_CHUNCK_SIZE requests, so ensure you have enough
                // buffer for them
                let device = self.config.mem_id_to_device_id(mem_sub.global_id);

                if mem_sub
                    .interconn_to_l2_queue
                    .can_fit(mem_sub_partition::NUM_SECTORS as usize)
                {
                    if let Some(ic::Packet { fetch, .. }) = self.interconn.pop(device) {
                        assert_eq!(fetch.sub_partition_id(), mem_sub.global_id);
                        log::debug!(
                            "got new fetch {} for mem sub partition {} ({})",
                            fetch,
                            mem_sub.global_id,
                            device
                        );

                        // assert_eq!(cycle, packet.time);
                        // TODO: changed form packet.time to cycle
                        mem_sub.push(fetch, cycle);
                    }
                } else {
                    log::debug!(
                        "SKIP sub partition {} ({}): DRAM full stall",
                        mem_sub.global_id,
                        device
                    );
                    // let kernel_id = self
                    //     .kernel_manager
                    //     // .lock()
                    //     .current_kernel()
                    //     // .lock()
                    //     .as_ref()
                    //     .map(|kernel| kernel.id() as usize);
                    // let mut stats = self.stats.lock();
                    // let kernel_stats = self.stats.get_mut(kernel_id);
                    // let kernel_stats = self.stats.get_mut(kernel_id);
                    // kernel_stats.stall_dram_full += 1;
                }
                // we borrow all of sub here, which is a problem for the cyclic reference in l2
                // interface
                crate::timeit!("serial::subpartitions", mem_sub.cycle(cycle));
            }
        }
    }
}

impl<I, MC> Simulator<I, MC>
where
    I: ic::Interconnect<ic::Packet<mem_fetch::MemFetch>>,
    MC: crate::mcu::MemoryController,
{
    #[tracing::instrument]
    pub fn run_to_completion_parallel_nondeterministic(
        &mut self,
        mut run_ahead: usize,
    ) -> eyre::Result<()> {
        #[cfg(feature = "timings")]
        crate::TIMINGS.lock().clear();

        let num_threads = super::get_num_threads()?
            .or(self.config.simulation_threads)
            .unwrap_or_else(num_cpus::get_physical);

        let serial = std::env::var("SERIAL").unwrap_or_default().to_lowercase() == "yes";

        super::rayon_pool(num_threads)?.install(|| {
            println!("nondeterministic interleaved [{run_ahead} run ahead] using RAYON");
            println!(
                "\t => launching {num_threads} worker threads for {} cores",
                self.config.total_cores()
            );
            // println!("\t => interleave serial={interleave_serial}");
            println!("");

            if serial {
                self.run_to_completion_parallel_nondeterministic_serial(run_ahead)?;
            } else {
                self.run_to_completion_parallel_nondeterministic_fifo(run_ahead)?;
            }
            Ok::<_, eyre::Report>(())
        })
    }

    #[tracing::instrument]
    pub fn test_serial(&mut self, cycle: u64) {
        for cluster in self.clusters.iter_mut() {
            // Receive memory responses addressed to each cluster and forward to cores

            // perf: blocks on the lock of CORE_INSTR_FETCH_RESPONSE_QUEUE
            // perf: blocks on the lock of CORE_LOAD_STORE_RESPONSE_QUEUE
            // perf: pops from INTERCONN (mem->cluster)
            crate::timeit!("serial::interconn", cluster.interconn_cycle(cycle));
        }

        for partition in self.mem_partition_units.iter_mut() {
            for mem_sub in partition.sub_partitions.iter_mut() {
                if let Some(fetch) = mem_sub.top() {
                    let response_packet_size = if fetch.is_write() {
                        fetch.control_size()
                    } else {
                        fetch.size()
                    };
                    let device = self.config.mem_id_to_device_id(mem_sub.global_id);
                    if self.interconn.has_buffer(device, response_packet_size) {
                        let mut fetch = mem_sub.pop().unwrap();
                        if let Some(cluster_id) = fetch.cluster_id {
                            fetch.set_status(mem_fetch::Status::IN_ICNT_TO_SHADER, 0);
                            // perf: pushes to INTERCONN (mem->cluster)
                            self.interconn.push(
                                device,
                                cluster_id,
                                ic::Packet { fetch, time: cycle },
                                response_packet_size,
                            );
                        }
                    }
                }
            }
        }

        // dram cycle
        for (_i, unit) in self.mem_partition_units.iter_mut().enumerate() {
            // perf: does not lock at all
            crate::timeit!("serial::dram", unit.simple_dram_cycle(cycle));
        }

        for partition in self.mem_partition_units.iter_mut() {
            for mem_sub in partition.sub_partitions.iter_mut() {
                // move memory request from interconnect into memory partition
                // (if not backed up)
                //
                // Note:This needs to be called in DRAM clock domain if there
                // is no L2 cache in the system In the worst case, we may need
                // to push SECTOR_CHUNCK_SIZE requests, so ensure you have enough
                // buffer for them
                let device = self.config.mem_id_to_device_id(mem_sub.global_id);

                if mem_sub
                    .interconn_to_l2_queue
                    .can_fit(mem_sub_partition::NUM_SECTORS as usize)
                {
                    // perf: pops from INTERCONN (cluster->mem)
                    if let Some(ic::Packet { fetch, .. }) = self.interconn.pop(device) {
                        assert_eq!(fetch.sub_partition_id(), mem_sub.global_id);
                        // assert_eq!(cycle, packet.time);
                        log::debug!(
                            "got new fetch {} for mem sub partition {} ({})",
                            fetch,
                            mem_sub.global_id,
                            device
                        );
                        // TODO: changed form packet.time to cycle
                        mem_sub.push(fetch, cycle);
                    }
                } else {
                    log::debug!(
                        "SKIP sub partition {} ({}): DRAM full stall",
                        mem_sub.global_id,
                        device
                    );
                }
                crate::timeit!("serial::subpartitions", mem_sub.cycle(cycle));
            }
        }
    }

    #[tracing::instrument]
    pub fn run_to_completion_parallel_nondeterministic_serial(
        &mut self,
        run_ahead: usize,
    ) -> eyre::Result<()> {
        let run_ahead = run_ahead.max(1) as u64;
        let mut active_clusters = utils::box_slice![false; self.clusters.len()];

        let mut cycle: u64 = 0;
        let log_every = 10_000;
        let mut last_time = std::time::Instant::now();

        while (self.trace.commands_left() || self.kernels_left()) && !self.reached_limit(cycle) {
            cycle = self.process_commands(cycle);
            self.launch_kernels(cycle);

            let mut finished_kernel = None;
            loop {
                log::info!("======== cycle {cycle} ========");
                log::info!("");
                if cycle % log_every == 0 && cycle > 0 {
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

                // let kernels_completed = self.kernel_manager.all_kernels_completed();

                for i in 0..run_ahead {
                    for cluster in self.clusters.iter_mut() {
                        // Receive memory responses addressed to each cluster and forward to cores
                        crate::timeit!("serial::interconn", cluster.interconn_cycle(cycle + i));
                    }
                }

                for i in 0..run_ahead {
                    crate::timeit!("serial::total", self.test_serial(cycle + i));
                }

                for cluster in self.clusters.iter_mut() {
                    let kernel_manager = &self.kernel_manager;

                    let cluster_id = &cluster.cluster_id;

                    for core in cluster.cores.iter_mut() {
                        let mut core = core.try_write();
                        for i in 0..run_ahead {
                            crate::timeit!("core::cycle", core.cycle(cycle + i));

                            // do not enforce ordering of interconnect requests and round robin
                            // core simualation ordering
                            // let port = &mut core.mem_port;
                            for ic::Packet {
                                fetch: (dest, fetch, size),
                                time,
                            } in core.mem_port.buffer.drain(..)
                            {
                                assert_eq!(time, cycle + i);
                                // let mut accesses = crate::core::debug::ACCESSES.lock();
                                // let mut num_accesses = accesses
                                //     .entry((
                                //         fetch.allocation_id().unwrap_or(100),
                                //         fetch.access_kind().into(),
                                //     ))
                                //     .or_insert(0);
                                // *num_accesses += 1;

                                self.interconn.push(
                                    *cluster_id,
                                    dest,
                                    ic::Packet { fetch, time },
                                    size,
                                );
                            }

                            assert!(core.mem_port.buffer.is_empty());

                            core.maybe_issue_block(kernel_manager, cycle + i);
                        }
                    }
                }

                cycle += run_ahead;

                if !self.active() {
                    finished_kernel = self.kernel_manager.get_finished_kernel();
                    if finished_kernel.is_some() {
                        break;
                    }
                }
            }

            // self.debug_non_exit();

            if let Some(kernel) = finished_kernel {
                self.cleanup_finished_kernel(&*kernel, cycle);
            }

            log::trace!(
                "commands left={} kernels left={}",
                self.trace.commands_left(),
                self.kernels_left()
            );
        }

        self.debug_completed_blocks();

        self.stats.no_kernel.sim.cycles = cycle;
        log::info!("exit after {cycle} cycles");

        Ok(())
    }

    #[tracing::instrument]
    pub fn run_to_completion_parallel_nondeterministic_fifo(
        &mut self,
        run_ahead: usize,
    ) -> eyre::Result<()> {
        let run_ahead = run_ahead.max(1) as u64;

        let cores: Box<[Arc<_>]> = self
            .clusters
            .iter()
            .flat_map(|cluster| cluster.cores.iter())
            .map(Arc::clone)
            .collect();

        let mut cycle: u64 = 0;
        let log_every = 10_000;
        let mut last_time = std::time::Instant::now();

        while (self.trace.commands_left() || self.kernels_left()) && !self.reached_limit(cycle) {
            cycle = self.process_commands(cycle);
            self.launch_kernels(cycle);

            let mut finished_kernel = None;
            loop {
                log::info!("======== cycle {cycle} ========");
                log::info!("");
                if cycle % log_every == 0 && cycle > 0 {
                    eprintln!(
                        "cycle {cycle:<10} ({:>8.4} cycle/sec)",
                        log_every as f64 / last_time.elapsed().as_secs_f64()
                    );
                    last_time = std::time::Instant::now()
                }

                if self.reached_limit(cycle) {
                    break;
                }

                let serial_cycle = SerialCycle {
                    clusters: &mut self.clusters,
                    mem_partition_units: &mut self.mem_partition_units,
                    interconn: &self.interconn,
                    config: &self.config,
                };
                // let serial_cycle = Arc::new(Mutex::new(serial_cycle));
                let serial_cycle = Mutex::new(serial_cycle);

                let fanout = 1;

                rayon::scope_fifo(|wave| {
                    // rayon::scope(|wave| {
                    let serial_cycle_ref = &serial_cycle;
                    let kernel_manager = &self.kernel_manager;
                    let interconn = &self.interconn;

                    for i in 0..run_ahead {
                        // let serial_cycle = serial_cycle.clone();
                        wave.spawn_fifo(move |_| {
                            // wave.spawn(move |_| {
                            // serial cycle
                            for i in 0..fanout {
                                crate::timeit!(
                                    "serial::total",
                                    serial_cycle_ref.lock().cycle(cycle + i)
                                );
                            }
                        });

                        // idea: could also make core use mutex honestly, maybe
                        // thats better for perf

                        for core in cores.iter() {
                            wave.spawn_fifo(move |_| {
                                // wave.spawn(move |_| {
                                // core cycle
                                let mut core = core.write();
                                let cluster_id = core.cluster_id;

                                // if core.current_kernel.is_none() {
                                //     return;
                                // }

                                for i in 0..fanout {
                                    if core.current_kernel.is_some() {
                                        crate::timeit!("core::cycle", core.cycle(cycle + i));
                                    }

                                    // do not enforce ordering of interconnect requests and round robin
                                    // core simualation ordering
                                    // let port = &mut core.mem_port;
                                    {
                                        for ic::Packet {
                                            fetch: (dest, fetch, size),
                                            time,
                                        } in core.mem_port.buffer.drain(..)
                                        {
                                            assert_eq!(time, cycle + i);
                                            interconn.push(
                                                cluster_id,
                                                dest,
                                                ic::Packet { fetch, time },
                                                size,
                                            );
                                        }
                                    }

                                    crate::timeit!(
                                        "issue_block::maybe",
                                        core.maybe_issue_block(kernel_manager, cycle + i)
                                    );
                                }
                            });
                        }
                    }
                });

                // end of parallel section

                cycle += run_ahead * fanout;

                // for i in 0..run_ahead * fanout {
                //     crate::timeit!(
                //         "issue_to_core",
                //         self.issue_block_to_core_deterministic(cycle)
                //     );
                // }

                // self.issue_block_to_core(cycle);
                // self.flush_caches(cycle);

                if !self.active() {
                    finished_kernel = self.kernel_manager.get_finished_kernel();
                    if finished_kernel.is_some() {
                        break;
                    }
                }
            }

            // self.debug_non_exit();

            if let Some(kernel) = finished_kernel {
                self.cleanup_finished_kernel(&*kernel, cycle);
            }

            log::trace!(
                "commands left={} kernels left={}",
                self.trace.commands_left(),
                self.kernels_left()
            );
        }

        self.debug_completed_blocks();

        self.stats.no_kernel.sim.cycles = cycle;
        log::info!("exit after {cycle} cycles");

        Ok(())
    }

    #[tracing::instrument]
    pub fn run_to_completion_parallel_nondeterministic_naive(
        &mut self,
        run_ahead: usize,
    ) -> eyre::Result<()> {
        let run_ahead = run_ahead.max(1) as u64;

        // TODO: fix run ahead for now..
        // let run_ahead = 1;

        let cores: Box<[Arc<_>]> = self
            .clusters
            .iter()
            .flat_map(|cluster| cluster.cores.iter())
            .map(Arc::clone)
            .collect();

        let mut active_clusters = utils::box_slice![false; self.clusters.len()];

        let mut cycle: u64 = 0;
        let log_every = 10_000;
        let mut last_time = std::time::Instant::now();

        while (self.trace.commands_left() || self.kernels_left()) && !self.reached_limit(cycle) {
            cycle = self.process_commands(cycle);
            self.launch_kernels(cycle);

            let mut finished_kernel = None;
            loop {
                log::info!("======== cycle {cycle} ========");
                log::info!("");
                if cycle % log_every == 0 && cycle > 0 {
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

                // let kernels_completed = self.kernel_manager.all_kernels_completed();

                // assumption: most likely we are just exiting the simulation too
                // early...
                // probably because we need a busy() method on different things,
                // e.g. right now we cannot really look into the cluster to core
                // queues, which therefore maybe would indicate that the simulation
                // is over already...
                //
                //
                // Idea: to prevent higher L1 hit rate when issueing to same
                // core a few times: use fair mutex for select_kernel?
                // use fair mutex for get block reader of kernel?
                //
                // todo: debug the exit states of all components..
                // todo: debug the number of blocks per in final stats
                // we are on a good path

                // for i in 0..run_ahead {
                //     for cluster in self.clusters.iter_mut() {
                //         // Receive memory responses addressed to each cluster and forward to cores
                //         crate::timeit!("serial::interconn", cluster.interconn_cycle(cycle));
                //     }
                // }

                // rayon::scope_fifo(|wave| {

                rayon::scope(|wave| {
                    // SERIAL cycle
                    // cluster: interconn cycle: could be mutable, but then instead of locking
                    // a full core, keep separate

                    // let kernel_manager = Arc::new(Mutex::new(&mut self.kernel_manager));

                    wave.spawn(|_| {
                        for i in 0..run_ahead {
                            let start = Instant::now();

                            for cluster in self.clusters.iter_mut() {
                                // Receive memory responses addressed to each cluster and forward to cores
                                crate::timeit!("serial::interconn", cluster.interconn_cycle(cycle));
                            }

                            for partition in self.mem_partition_units.iter_mut() {
                                for mem_sub in partition.sub_partitions.iter_mut() {
                                    // let mut mem_sub = mem_sub.try_lock();
                                    if let Some(fetch) = mem_sub.top() {
                                        let response_packet_size = if fetch.is_write() {
                                            fetch.control_size()
                                        } else {
                                            fetch.size()
                                        };
                                        let device =
                                            self.config.mem_id_to_device_id(mem_sub.global_id);
                                        if self.interconn.has_buffer(device, response_packet_size) {
                                            let mut fetch = mem_sub.pop().unwrap();
                                            if let Some(cluster_id) = fetch.cluster_id {
                                                fetch.set_status(
                                                    mem_fetch::Status::IN_ICNT_TO_SHADER,
                                                    0,
                                                );
                                                self.interconn.push(
                                                    device,
                                                    cluster_id,
                                                    ic::Packet {
                                                        fetch,
                                                        time: cycle + i,
                                                    },
                                                    response_packet_size,
                                                );
                                            }
                                        }
                                    }
                                }
                            }

                            // dram cycle
                            for (_i, unit) in self.mem_partition_units.iter_mut().enumerate() {
                                crate::timeit!("serial::dram", unit.simple_dram_cycle(cycle + i));
                                // crate::timeit!("serial::dram", unit.try_write().simple_dram_cycle(cycle));
                            }

                            for partition in self.mem_partition_units.iter_mut() {
                                for mem_sub in partition.sub_partitions.iter_mut() {
                                    // let mut mem_sub = mem_sub.try_lock();
                                    // move memory request from interconnect into memory partition
                                    // (if not backed up)
                                    //
                                    // Note:This needs to be called in DRAM clock domain if there
                                    // is no L2 cache in the system In the worst case, we may need
                                    // to push SECTOR_CHUNCK_SIZE requests, so ensure you have enough
                                    // buffer for them
                                    let device = self.config.mem_id_to_device_id(mem_sub.global_id);

                                    if mem_sub
                                        .interconn_to_l2_queue
                                        .can_fit(mem_sub_partition::NUM_SECTORS as usize)
                                    {
                                        if let Some(ic::Packet { fetch, .. }) =
                                            self.interconn.pop(device)
                                        {
                                            assert_eq!(fetch.sub_partition_id(), mem_sub.global_id);
                                            log::debug!(
                                                "got new fetch {} for mem sub partition {} ({})",
                                                fetch,
                                                mem_sub.global_id,
                                                device
                                            );

                                            // assert_eq!(cycle, packet.time);
                                            // TODO: changed form packet.time to cycle
                                            mem_sub.push(fetch, cycle + i);
                                        }
                                    } else {
                                        log::debug!(
                                            "SKIP sub partition {} ({}): DRAM full stall",
                                            mem_sub.global_id,
                                            device
                                        );
                                        // let kernel_id = self
                                        //     .kernel_manager
                                        //     // .lock()
                                        //     .current_kernel()
                                        //     // .lock()
                                        //     .as_ref()
                                        //     .map(|kernel| kernel.id() as usize);
                                        // let mut stats = self.stats.lock();
                                        // let kernel_stats = self.stats.get_mut(kernel_id);
                                        // let kernel_stats = self.stats.get_mut(kernel_id);
                                        // kernel_stats.stall_dram_full += 1;
                                    }
                                    // we borrow all of sub here, which is a problem for the cyclic reference in l2
                                    // interface
                                    crate::timeit!(
                                        "serial::subpartitions",
                                        mem_sub.cycle(cycle + i)
                                    );
                                }
                            }
                            #[cfg(feature = "timings")]
                            crate::TIMINGS
                                .lock()
                                .entry("serial::total")
                                .or_default()
                                .add(start.elapsed());
                        }
                    });

                    let kernel_manager = &self.kernel_manager;

                    // for cluster in self.clusters.iter_mut() {
                    //     // let cores_completed = cluster.num_active_threads() == 0;
                    //     // let cluster_active = !(cores_completed && kernels_completed);
                    //     // active_clusters[cluster.cluster_id] = cluster_active;
                    //     //
                    //     // if !cluster_active {
                    //     //     continue;
                    //     // }
                    //
                    //
                    //     let cluster_id = &cluster.cluster_id;
                    //
                    //     for core in cluster.cores.iter_mut() {
                    for core in cores.iter() {
                        wave.spawn(|_| {
                            let mut core = core.try_write();
                            let cluster_id = core.cluster_id;
                            for i in 0..run_ahead {
                                crate::timeit!("core::cycle", core.cycle(cycle + i));

                                // do not enforce ordering of interconnect requests and round robin
                                // core simualation ordering
                                let port = &mut core.mem_port;
                                for ic::Packet {
                                    fetch: (dest, fetch, size),
                                    time,
                                } in port.buffer.drain(..)
                                {
                                    assert_eq!(time, cycle + i);
                                    self.interconn.push(
                                        cluster_id,
                                        dest,
                                        ic::Packet { fetch, time },
                                        size,
                                    );
                                }

                                core.maybe_issue_block(kernel_manager, cycle + i);

                                //     // todo: split this into two separate paths
                                //     // check if core needs more blocks
                                //     let mut current_kernel = core.current_kernel.as_ref(); //.map(Arc::clone);
                                //                                                            // core.current_kernel.try_lock().as_ref().map(Arc::clone);
                                //     let should_select_new_kernel =
                                //         if let Some(ref current) = current_kernel {
                                //             // if no more blocks left, get new kernel once current block completes
                                //             current.no_more_blocks_to_run()
                                //                 && core.num_active_threads() == 0
                                //         } else {
                                //             // core was not assigned a kernel yet
                                //             true
                                //         };
                                //
                                //     let mut new_kernel = None;
                                //     if should_select_new_kernel {
                                //         new_kernel = kernel_manager.select_kernel();
                                //     }
                                //     if should_select_new_kernel {
                                //         current_kernel = new_kernel.as_ref();
                                //     }
                                //
                                //     if let Some(kernel) = current_kernel {
                                //         log::debug!(
                                //             "core {:?}: selected kernel {} more blocks={} can issue={}",
                                //             core.id(),
                                //             kernel,
                                //             !kernel.no_more_blocks_to_run(),
                                //             core.can_issue_block(&**kernel),
                                //         );
                                //
                                //         let can_issue = !kernel.no_more_blocks_to_run()
                                //             && core.can_issue_block(&**kernel);
                                //         // drop(core);
                                //         if can_issue {
                                //             // let mut core = self.cores[core_id].write();
                                //             // let core = &mut self.cores[core_id];
                                //             core.issue_block(&*Arc::clone(kernel), cycle + i);
                                //             // num_blocks_issued += 1;
                                //             // self.block_issue_next_core = core_id;
                                //             // break;
                                //         }
                                //     }
                            }
                        });
                    }
                }); // end of parallel section

                // collect the core packets pushed to the interconn
                // if false {
                //     for (cluster_id, active) in active_clusters.iter().enumerate() {
                //         if !active {
                //             continue;
                //         }
                //         let cluster = &mut self.clusters[cluster_id];
                //         let mut core_sim_order = cluster.core_sim_order.try_lock();
                //         for core_id in &*core_sim_order {
                //             // let core = cluster.cores[*core_id].try_read();
                //             let core = &mut cluster.cores[*core_id];
                //             // let mut port = core.mem_port.lock();
                //             let port = &mut core.mem_port;
                //             for ic::Packet {
                //                 fetch: (dest, fetch, size),
                //                 time,
                //             } in port.buffer.drain(..)
                //             {
                //                 self.interconn.push(
                //                     core.cluster_id,
                //                     dest,
                //                     ic::Packet { fetch, time },
                //                     size,
                //                 );
                //             }
                //         }
                //         if let config::SchedulingOrder::RoundRobin = self.config.simt_core_sim_order
                //         {
                //             core_sim_order.rotate_left(1);
                //         }
                //     }
                // }

                // self.issue_block_to_core(cycle);
                // todo:
                // self.flush_caches(cycle);

                cycle += run_ahead;

                if !self.active() {
                    finished_kernel = self.kernel_manager.get_finished_kernel();
                    if finished_kernel.is_some() {
                        break;
                    }
                }
            }

            // self.debug_non_exit();

            if let Some(kernel) = finished_kernel {
                self.cleanup_finished_kernel(&*kernel, cycle);
            }

            log::trace!(
                "commands left={} kernels left={}",
                self.trace.commands_left(),
                self.kernels_left()
            );
        }

        self.debug_completed_blocks();

        self.stats.no_kernel.sim.cycles = cycle;
        log::info!("exit after {cycle} cycles");

        Ok(())
    }

    pub fn debug_completed_blocks(&self) {
        // use itertools::Itertools;
        //
        // let mut accesses = crate::core::debug::ACCESSES.lock();
        // dbg!(&accesses);
        // accesses.clear();
        //
        // let mut completed_blocks = crate::core::debug::COMPLETED_BLOCKS.lock();
        //
        // use std::collections::HashSet;
        // let unique_blocks: HashSet<_> = completed_blocks.iter().map(|block| &block.block).collect();
        // let unique_block_ids: HashSet<_> = completed_blocks
        //     .iter()
        //     .map(|block| block.block.id())
        //     .collect();
        // let unique_cores: HashSet<_> = completed_blocks
        //     .iter()
        //     .map(|block| block.global_core_id)
        //     .collect();
        // let unique_kernel_ids: HashSet<_> = completed_blocks
        //     .iter()
        //     .map(|block| block.kernel_id)
        //     .collect();
        //
        // eprintln!("unique blocks: {}", unique_blocks.len());
        // eprintln!("unique block ids: {}", unique_block_ids.len());
        // eprintln!("unique cores: {}", unique_cores.len());
        // eprintln!("unique kernels: {}", unique_kernel_ids.len());
        //
        // let blocks_per_core: HashMap<usize, HashSet<&trace_model::Point>> = unique_cores
        //     .iter()
        //     .copied()
        //     .map(|core_id| {
        //         let blocks: Vec<_> = completed_blocks
        //             .iter()
        //             .filter_map(|block| {
        //                 if block.global_core_id == core_id {
        //                     Some(&block.block)
        //                 } else {
        //                     None
        //                 }
        //             })
        //             .collect();
        //         assert!(
        //             blocks.iter().all_unique(),
        //             "a single core should never run the same block more than once"
        //         );
        //         (core_id, blocks.into_iter().collect())
        //     })
        //     .collect();
        //
        // for core_id in unique_cores.iter() {
        //     eprintln!(
        //         "core {} num blocks: {}",
        //         core_id,
        //         blocks_per_core[core_id].len(),
        //         // completed_blocks
        //         //     .iter()
        //         //     .filter(|block| block.global_core_id == *core_id)
        //         //     .count()
        //     );
        //     for (other_core_id, other_blocks) in blocks_per_core.iter() {
        //         let blocks = &blocks_per_core[core_id];
        //         if other_core_id != core_id {
        //             let isect: Vec<_> = other_blocks.intersection(blocks).collect();
        //             if !isect.is_empty() {
        //                 dbg!(&core_id, &other_core_id, &blocks, &other_blocks, &isect);
        //             }
        //             assert!(
        //                 isect.is_empty(),
        //                 "a block should never be run by more than once core"
        //             );
        //         }
        //     }
        // }
        //
        // completed_blocks.clear();
    }

    #[tracing::instrument]
    pub fn run_to_completion_parallel_nondeterministic_test(
        &mut self,
        run_ahead: usize,
    ) -> eyre::Result<()> {
        let run_ahead = run_ahead.max(1) as u64;

        let mut active_clusters = utils::box_slice![false; self.clusters.len()];

        let kernels_completed = self.kernel_manager.all_kernels_completed();

        let mut cycle = 100;

        // start the serial cycle
        // crate::timeit!("core::cycle", self.serial_cycle(cycle));
        // for cluster in self.clusters.iter() {
        //     // Receive memory responses addressed to each cluster and forward to cores
        //     wave.spawn(move |_| {
        //         crate::timeit!("serial::interconn", cluster.interconn_cycle(cycle));
        //     });
        // }

        // wave.spawn(move |_| {
        // crate::timeit!(
        //     "core::cycle",
        //     newer_serial_cycle(
        //         cycle,
        //         // &need_issue,
        //         // &last_issued_kernel,
        //         // &block_issue_next_core,
        //         // &running_kernels,
        //         // &executed_kernels,
        //         // &mem_sub_partitions,
        //         // &mem_partition_units,
        //         // &interconn,
        //         // &clusters,
        //         // &cores,
        //         // &last_cluster_issue,
        //         // &config,
        //     )
        // );
        // });

        // check if we can run the cores and the serial cycle in parallel
        rayon::scope(|wave| {
            for cluster in self.clusters.iter_mut() {
                let cores_completed = cluster.num_active_threads() == 0;
                let cluster_active = !(cores_completed && kernels_completed);
                active_clusters[cluster.cluster_id] = cluster_active;

                if !cluster_active {
                    continue;
                }

                for core in cluster.cores.iter_mut() {
                    wave.spawn(move |_| {
                        let mut core = core.try_write();
                        crate::timeit!("core::cycle", core.cycle(cycle));
                    });
                }
            }
        });

        // check if we can run the cores in parallel for multiple cycles
        rayon::scope(|wave| {
            for cluster in self.clusters.iter_mut() {
                let cores_completed = cluster.num_active_threads() == 0;
                let cluster_active = !(cores_completed && kernels_completed);
                active_clusters[cluster.cluster_id] = cluster_active;

                if !cluster_active {
                    continue;
                }

                for core in cluster.cores.iter_mut() {
                    wave.spawn(move |_| {
                        // this works but creates some skew of course
                        let mut core = core.try_write();
                        for i in 0..run_ahead {
                            crate::timeit!("core::cycle", core.cycle(cycle + i));
                        }
                    });
                }
            }
        });

        // as expected, the fine grained fifo approach does not work without
        // mutexes, as there is always a chance two cycles
        // will run at the same time for a single core, hence core cycles
        // must be serialized
        // rayon::scope_fifo(|wave| {
        //     for i in 0..run_ahead {
        //         for cluster in self.clusters.iter_mut() {
        //             let cores_completed = cluster.num_active_threads() == 0;
        //             let cluster_active = !(cores_completed && kernels_completed);
        //             active_clusters[cluster.cluster_id] = cluster_active;
        //
        //             if !cluster_active {
        //                 continue;
        //             }
        //
        //             for core in cluster.cores.iter_mut() {
        //                 wave.spawn_fifo(move |_| {
        //                     crate::timeit!("core::cycle", core.cycle(cycle + i));
        //                 });
        //             }
        //         }
        //     }
        // });

        Ok(())
    }

    #[tracing::instrument]
    pub fn run_to_completion_parallel_nondeterministic_old(
        &mut self,
        mut run_ahead: usize,
    ) -> eyre::Result<()> {
        // todo!("under (re)construction");
        // crate::TIMINGS.lock().clear();
        //
        // run_ahead = run_ahead.max(1);
        //
        // let interleave_serial = true;
        // // interleave_serial |= std::env::var("INTERLEAVE")
        // //     .unwrap_or_default()
        // //     .to_lowercase()
        // //     == "yes";
        //
        // let num_threads = super::get_num_threads()?
        //     .or(self.config.simulation_threads)
        //     .unwrap_or_else(num_cpus::get_physical);
        //
        // super::rayon_pool(num_threads)?.install(|| {
        //     println!("nondeterministic interleaved [{run_ahead} run ahead] using RAYON");
        //     println!(
        //         "\t => launching {num_threads} worker threads for {} cores",
        //         self.config.total_cores()
        //     );
        //     // println!("\t => interleave serial={interleave_serial}");
        //     println!("");
        //
        //     let sim_orders: Vec<Arc<_>> = self
        //         .clusters
        //         .iter()
        //         .map(|cluster| Arc::clone(&cluster.core_sim_order))
        //         .collect();
        //     let mem_ports: Vec<Vec<Arc<_>>> = self
        //         .clusters
        //         .iter()
        //         .map(|cluster| {
        //             cluster
        //                 .cores
        //                 .iter()
        //                 .map(|core| Arc::clone(&core.try_read().mem_port))
        //                 .collect()
        //         })
        //         .collect();
        //     let cores: Vec<Vec<Arc<_>>> = self
        //         .clusters
        //         .iter()
        //         .map(|cluster| cluster.cores.clone())
        //         .collect();
        //
        //     let last_cycle: Arc<Mutex<Vec<Vec<u64>>>> = Arc::new(Mutex::new(
        //         self.clusters
        //             .iter()
        //             .map(|cluster| {
        //                 let num_cores = cluster.cores.len();
        //                 vec![0; num_cores]
        //             })
        //             .collect(),
        //     ));
        //
        //     let num_clusters = self.clusters.len();
        //     let num_cores_per_cluster = self.clusters[0].cores.len();
        //     let shape = (run_ahead, num_clusters, num_cores_per_cluster);
        //     let progress = Array3::<Option<bool>>::from_elem(shape, None);
        //
        //     let progress = Arc::new(Mutex::new(progress));
        //     let clusters = Arc::new(self.clusters.clone());
        //     let cores = Arc::new(cores);
        //     let sim_orders = Arc::new(sim_orders);
        //     let mem_ports = Arc::new(mem_ports);
        //     let mem_sub_partitions = Arc::new(self.mem_sub_partitions.clone());
        //     let mem_partition_units = Arc::new(self.mem_partition_units.clone());
        //
        //     let use_round_robin =
        //         self.config.simt_core_sim_order == config::SchedulingOrder::RoundRobin;
        //
        //     let last_issued_kernel = Arc::new(Mutex::new(0));
        //     let block_issue_next_core = Arc::new(
        //         (0..num_clusters)
        //             .into_iter()
        //             .map(|_| Mutex::new(num_cores_per_cluster - 1))
        //             .collect(),
        //     );
        //     let need_issue: Arc<RwLock<Vec<Vec<_>>>> = Arc::new(RwLock::new(vec![
        //         vec![(false, false); num_cores_per_cluster];
        //         num_clusters
        //     ]));
        //
        //     let serial_lock = Arc::new(Mutex::new(()));
        //     let issue_guard = Arc::new(Mutex::new(()));
        //
        //     let mut cycle: u64 = 0;
        //     let log_every = 10_000;
        //     let mut last_time = std::time::Instant::now();
        //
        //     let mut active_clusters = vec![false; num_clusters];
        //
        //     while (self.commands_left() || self.kernels_left()) && !self.reached_limit(cycle) {
        //         cycle = self.process_commands(cycle);
        //         self.launch_kernels(cycle);
        //
        //         let start_cycle = cycle;
        //
        //         let mut finished_kernel = None;
        //         loop {
        //             log::info!("======== cycle {cycle} ========");
        //             log::info!("");
        //             if (cycle - start_cycle) % log_every == 0 && (cycle - start_cycle) > 0 {
        //                 eprintln!(
        //                     "cycle {cycle:<10} ({:>8.4} cycle/sec)",
        //                     log_every as f64 / last_time.elapsed().as_secs_f64()
        //                 );
        //                 last_time = std::time::Instant::now()
        //             }
        //
        //             // if self.reached_limit(cycle) || !self.active() {
        //             if self.reached_limit(cycle) {
        //                 break;
        //             }
        //
        //             let span = tracing::span!(tracing::Level::INFO, "wave", cycle, run_ahead);
        //             let enter = span.enter();
        //
        //             if interleave_serial {
        //                 rayon::scope_fifo(|wave| {
        //                     for i in 0..run_ahead {
        //                         for (cluster_id, _cluster_arc) in
        //                             clusters.iter().cloned().enumerate()
        //                         {
        //                             for (core_id, core) in
        //                                 cores[cluster_id].iter().cloned().enumerate()
        //                             {
        //                                 let progress = Arc::clone(&progress);
        //
        //                                 let sim_orders = Arc::clone(&sim_orders);
        //                                 let mem_ports = Arc::clone(&mem_ports);
        //                                 let cores = Arc::clone(&cores);
        //
        //                                 let interconn = Arc::clone(&self.interconn);
        //                                 let clusters = Arc::clone(&clusters);
        //
        //                                 // let stats = Arc::clone(&self.stats);
        //                                 let mem_sub_partitions = Arc::clone(&mem_sub_partitions);
        //                                 let mem_partition_units = Arc::clone(&mem_partition_units);
        //                                 let config = Arc::clone(&self.config);
        //
        //                                 let running_kernels = Arc::clone(&self.running_kernels);
        //                                 let executed_kernels = Arc::clone(&self.executed_kernels);
        //
        //                                 let last_cluster_issue =
        //                                     Arc::clone(&self.last_cluster_issue);
        //                                 let last_issued_kernel = Arc::clone(&last_issued_kernel);
        //                                 let block_issue_next_core =
        //                                     Arc::clone(&block_issue_next_core);
        //                                 let need_issue = Arc::clone(&need_issue);
        //                                 let issue_guard = Arc::clone(&issue_guard);
        //                                 let serial_lock = Arc::clone(&serial_lock);
        //
        //                                 wave.spawn_fifo(move |_| {
        //                                     let mut core = core.write();
        //
        //                                     let kernels_completed = running_kernels
        //                                         .try_read()
        //                                         .iter()
        //                                         .filter_map(Option::as_ref)
        //                                         .all(|(_, k)| k.no_more_blocks_to_run());
        //
        //                                     let core_active =
        //                                         !(kernels_completed && core.num_active_threads() == 0);
        //
        //                                     if core_active {
        //                                         crate::timeit!(
        //                                             "core::cycle",
        //                                             core.cycle(cycle + i as u64)
        //                                         );
        //                                     }
        //
        //                                     drop(core);
        //
        //                                     let last_to_finish = {
        //                                         let mut progress = progress.lock();
        //                                         progress[[i, cluster_id, core_id]] =
        //                                             Some(core_active);
        //
        //                                         let res = progress
        //                                             .slice(s![i, .., ..])
        //                                             .iter()
        //                                             .all(|&c| c.is_some());
        //                                         res
        //                                     };
        //
        //                                     if last_to_finish {
        //                                         let guard = serial_lock.lock();
        //                                         let ready_serial_i = {
        //                                             let mut progress = progress.lock();
        //                                             let ready: Vec<_> = (0..run_ahead)
        //                                                 .into_iter()
        //                                                 .skip_while(|&si| {
        //                                                     !progress
        //                                                         .slice(s![si, .., ..])
        //                                                         .iter()
        //                                                         .all(|&c| c.is_some())
        //                                                 })
        //                                                 .take_while(|&si| {
        //                                                     progress
        //                                                         .slice(s![si, .., ..])
        //                                                         .iter()
        //                                                         .all(|&c| c.is_some())
        //                                                 })
        //                                                 .map(|si| {
        //                                                     let active_clusters: Vec<_> = progress
        //                                                         .slice(s![si, .., ..])
        //                                                         .axis_iter(Axis(0))
        //                                                         .map(|cluster_cores| {
        //                                                             cluster_cores
        //                                                                 .iter()
        //                                                                 .any(|&c| c == Some(true))
        //                                                         })
        //                                                         .collect();
        //
        //                                                     assert_eq!(
        //                                                         active_clusters.len(),
        //                                                         num_clusters
        //                                                     );
        //
        //                                                     (si, active_clusters)
        //                                                 })
        //                                                 .collect();
        //
        //                                             for (ri, _) in &ready {
        //                                                 progress
        //                                                     .slice_mut(s![*ri, .., ..])
        //                                                     .fill(None);
        //                                             }
        //                                             ready
        //                                         };
        //
        //                                         for (i, active_clusters) in ready_serial_i {
        //                                             crate::timeit!(
        //                                                 "serial::postcore",
        //                                                 interleaved_serial_cycle(
        //                                                     cycle + i as u64,
        //                                                     &active_clusters,
        //                                                     &cores,
        //                                                     &sim_orders,
        //                                                     &mem_ports,
        //                                                     &interconn,
        //                                                     &clusters,
        //                                                     &config,
        //                                                 )
        //                                             );
        //                                             // if (cycle + i as u64) % 4 == 0 {
        //                                             crate::timeit!(
        //                                                 "serial::cycle",
        //                                                 new_serial_cycle(
        //                                                     cycle + i as u64,
        //                                                     // &stats,
        //                                                     &need_issue,
        //                                                     &last_issued_kernel,
        //                                                     &block_issue_next_core,
        //                                                     &running_kernels,
        //                                                     &executed_kernels,
        //                                                     &mem_sub_partitions,
        //                                                     &mem_partition_units,
        //                                                     &interconn,
        //                                                     &clusters,
        //                                                     &cores,
        //                                                     &last_cluster_issue,
        //                                                     &config,
        //                                                 )
        //                                             );
        //                                             // }
        //                                         }
        //
        //                                         drop(guard);
        //                                     }
        //                                 });
        //                             }
        //                         }
        //                     }
        //                 });
        //                 // all run_ahead cycles completed
        //                 progress.lock().fill(None);
        //                 crate::timeit!("cycle::issue_blocks", self.issue_block_to_core(cycle));
        //             }
        //
        //             if !interleave_serial {
        //                 rayon::scope_fifo(|wave| {
        //                     for i in 0..run_ahead {
        //                         for (cluster_id, _cluster_arc) in
        //                             clusters.iter().cloned().enumerate()
        //                         {
        //                             for (core_id, core) in
        //                                 cores[cluster_id].iter().cloned().enumerate()
        //                             {
        //                                 let running_kernels = Arc::clone(&self.running_kernels);
        //                                 wave.spawn_fifo(move |_| {
        //                                     let mut core = core.write();
        //
        //                                     // let kernels_completed = running_kernels
        //                                     //     .try_read()
        //                                     //     .iter()
        //                                     //     .filter_map(Option::as_ref)
        //                                     //     .all(|(_, k)| k.no_more_blocks_to_run());
        //                                     //
        //                                     // // let core_active = core.not_completed() != 0;
        //                                     // let core_active =
        //                                     //     !(kernels_completed && core.not_completed() == 0);
        //                                     //
        //                                     // if core_active {
        //                                     //     core.cycle(cycle + i as u64);
        //                                     //     core.last_cycle =
        //                                     //         core.last_cycle.max(cycle + i as u64);
        //                                     //     core.last_active_cycle =
        //                                     //         core.last_active_cycle.max(cycle + i as u64);
        //                                     // } else {
        //                                     //     core.last_cycle =
        //                                     //         core.last_cycle.max(cycle + i as u64);
        //                                     // }
        //                                     // if core_active {
        //                                     //     core.cycle(cycle + i as u64);
        //                                     // }
        //                                     crate::timeit!(
        //                                         "core::cycle",
        //                                         core.cycle(cycle + i as u64)
        //                                     );
        //                                 });
        //                             }
        //                         }
        //
        //                         // let progress = Arc::clone(&progress);
        //                         //
        //                         // let sim_orders = Arc::clone(&sim_orders);
        //                         // let mem_ports = Arc::clone(&mem_ports);
        //                         // let cores = Arc::clone(&cores);
        //                         //
        //                         // let interconn = Arc::clone(&self.interconn);
        //                         // let clusters = Arc::clone(&clusters);
        //                         //
        //                         // let stats = Arc::clone(&self.stats);
        //                         // let mem_sub_partitions = Arc::clone(&mem_sub_partitions);
        //                         // let mem_partition_units = Arc::clone(&mem_partition_units);
        //                         // let config = Arc::clone(&self.config);
        //                         //
        //                         // let executed_kernels = Arc::clone(&self.executed_kernels);
        //                         // let running_kernels = Arc::clone(&self.running_kernels);
        //                         //
        //                         // let last_cluster_issue = Arc::clone(&self.last_cluster_issue);
        //                         // let last_issued_kernel = Arc::clone(&last_issued_kernel);
        //                         // let block_issue_next_core = Arc::clone(&block_issue_next_core);
        //                         // let need_issue = Arc::clone(&need_issue);
        //                         // let issue_guard = Arc::clone(&issue_guard);
        //                         // let serial_lock = Arc::clone(&serial_lock);
        //                         //
        //                         // let active_clusters = vec![true; self.config.num_simt_clusters];
        //                         //
        //                         // wave.spawn_fifo(move |_| {
        //                         //     let guard = serial_lock.lock();
        //                         //     crate::timeit!(
        //                         //         "serial::postcore",
        //                         //         interleaved_serial_cycle(
        //                         //             cycle + i as u64,
        //                         //             &active_clusters,
        //                         //             &cores,
        //                         //             &sim_orders,
        //                         //             &mem_ports,
        //                         //             &interconn,
        //                         //             &clusters,
        //                         //             &config,
        //                         //         )
        //                         //     );
        //                         // });
        //                     }
        //                 });
        //
        //                 let progress = Arc::clone(&progress);
        //
        //                 let sim_orders = Arc::clone(&sim_orders);
        //                 let mem_ports = Arc::clone(&mem_ports);
        //                 let cores = Arc::clone(&cores);
        //
        //                 let interconn = Arc::clone(&self.interconn);
        //                 let clusters = Arc::clone(&clusters);
        //
        //                 // let stats = Arc::clone(&self.stats);
        //                 let mem_sub_partitions = Arc::clone(&mem_sub_partitions);
        //                 let mem_partition_units = Arc::clone(&mem_partition_units);
        //                 let config = Arc::clone(&self.config);
        //
        //                 let executed_kernels = Arc::clone(&self.executed_kernels);
        //                 let running_kernels = Arc::clone(&self.running_kernels);
        //
        //                 let last_cluster_issue = Arc::clone(&self.last_cluster_issue);
        //                 let last_issued_kernel = Arc::clone(&last_issued_kernel);
        //                 let block_issue_next_core = Arc::clone(&block_issue_next_core);
        //                 let need_issue = Arc::clone(&need_issue);
        //                 let issue_guard = Arc::clone(&issue_guard);
        //                 let serial_lock = Arc::clone(&serial_lock);
        //
        //                 let active_clusters = vec![true; self.config.num_simt_clusters];
        //                 //
        //                 // wave.spawn_fifo(move |_| {
        //                 //     let guard = serial_lock.lock();
        //                 //     crate::timeit!(
        //                 //         "serial::postcore",
        //                 //         interleaved_serial_cycle(
        //                 //             cycle + i as u64,
        //                 //             &active_clusters,
        //                 //             &cores,
        //                 //             &sim_orders,
        //                 //             &mem_ports,
        //                 //             &interconn,
        //                 //             &clusters,
        //                 //             &config,
        //                 //         )
        //                 //     );
        //                 //     if (cycle + i as u64) % 2 == 0 {
        //                 for i in 0..run_ahead {
        //                     crate::timeit!(
        //                         "serial::postcore",
        //                         interleaved_serial_cycle(
        //                             cycle + i as u64,
        //                             &active_clusters,
        //                             &cores,
        //                             &sim_orders,
        //                             &mem_ports,
        //                             &interconn,
        //                             &clusters,
        //                             &config,
        //                         )
        //                     );
        //
        //                     crate::timeit!(
        //                         "serial::cycle",
        //                         new_serial_cycle(
        //                             cycle + i as u64,
        //                             // &stats,
        //                             &need_issue,
        //                             &last_issued_kernel,
        //                             &block_issue_next_core,
        //                             &running_kernels,
        //                             &executed_kernels,
        //                             &mem_sub_partitions,
        //                             &mem_partition_units,
        //                             &interconn,
        //                             &clusters,
        //                             &cores,
        //                             &last_cluster_issue,
        //                             &config,
        //                         )
        //                     );
        //                     //         }
        //                     //         drop(guard);
        //                     //     });
        //                     // }
        //                     // });
        //                     crate::timeit!("issue blocks", self.issue_block_to_core(cycle));
        //                 }
        //                 //     for i in 0..run_ahead {
        //                 //         // run cores in any order
        //                 //         rayon::scope(|core_scope| {
        //                 //             for cluster_arc in &self.clusters {
        //                 //                 let cluster = cluster_arc.try_read();
        //                 //                 let kernels_completed = self
        //                 //                     .running_kernels
        //                 //                     .try_read()
        //                 //                     .iter()
        //                 //                     .filter_map(std::option::Option::as_ref)
        //                 //                     .all(|(_, k)| k.no_more_blocks_to_run());
        //                 //
        //                 //                 let cores_completed = cluster.not_completed() == 0;
        //                 //                 active_clusters[cluster.cluster_id] =
        //                 //                     !(cores_completed && kernels_completed);
        //                 //
        //                 //                 if cores_completed && kernels_completed {
        //                 //                     continue;
        //                 //                 }
        //                 //                 for core in cluster.cores.iter().cloned() {
        //                 //                     core_scope.spawn(move |_| {
        //                 //                         let mut core = core.write();
        //                 //                         core.cycle(cycle + i as u64);
        //                 //                     });
        //                 //                 }
        //                 //             }
        //                 //         });
        //                 //
        //                 //         // push memory request packets generated by cores to the interconnection network.
        //                 //         for (cluster_id, cluster) in self.clusters.iter().enumerate() {
        //                 //             let cluster = cluster.try_read();
        //                 //             assert_eq!(cluster.cluster_id, cluster_id);
        //                 //
        //                 //             let mut core_sim_order = cluster.core_sim_order.try_lock();
        //                 //             for core_id in &*core_sim_order {
        //                 //                 let core = cluster.cores[*core_id].try_read();
        //                 //                 // was_updated |= core.last_active_cycle >= (cycle + i as u64);
        //                 //
        //                 //                 let mut port = core.mem_port.lock();
        //                 //                 if !active_clusters[cluster_id] {
        //                 //                     assert!(port.buffer.is_empty());
        //                 //                 }
        //                 //                 for ic::Packet {
        //                 //                     data: (dest, fetch, size),
        //                 //                     time,
        //                 //                 } in port.buffer.drain(..)
        //                 //                 {
        //                 //                     self.interconn.push(
        //                 //                         cluster_id,
        //                 //                         dest,
        //                 //                         ic::Packet { data: fetch, time },
        //                 //                         size,
        //                 //                     );
        //                 //                 }
        //                 //             }
        //                 //
        //                 //             if active_clusters[cluster_id] {
        //                 //                 if use_round_robin {
        //                 //                     core_sim_order.rotate_left(1);
        //                 //                 }
        //                 //             } else {
        //                 //                 // println!(
        //                 //                 //     "cluster {} not updated in cycle {}",
        //                 //                 //     cluster.cluster_id,
        //                 //                 //     cycle + i as u64
        //                 //                 // );
        //                 //             }
        //                 //         }
        //                 //
        //                 //         // after cores complete, run serial cycle
        //                 //         self.serial_cycle(cycle + i as u64);
        //                 //     }
        //                 //     crate::timeit!("issue blocks", self.issue_block_to_core(cycle));
        //             }
        //
        //             cycle += run_ahead as u64;
        //
        //             drop(enter);
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
        //             self.cleanup_finished_kernel(&*kernel, cycle);
        //         }
        //
        //         log::trace!(
        //             "commands left={} kernels left={}",
        //             self.commands_left(),
        //             self.kernels_left()
        //         );
        //     }
        //
        //     self.stats.no_kernel.sim.cycles = cycle;
        //     // self.stats.lock().no_kernel.sim.cycles = cycle;
        //
        //     if let Some(log_after_cycle) = self.log_after_cycle {
        //         if log_after_cycle >= cycle {
        //             eprintln!("WARNING: log after {log_after_cycle} cycles but simulation ended after {cycle} cycles");
        //         }
        //     }
        //
        //     log::info!("exit after {cycle} cycles");
        //     dbg!(&cycle);
        //     Ok::<_, eyre::Report>(())
        // })
        Ok(())
    }

    #[tracing::instrument]
    pub fn serial_cycle(&mut self, cycle: u64) {
        for cluster in self.clusters.iter_mut() {
            // Receive memory responses addressed to each cluster and forward to cores
            crate::timeit!("serial::interconn", cluster.interconn_cycle(cycle));
        }

        // let sub_partitions = crate::MemSubPartitionIterMut {
        //     partition_units: &mut self.mem_partition_units,
        //     sub_partitions_per_partition: self.config.num_sub_partitions_per_memory_controller,
        //     global_sub_id: 0,
        // };

        // send memory responses from memory sub partitions to the requestor clusters via
        // interconnect
        // for (i, mem_sub) in self.mem_sub_partitions.iter().enumerate() {
        // for (i, mem_sub) in self.mem_sub_partitions().enumerate() {
        // for mem_sub in sub_partitions {
        for partition in self.mem_partition_units.iter_mut() {
            for mem_sub in partition.sub_partitions.iter_mut() {
                // let mut mem_sub = mem_sub.try_lock();
                if let Some(fetch) = mem_sub.top() {
                    let response_packet_size = if fetch.is_write() {
                        fetch.control_size()
                    } else {
                        fetch.size()
                    };
                    let device = self.config.mem_id_to_device_id(mem_sub.global_id);
                    if self.interconn.has_buffer(device, response_packet_size) {
                        let mut fetch = mem_sub.pop().unwrap();
                        if let Some(cluster_id) = fetch.cluster_id {
                            fetch.set_status(mem_fetch::Status::IN_ICNT_TO_SHADER, 0);
                            self.interconn.push(
                                device,
                                cluster_id,
                                ic::Packet { fetch, time: cycle },
                                response_packet_size,
                            );
                        }
                    }
                }
            }
        }

        // dram cycle
        for (_i, unit) in self.mem_partition_units.iter_mut().enumerate() {
            crate::timeit!("serial::dram", unit.simple_dram_cycle(cycle));
            // crate::timeit!("serial::dram", unit.try_write().simple_dram_cycle(cycle));
        }

        // receive requests sent to L2 from interconnect and push them to the
        // targeted memory sub partition.
        // Run cycle for each sub partition
        // for (sub_id, mem_sub) in self.mem_sub_partitions.iter().enumerate() {

        // let sub_partitions = crate::MemSubPartitionIterMut {
        //     partition_units: &mut self.mem_partition_units,
        //     sub_partitions_per_partition: self.config.num_sub_partitions_per_memory_controller,
        //     global_sub_id: 0,
        // };

        // for (sub_id, mem_sub) in self.mem_sub_partitions().enumerate() {
        // for mem_sub in sub_partitions {
        for partition in self.mem_partition_units.iter_mut() {
            for mem_sub in partition.sub_partitions.iter_mut() {
                // let mut mem_sub = mem_sub.try_lock();
                // move memory request from interconnect into memory partition
                // (if not backed up)
                //
                // Note:This needs to be called in DRAM clock domain if there
                // is no L2 cache in the system In the worst case, we may need
                // to push SECTOR_CHUNCK_SIZE requests, so ensure you have enough
                // buffer for them
                let device = self.config.mem_id_to_device_id(mem_sub.global_id);

                if mem_sub
                    .interconn_to_l2_queue
                    .can_fit(mem_sub_partition::NUM_SECTORS as usize)
                {
                    if let Some(ic::Packet { fetch, .. }) = self.interconn.pop(device) {
                        assert_eq!(fetch.sub_partition_id(), mem_sub.global_id);
                        log::debug!(
                            "got new fetch {} for mem sub partition {} ({})",
                            fetch,
                            mem_sub.global_id,
                            device
                        );

                        // assert_eq!(cycle, packet.time);
                        // TODO: changed form packet.time to cycle
                        mem_sub.push(fetch, cycle);
                    }
                } else {
                    log::debug!(
                        "SKIP sub partition {} ({}): DRAM full stall",
                        mem_sub.global_id,
                        device
                    );
                    let kernel_id = self
                        .kernel_manager
                        .current_kernel()
                        // .lock()
                        .as_ref()
                        .map(|kernel| kernel.id() as usize);
                    // let mut stats = self.stats.lock();
                    let kernel_stats = self.stats.get_mut(kernel_id);
                    kernel_stats.stall_dram_full += 1;
                }
                // we borrow all of sub here, which is a problem for the cyclic reference in l2
                // interface
                crate::timeit!("serial::subpartitions", mem_sub.cycle(cycle));
            }
        }
    }
}
