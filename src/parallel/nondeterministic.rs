#![allow(warnings, clippy::all)]

use crate::ic::ToyInterconnect;
use crate::sync::{Arc, Mutex, RwLock};
use crate::{
    config, core, engine::cycle::Component, ic, mem_fetch, mem_sub_partition, MockSimulator,
    TIMINGS,
};
use color_eyre::eyre;
use ndarray::prelude::*;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::time::Instant;

#[tracing::instrument]
fn interleaved_serial_cycle<I, C>(
    cycle: u64,
    i: usize,
    progress: &Arc<Mutex<Array3<Option<bool>>>>,
    // cluster_active: bool,
    cores: &Arc<Vec<Vec<Arc<RwLock<crate::core::Core<I>>>>>>,
    sim_orders: &Arc<Vec<Arc<Mutex<VecDeque<usize>>>>>,
    mem_ports: &Arc<Vec<Vec<Arc<Mutex<crate::core::CoreMemoryConnection<C>>>>>>,
    // VecDeque<ic::Packet<(usize, mem_fetch::MemFetch, u32)>>,
    // stats: Arc<Mutex<stats::Stats>>,
    // mem_sub_partitions: Vec<Arc<Mutex<crate::mem_sub_partition::MemorySubPartition>>>,
    // mem_partition_units: Vec<Arc<RwLock<crate::mem_partition_unit::MemoryPartitionUnit>>>,
    interconn: &Arc<I>,
    clusters: &Vec<Arc<RwLock<crate::Cluster<I>>>>,
    // clusters: Vec<Arc<RwLock<crate::Cluster<I>>>>,
    config: &config::GPU,
) where
    C: ic::BufferedConnection<ic::Packet<(usize, mem_fetch::MemFetch, u32)>>,
    I: ic::Interconnect<ic::Packet<mem_fetch::MemFetch>>,
{
    let use_round_robin = config.simt_core_sim_order == config::SchedulingOrder::RoundRobin;

    // std::thread::sleep(std::time::Duration::from_nanos(100));
    for (cluster_id, cluster) in clusters.iter().enumerate() {
        let cluster_active = {
            let progress = progress.lock();
            progress
                .slice(s![i, cluster_id, ..])
                .iter()
                .all(|c| c.unwrap())
        };
        // while !cluster_completed_cycle
        //     .lock()
        //     .get(&cluster_id)
        //     .is_some_and(|&c| c >= cycle + i as u64)
        // {
        //     // busy wait
        //     std::thread::sleep(std::time::Duration::from_nanos(10));
        // }
        //
        // let last_cluster_cycle =
        //     *cluster_completed_cycle.lock().get(&cluster_id).unwrap();
        // assert!(last_cluster_cycle >= cycle + i as u64);

        // let cluster = cluster.try_read();
        // assert_eq!(cluster.cluster_id, cluster_id);

        // let mut active_clusters_per_cycle =
        //     active_clusters_per_cycle.lock();
        // let mut active_cycles_per_cluster =
        //     active_cycles_per_cluster.lock();
        //
        // let was_updated = active_clusters_per_cycle
        //     .entry(cycle + i as u64)
        //     .or_default()
        //     .contains(&cluster.cluster_id);

        // let mut was_updated = false;
        let mut core_sim_order = sim_orders[cluster_id].try_lock();
        // let mut core_sim_order = cluster.core_sim_order.try_lock();

        // let all_ready = || -> bool {
        //     for core_id in core_sim_order.iter() {
        //         let core = &cores[cluster_id][*core_id];
        //         let Some(core) = core.0.try_read() else {
        //             return false;
        //         };
        //         if core.last_cycle < cycle + i as u64 {
        //             return false;
        //         }
        //     }
        //     true
        // };
        //
        // while !all_ready() {
        //     // busy wait
        //     std::thread::sleep(std::time::Duration::from_nanos(100));
        // }

        for core_id in core_sim_order.iter() {
            // let mut port = mem_ports[cluster_id][*core_id].lock();
            // let core = cluster.cores[*core_id].try_read();
            // let core = cores[cluster_id][*core_id].try_read();
            // was_updated |= core.last_cycle >= (cycle + i as u64);
            // was_updated |= core.last_cycle == (cycle + i as u64);

            // lets say we are waiting for core 1 here first
            // meanwhile we want to aquire the lock for core
            //

            // let core_ready = || -> bool {
            //     let core = &cores[cluster_id][*core_id];
            //     let core = core.read();
            //     // let Some(core) = core.0.try_read() else {
            //     //     return false;
            //     // };
            //     core.last_cycle >= cycle
            //     // let ready = core.last_cycle >= cycle + i as u64;
            //     // drop(core);
            //     // ready
            // };
            // // log::info!("waiting for core {core_id}");
            // while !core_ready() {
            //     // busy wait
            //     std::thread::sleep(std::time::Duration::from_nanos(10));
            // }

            // log::info!("core {core_id} ready");
            // let core = cores[cluster_id][*core_id].try_read();
            // log::info!("core {core_id} locked");

            // was_updated |= core.last_active_cycle >= cycle;
            // drop(core);
            // log::info!("core {core_id} unlocked");

            // let mut port = core.mem_port.lock();

            let mut port = mem_ports[cluster_id][*core_id].lock();
            // if !was_updated {
            if cluster_active {
                if !port.buffer.is_empty() {
                    // println!(
                    //     "cluster {} not updated in cycle {} but has full buffer {:#?}",
                    //     cluster.cluster_id,
                    //     cycle + i as u64,
                    //     port.buffer.iter().map(|ic::Packet { data: (_, fetch, _), time }| fetch.to_string()).collect::<Vec<_>>(),
                    // );
                }
                // assert!(port.buffer.is_empty());
            }

            for ic::Packet {
                data: (dest, fetch, size),
                time,
            } in port.buffer.drain()
            {
                interconn.push(cluster_id, dest, ic::Packet { data: fetch, time }, size);
            }

            // drop(port);
        }

        // match (
        //     was_updated,
        //     active_clusters_per_cycle[&(cycle + i as u64)]
        //         .contains(&cluster.cluster_id),
        // ) {
        //     (true, false) => {
        //         panic!("should have been updated");
        //     }
        //     (false, true) => {
        //         panic!("should not have been updated");
        //     }
        //     (false, false) | (true, true) => { /* fine */ }
        // }

        // if was_updated {
        //     if !active_clusters_per_cycle[&(cycle + i as u64)]
        //         .contains(&cluster.cluster_id)
        //     {
        //         // for
        //         // println!("{:?}", active_clusters_per_cycle.lock());
        //         panic!("should have been updated");
        //     }
        //     // assert!(active_clusters.lock().contains(&key));
        // } else {
        //     if active_clusters_per_cycle[&(cycle + i as u64)]
        //         .contains(&cluster.cluster_id)
        //     {
        //         // println!("{:?}", active_clusters.lock());
        //         panic!("should not have been updated");
        //     }
        //     // assert!(!active_clusters.lock().contains(&key));
        // }

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

    // for cluster_id in 0..num_clusters {
    //     let mut core_sim_order = sim_orders[cluster_id].try_lock();
    //     for core_id in core_sim_order.iter() {
    //         // was_updated |= core.last_cycle >= (cycle + i as u64);
    //         let mut port = mem_ports[cluster_id][*core_id].lock();
    //         for ic::Packet {
    //             data: (dest, fetch, size),
    //             time,
    //         } in port.buffer.drain(..)
    //         {
    //             interconn.push(
    //                 cluster_id,
    //                 dest,
    //                 ic::Packet { data: fetch, time },
    //                 size,
    //             );
    //         }
    //     }
    //
    //     if use_round_robin {
    //         core_sim_order.rotate_left(1);
    //     }
    // }

    // after cores complete, run serial cycle
    // self.serial_cycle(cycle + i as u64);

    //
    // // locks are uncontended now
    // self.issue_block_to_core(cycle + i as 64);
}

#[tracing::instrument]
fn new_serial_cycle<I>(
    cycle: u64,
    stats: Arc<Mutex<stats::Stats>>,
    mem_sub_partitions: Vec<Arc<Mutex<crate::mem_sub_partition::MemorySubPartition>>>,
    mem_partition_units: Vec<Arc<RwLock<crate::mem_partition_unit::MemoryPartitionUnit>>>,
    interconn: Arc<I>,
    clusters: Vec<Arc<RwLock<crate::Cluster<I>>>>,
    config: &config::GPU,
) where
    I: ic::Interconnect<ic::Packet<mem_fetch::MemFetch>>,
{
    // it could happen that two serial cycles overlap when using spawn fifo, so we need
    for cluster in &clusters {
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
                mem_sub.push(packet.data, cycle);
                // self.parallel_mem_partition_reqs += 1;
            }
        } else {
            log::debug!("SKIP sub partition {} ({}): DRAM full stall", i, device);
            #[cfg(feature = "stats")]
            {
                stats.lock().stall_dram_full += 1;
            }
        }
        // we borrow all of sub here, which is a problem for the cyclic reference in l2
        // interface
        mem_sub.cache_cycle(cycle);
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

        let num_threads: usize = std::env::var("NUM_THREADS")
            .ok()
            .as_deref()
            .map(str::parse)
            .transpose()?
            .unwrap_or_else(num_cpus::get_physical);
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global();

        println!("nondeterministic [{run_ahead} run ahead] using RAYON");
        println!("\t => launching {num_threads} worker threads");
        println!("\t => interleave serial={interleave_serial}");
        println!("");

        // let num_clusters = self.clusters.len();
        // let cores: Vec<(
        //     Arc<RwLock<crate::Cluster<_>>>,
        //     Arc<RwLock<crate::Core<_>>>,
        //     usize,
        // )> = self
        //     .clusters
        //     .iter()
        //     .flat_map(|cluster| {
        //         cluster
        //             .try_read()
        //             .cores
        //             .iter()
        //             .enumerate()
        //             .map(|(core_id, core)| (Arc::clone(cluster), Arc::clone(core), core_id))
        //             .collect::<Vec<_>>()
        //     })
        //     .collect();

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

        let last_cycle: Vec<Arc<Mutex<Vec<(u64, u64)>>>> = self
            .clusters
            .iter()
            .map(|cluster| {
                Arc::new(Mutex::new(
                    cluster.try_read().cores.iter().map(|_| (0, 0)).collect(),
                ))
            })
            .collect();

        let last_cycle: Arc<Mutex<Vec<Vec<u64>>>> = Arc::new(Mutex::new(
            self.clusters
                .iter()
                .map(|cluster| {
                    let num_cores = cluster.try_read().cores.len();
                    vec![0; num_cores]
                    // iter().map(|_| 0).collect()
                    // Arc::new(Mutex::new(
                    //     cluster.try_read().cores.iter().map(|_| (0, 0)).collect(),
                    // ))
                })
                .collect(),
        ));

        let num_clusters = self.clusters.len();
        let num_cores_per_cluster = self.clusters[0].try_read().cores.len();
        let shape = (run_ahead, num_clusters, num_cores_per_cluster);
        let progress = Array3::<Option<bool>>::from_elem(shape, None);

        let progress = Arc::new(Mutex::new(progress));
        let cores = Arc::new(cores);
        let sim_orders = Arc::new(sim_orders);
        let mem_ports = Arc::new(mem_ports);

        let use_round_robin =
            self.config.simt_core_sim_order == config::SchedulingOrder::RoundRobin;

        let mut cycle: u64 = 0;

        use std::collections::{HashMap, HashSet};

        // rayon::scope_fifo(|s| {
        while (self.commands_left() || self.kernels_left()) && !self.reached_limit(cycle) {
            self.process_commands(cycle);
            self.launch_kernels(cycle);

            // let mut active_clusters_per_cycle: Arc<Mutex<HashMap<u64, HashSet<usize>>>> =
            //     Default::default();
            // let mut active_cycles_per_cluster: Arc<Mutex<HashMap<usize, HashSet<u64>>>> =
            //     Default::default();
            // let mut cluster_completed_cycle: Arc<Mutex<HashMap<usize, u64>>> = Default::default();

            let mut finished_kernel = None;
            loop {
                // log::info!("======== cycle {cycle} ========");
                // println!("======== cycle {cycle} ========");

                if self.reached_limit(cycle) || !self.active() {
                    break;
                }

                let span = tracing::span!(tracing::Level::INFO, "wave", cycle, run_ahead);
                let enter = span.enter();
                if interleave_serial {
                    rayon::scope_fifo(|wave| {
                        for i in 0..run_ahead {
                            // for (cluster_arc, core, core_id) in cores.iter() {
                            for (cluster_id, cluster_arc) in
                                self.clusters.iter().cloned().enumerate()
                            {
                                let running_kernels = self.running_kernels.clone();

                                let last_cycle = last_cycle.clone();
                                let progress = progress.clone();
                                // let cluster_completed_cycle = cluster_completed_cycle.clone();
                                // let active_clusters_per_cycle = active_clusters_per_cycle.clone();
                                // let active_cycles_per_cluster = active_cycles_per_cluster.clone();

                                // small optimizations
                                let sim_orders = sim_orders.clone();
                                let mem_ports = mem_ports.clone();
                                let cores = cores.clone();

                                let interconn = self.interconn.clone();
                                let clusters = self.clusters.clone();

                                let stats = self.stats.clone();
                                let mem_sub_partitions = self.mem_sub_partitions.clone();
                                let mem_partition_units = self.mem_partition_units.clone();
                                let config = self.config.clone();

                                wave.spawn_fifo(move |intra_wave| {
                                    // if *core_id == 0 {
                                    //     cluster.write().interconn_cycle(cycle + i as u64);
                                    // }
                                    let cluster = cluster_arc.try_read();
                                    assert_eq!(cluster.cluster_id, cluster_id);

                                    let kernels_completed = running_kernels
                                        .try_read()
                                        .iter()
                                        .filter_map(std::option::Option::as_ref)
                                        .all(|k| k.no_more_blocks_to_run());
                                    let cores_completed = cluster.not_completed() == 0;

                                    // if !(cores_completed && kernels_completed) {
                                    //     // active_clusters_per_cycle
                                    //     //     .lock()
                                    //     //     .entry(cycle + i as u64)
                                    //     //     .or_default()
                                    //     //     .insert(cluster.cluster_id);
                                    //     // active_cycles_per_cluster
                                    //     //     .lock()
                                    //     //     .entry(cluster.cluster_id)
                                    //     //     .or_default()
                                    //     //     .insert(cycle + i as u64);
                                    //
                                    //     for core in cluster.cores.iter().cloned() {
                                    //         intra_wave.spawn_fifo(move |_| {
                                    //             core.write().cycle(cycle + i as u64);
                                    //         });
                                    //     }
                                    // }

                                    for (core_id, core) in cluster.cores.iter().cloned().enumerate()
                                    {
                                        let last_cycle = last_cycle.clone();
                                        let progress = progress.clone();

                                        // small optimizations
                                        let sim_orders = sim_orders.clone();
                                        let mem_ports = mem_ports.clone();
                                        let cores = cores.clone();

                                        let interconn = interconn.clone();
                                        let clusters = clusters.clone();

                                        let stats = stats.clone();
                                        let mem_sub_partitions = mem_sub_partitions.clone();
                                        let mem_partition_units = mem_partition_units.clone();
                                        let config = config.clone();

                                        let cluster_active =
                                            !(cores_completed && kernels_completed);

                                        intra_wave.spawn_fifo(move |_| {
                                            let mut core = core.write();
                                            if cluster_active {
                                                core.cycle(cycle + i as u64);
                                                core.last_cycle =
                                                    core.last_cycle.max(cycle + i as u64);
                                                core.last_active_cycle =
                                                    core.last_active_cycle.max(cycle + i as u64);
                                            } else {
                                                core.last_cycle =
                                                    core.last_cycle.max(cycle + i as u64);
                                            }
                                            drop(core);

                                            // check if core is last to finish cycle + i
                                            // let last_to_finish = {
                                            //     let mut current_cycle = last_cycle.lock();
                                            //     current_cycle[cluster_id][core_id] =
                                            //         cycle + i as u64;
                                            //
                                            //     current_cycle
                                            //         .iter()
                                            //         .flat_map(|cores| cores)
                                            //         .all(|&c| c >= cycle + i as u64)
                                            // };
                                            let last_to_finish = {
                                                let mut progress = progress.lock();
                                                progress[[i, cluster_id, core_id]] =
                                                    Some(cluster_active);

                                                progress
                                                    .slice(s![i, .., ..])
                                                    .iter()
                                                    .all(|&c| c.is_some())
                                                // current_cycle
                                                //     .iter()
                                                //     .flat_map(|cores| cores)
                                                //     .all(|&c| c >= cycle + i as u64)
                                            };

                                            if last_to_finish {
                                                // println!(
                                                //     "core {:?} is last to finish cycle {}",
                                                //     (cluster_id, core_id),
                                                //     cycle + i as u64
                                                // );
                                                interleaved_serial_cycle(
                                                    cycle + i as u64,
                                                    i,
                                                    &progress,
                                                    // cluster_active,
                                                    &cores,
                                                    &sim_orders,
                                                    &mem_ports,
                                                    &interconn,
                                                    &clusters,
                                                    &config,
                                                );
                                                new_serial_cycle(
                                                    cycle + i as u64,
                                                    stats,
                                                    mem_sub_partitions,
                                                    mem_partition_units,
                                                    interconn,
                                                    clusters,
                                                    &config,
                                                );
                                            }
                                        });
                                        // } else {
                                        // let mut core = core.write();
                                        // let (last_core_cycle, last_active_core_cycle) =
                                        //     last_cycle[cluster.cluster_id].lock()[core_id];
                                        // }
                                    }

                                    // cluster_completed_cycle
                                    //     .lock()
                                    //     .entry(cluster.cluster_id)
                                    //     .and_modify(|c| *c = (*c).max(cycle + i as u64))
                                    //     .or_insert(cycle + i as u64);
                                });
                            }

                            // // small optimizations
                            // let sim_orders = sim_orders.clone();
                            // let mem_ports = mem_ports.clone();
                            // let cores = cores.clone();
                            //
                            // let interconn = self.interconn.clone();
                            // let clusters = self.clusters.clone();
                            //
                            // let stats = self.stats.clone();
                            // let mem_sub_partitions = self.mem_sub_partitions.clone();
                            // let mem_partition_units = self.mem_partition_units.clone();
                            // let config = self.config.clone();

                            // let cluster_completed_cycle = cluster_completed_cycle.clone();
                            // let active_clusters_per_cycle = active_clusters_per_cycle.clone();
                            // let active_cycles_per_cluster = active_cycles_per_cluster.clone();

                            // wave.spawn_fifo(move |_| {
                            //     interleaved_serial_cycle(
                            //         cycle + i as u64,
                            //         &cores,
                            //         &sim_orders,
                            //         &mem_ports,
                            //         &interconn,
                            //         &clusters,
                            //         &config,
                            //     );
                            //     new_serial_cycle(
                            //         cycle + i as u64,
                            //         stats,
                            //         mem_sub_partitions,
                            //         mem_partition_units,
                            //         interconn,
                            //         clusters,
                            //         &config,
                            //     );
                            // });
                        }
                    });
                    // all run_ahead cycles completed
                    progress.lock().fill(None);
                }

                if !interleave_serial {
                    // rayon::scope_fifo(|wave| {
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
                            for cluster_arc in self.clusters.iter() {
                                // if *core_id == 0 {
                                //     cluster.write().interconn_cycle(cycle + i as u64);
                                // }

                                let cluster = cluster_arc.try_read();
                                let kernels_completed = self
                                    .running_kernels
                                    .try_read()
                                    .iter()
                                    .filter_map(std::option::Option::as_ref)
                                    .all(|k| k.no_more_blocks_to_run());

                                let cores_completed = cluster.not_completed() == 0;

                                if cores_completed && kernels_completed {
                                    continue;
                                }
                                for core in cluster.cores.iter().cloned() {
                                    core_scope.spawn(move |_| {
                                        let mut core = core.write();
                                        core.cycle(cycle + i as u64);
                                        core.last_cycle = core.last_cycle.max(cycle + i as u64);
                                        core.last_active_cycle =
                                            core.last_active_cycle.max(cycle + i as u64);
                                    });
                                }
                            }
                        });

                        for (cluster_id, cluster) in self.clusters.iter().enumerate() {
                            // check if cluster has been updated
                            // }
                            // for cluster_id in 0..num_clusters {
                            // let mut core_sim_order = sim_orders[cluster_id].try_lock();
                            let mut was_updated = false;
                            let cluster = cluster.try_read();
                            assert_eq!(cluster.cluster_id, cluster_id);

                            let mut core_sim_order = cluster.core_sim_order.try_lock();
                            for core_id in core_sim_order.iter() {
                                let core = cluster.cores[*core_id].try_read();
                                was_updated |= core.last_active_cycle >= (cycle + i as u64);

                                let mut port = core.mem_port.lock();
                                if !was_updated {
                                    assert!(port.buffer.is_empty());
                                }
                                // let mut port = mem_ports[cluster_id][*core_id].lock();
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

                            if was_updated {
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
                    // });
                }

                self.issue_block_to_core(cycle);

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
        // });
        log::info!("exit after {cycle} cycles");
        dbg!(&cycle);

        Ok(())
    }

    #[tracing::instrument]
    pub fn run_to_completion_parallel_nondeterministic_old(
        &mut self,
        mut run_ahead: usize,
    ) -> eyre::Result<()> {
        run_ahead = run_ahead.max(1);

        let num_threads: usize = std::env::var("NUM_THREADS")
            .ok()
            .as_deref()
            .map(str::parse)
            .transpose()?
            .unwrap_or_else(num_cpus::get_physical);

        let cores_per_thread = self.clusters.len() as f64 / num_threads as f64;
        // prefer less cores
        let cores_per_thread = cores_per_thread.ceil() as usize;
        // todo: tune this
        let core_chunks: Vec<Vec<(_, usize, Vec<_>)>> = self
            .clusters
            .chunks(cores_per_thread)
            .map(|clusters| {
                clusters
                    .iter()
                    .map(|cluster| {
                        let cluster = cluster.try_read();
                        (
                            Arc::clone(&cluster.core_sim_order),
                            cluster.cluster_id,
                            cluster
                                .cores
                                .iter()
                                .map(|core| (core.clone(), core.try_read().mem_port.clone()))
                                .collect(),
                        )
                    })
                    .collect()
            })
            .collect();
        let num_chunks = core_chunks.len();

        println!("non deterministic [{run_ahead} run ahead]");
        println!("\t => launching {num_chunks} threads with {cores_per_thread} cores per thread");

        let core_reached: Vec<_> = vec![crossbeam::channel::bounded(1); run_ahead];

        let start_core: Vec<_> = vec![crossbeam::channel::bounded(1); num_chunks];

        let core_done: Vec<_> = vec![crossbeam::channel::bounded(1); num_chunks];

        let lockstep = true;

        let use_round_robin =
            self.config.simt_core_sim_order == config::SchedulingOrder::RoundRobin;

        // spawn worker threads for core cycles
        let core_worker_handles: Vec<_> = core_chunks
            .into_iter()
            .enumerate()
            .map(|(cluster_idx, clusters)| {
                let (_, start_core_rx) = start_core[cluster_idx].clone();
                let (core_done_tx, _) = core_done[cluster_idx].clone();
                let core_reached_tx: Vec<_> =
                    core_reached.iter().map(|(tx, _)| tx).cloned().collect();
                // let running_kernels = self.running_kernels.clone();
                let interconn = Arc::clone(&self.interconn);

                std::thread::spawn(move || loop {
                    let Ok(cycle) = start_core_rx.recv() else {
                        // println!("cluster {} exited", cluster.try_read().cluster_id);
                        break;
                    };

                    for i in 0..run_ahead {
                        // let kernels_completed = running_kernels
                        //     .try_read()
                        //     .unwrap()
                        //     .iter()
                        //     .filter_map(std::option::Option::as_ref)
                        //     .all(|k| k.no_more_blocks_to_run());
                        tracing::info!("cycle {cycle} + run ahead {i}");

                        for (core_sim_order, cluster_id, cores) in &clusters {
                            // let mut cluster = cluster.read();
                            // let cores_completed = cluster.not_completed() == 0;
                            // let cluster_done = cores_completed && kernels_completed;
                            // let start = Instant::now();
                            let cluster_done = false;
                            if !cluster_done {
                                for (core, _) in cores {
                                    let mut core = core.write();
                                    // println!("start core {:?} ({} clusters)", core.id(), num_cores);
                                    crate::timeit!("parallel::core", core.cycle(cycle));
                                    // println!("done core {:?} ({} clusters)", core.id(), num_cores);
                                }
                            }

                            let mut core_sim_order = core_sim_order.try_lock();
                            for core_id in core_sim_order.iter() {
                                let (_core, mem_port) = &cores[*core_id];
                                let mut port = mem_port.try_lock();
                                for ic::Packet {
                                    data: (dest, fetch, size),
                                    time,
                                } in port.buffer.drain(..)
                                {
                                    interconn.push(
                                        *cluster_id,
                                        dest,
                                        ic::Packet { data: fetch, time },
                                        size,
                                    );
                                }
                            }

                            if use_round_robin {
                                core_sim_order.rotate_left(1);
                            }

                            // #[cfg(feature = "stats")]
                            // {
                            //     TIMINGS
                            //         .lock()
                            //         .entry("parallel::cluster")
                            //         .or_default()
                            //         .add(start.elapsed());
                            // }

                            // issue new blocks
                            // issue_block_to_core
                        }

                        // collect the core packets pushed to the interconn
                        // for cluster in &clusters {
                        //     let mut cluster = cluster.write();
                        // }

                        if lockstep {
                            core_reached_tx[i].send(()).unwrap();
                        }
                        // std::thread::yield_now();
                    }

                    core_done_tx.send(()).unwrap();
                })
            })
            .collect();

        assert_eq!(core_worker_handles.len(), num_chunks);

        let mut cycle: u64 = 0;
        while (self.commands_left() || self.kernels_left()) && !self.reached_limit(cycle) {
            self.process_commands(cycle);
            self.launch_kernels(cycle);

            let mut finished_kernel = None;
            loop {
                if self.reached_limit(cycle) || !self.active() {
                    break;
                }

                let span = tracing::span!(tracing::Level::INFO, "wave", cycle, run_ahead);
                let enter = span.enter();

                // start all cores
                tracing::warn!("WAVE START");
                for core_idx in 0..num_chunks {
                    start_core[core_idx].0.send(cycle).unwrap();
                }

                for i in 0..run_ahead {
                    tracing::info!("cycle {cycle} + run ahead {i}");
                    if lockstep {
                        // wait until all cores are ready for this
                        // println!("waiting for cores to reach barrier {i}");
                        for _ in 0..num_chunks {
                            core_reached[i].1.recv().unwrap();
                        }
                        // println!("all cores reached reached barrier {i}");
                    }

                    log::info!("======== cycle {cycle} ========");
                    // log::info!("");

                    // could enforce round robin here

                    crate::timeit!("SERIAL CYCLE", self.serial_cycle(cycle + i as u64));

                    // issue new blocks
                    // let start = Instant::now();
                    // self.issue_block_to_core();
                    // #[cfg(feature = "stats")]
                    // {
                    //     TIMINGS
                    //         .lock()
                    //         .entry("serial::issue_block_to_core")
                    //         .or_default()
                    //         .add(start.elapsed());
                    // }
                }

                // wait for all cores to finish
                for core_idx in 0..num_chunks {
                    core_done[core_idx].1.recv().unwrap();
                }

                // locks are uncontended now
                crate::timeit!("SERIAL ISSUE", self.issue_block_to_core(cycle));
                drop(enter);

                cycle += run_ahead as u64;
                self.set_cycle(cycle);

                // dbg!(self.active());

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

        log::info!("exit after {cycle} cycles");
        dbg!(&cycle);
        Ok(())
    }

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
                #[cfg(feature = "stats")]
                {
                    self.stats.lock().stall_dram_full += 1;
                }
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
