#![allow(warnings)]

use crate::ic::ToyInterconnect;
use crate::sync::{Arc, Mutex, RwLock};
use crate::{
    config, core, engine::cycle::Component, ic, mem_fetch, mem_sub_partition, MockSimulator,
    TIMINGS,
};
use color_eyre::eyre;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::time::Instant;

#[tracing::instrument]
fn new_serial_cycle<I, T, Q>(
    cycle: u64,
    stats: Arc<Mutex<stats::Stats>>,
    mem_sub_partitions: Vec<Mutex<crate::mem_sub_partition::MemorySubPartition<Q>>>,
    mem_partition_units: Vec<RwLock<crate::mem_partition_unit::MemoryPartitionUnit>>,
    interconn: Arc<I>,
    clusters: Vec<Arc<RwLock<crate::Cluster<I>>>>,
    config: &config::GPU,
) where
    Q: crate::fifo::Queue<mem_fetch::MemFetch> + 'static,
    I: ic::Interconnect<crate::Packet>,
    T: std::fmt::Debug,
{
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
                let packet = core::Packet::Fetch(fetch);
                // fetch.set_return_timestamp(gpu_sim_cycle + gpu_tot_sim_cycle);
                // , gpu_sim_cycle + gpu_tot_sim_cycle);
                // drop(fetch);
                interconn.push(device, cluster_id, packet, response_packet_size);
                // self.partition_replies_in_parallel += 1;
            } else {
                // self.gpu_stall_icnt2sh += 1;
            }
        }
    }

    for (_i, unit) in mem_partition_units.iter().enumerate() {
        unit.try_write().simple_dram_cycle();
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
        if mem_sub.interconn_to_l2_can_fit(mem_sub_partition::SECTOR_CHUNCK_SIZE as usize) {
            if let Some(core::Packet::Fetch(fetch)) = interconn.pop(device) {
                log::debug!(
                    "got new fetch {} for mem sub partition {} ({})",
                    fetch,
                    i,
                    device
                );

                mem_sub.push(fetch, cycle);
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
    I: ic::Interconnect<core::Packet> + 'static,
{
    #[tracing::instrument]
    pub fn run_to_completion_parallel_nondeterministic(
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
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .unwrap();
        println!("nondeterministic [{run_ahead} run ahead] using RAYON");
        println!("\t => launching {num_threads} worker threads");

        let num_clusters = self.clusters.len();
        let cores: Vec<(
            Arc<RwLock<crate::Cluster<_>>>,
            Arc<RwLock<crate::Core<_>>>,
            usize,
        )> = self
            .clusters
            .iter()
            .flat_map(|cluster| {
                cluster
                    .try_read()
                    .cores
                    .iter()
                    .enumerate()
                    .map(|(core_id, core)| (Arc::clone(&cluster), Arc::clone(&core), core_id))
                    .collect::<Vec<_>>()
            })
            .collect();

        let sim_orders: Vec<Arc<_>> = self
            .clusters
            .iter()
            .map(|cluster| Arc::clone(&cluster.try_read().core_sim_order))
            .collect();
        let interconn_ports: Vec<Vec<Arc<_>>> = self
            .clusters
            .iter()
            .map(|cluster| {
                cluster
                    .try_read()
                    .cores
                    .iter()
                    .map(|core| Arc::clone(&core.try_read().interconn_port))
                    .collect()
            })
            .collect();

        let cores = Arc::new(cores);
        let sim_orders = Arc::new(sim_orders);
        let interconn_ports = Arc::new(interconn_ports);

        let use_round_robin =
            self.config.simt_core_sim_order == config::SchedulingOrder::RoundRobin;

        let mut cycle: u64 = 0;

        rayon::scope_fifo(|s| {
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

                    rayon::scope(|wave| {
                        for i in 0..run_ahead {
                            // run cores in any order
                            // rayon::scope(|core_scope| {
                            for (cluster, core, core_id) in cores.iter() {
                                // core_scope.spawn(move |_| {
                                wave.spawn(move |_| {
                                    if *core_id == 0 {
                                        cluster.write().interconn_cycle(cycle);
                                    }

                                    core.write().cycle(cycle);
                                });
                            }
                            // });

                            // let sim_orders = sim_orders.clone();
                            // let interconn_ports = interconn_ports.clone();
                            // let interconn = self.interconn.clone();
                            // s.spawn_fifo(move |_| {
                            for cluster_id in 0..num_clusters {
                                let mut core_sim_order = sim_orders[cluster_id].try_lock();
                                for core_id in core_sim_order.iter() {
                                    let mut port = interconn_ports[cluster_id][*core_id].lock();
                                    for (dest, fetch, size) in port.drain(..) {
                                        self.interconn.push(
                                            cluster_id,
                                            dest,
                                            core::Packet::Fetch(fetch),
                                            size,
                                        );
                                    }
                                }

                                if use_round_robin {
                                    core_sim_order.rotate_left(1);
                                }
                            }

                            // after cores complete, run serial cycle
                            self.serial_cycle(cycle + i as u64);
                            // new_serial_cycle(
                            //     cycle + i as u64,
                            //     stats: Arc<Mutex<stats::Stats>>,
                            //     mem_sub_partitions:
                            //         Vec<Mutex<crate::mem_sub_partition::MemorySubPartition<Q>>>,
                            //     mem_partition_units:
                            //         Vec<RwLock<crate::mem_partition_unit::MemoryPartitionUnit>>,
                            //     interconn: Arc<I>,
                            //     clusters: Vec<Arc<RwLock<crate::Cluster<I>>>>,
                            //     config: &config::GPU,
                            // );
                            //
                            // // locks are uncontended now
                            // self.issue_block_to_core(cycle);
                            // });
                        }
                    });

                    self.issue_block_to_core(cycle);

                    drop(enter);

                    cycle += run_ahead as u64;
                    // cycle += 1;
                    self.set_cycle(cycle);

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
        });

        // let mut cycle: u64 = 0;
        // while (self.commands_left() || self.kernels_left()) && !self.reached_limit(cycle) {
        //     self.process_commands(cycle);
        //     self.launch_kernels(cycle);
        //
        //     let mut finished_kernel = None;
        //     loop {
        //         if self.reached_limit(cycle) || !self.active() {
        //             break;
        //         }
        //
        //         let span = tracing::span!(tracing::Level::INFO, "wave", cycle, run_ahead);
        //         let enter = span.enter();
        //
        //         // for i in 0..run_ahead {
        //         let i = 0;
        //         // TODO: make this in place
        //         // rayon::in_place_scope_fifo(|s| {
        //         rayon::scope_fifo(|s| {
        //             // run cores in any order
        //             rayon::scope(|core_scope| {
        //                 for core in cores.iter() {
        //                     core_scope.spawn(move |_| {
        //                         core.write().cycle(cycle);
        //                     });
        //                 }
        //             });
        //
        //             // let sim_orders = sim_orders.clone();
        //             // let interconn_ports = interconn_ports.clone();
        //             // s.spawn_fifo(move |_| {
        //             for cluster_id in 0..num_clusters {
        //                 let mut core_sim_order = sim_orders[cluster_id].try_lock();
        //                 for core_id in core_sim_order.iter() {
        //                     let mut port = interconn_ports[cluster_id][*core_id].try_lock();
        //                     for (dest, fetch, size) in port.drain(..) {
        //                         // self.interconn.push(
        //                         //     cluster_id,
        //                         //     dest,
        //                         //     core::Packet::Fetch(fetch),
        //                         //     size,
        //                         // );
        //                     }
        //                 }
        //
        //                 if use_round_robin {
        //                     core_sim_order.rotate_left(1);
        //                 }
        //             }
        //             // after cores complete, run serial cycle
        //             self.serial_cycle(cycle + i as u64);
        //             // })
        //
        //             // s.spawn_fifo(|s| {
        //             //     // task s.1
        //             //     s.spawn_fifo(|s| {
        //             //         // task s.1.1
        //             //         rayon::scope_fifo(|t| {
        //             //             t.spawn_fifo(|_| ()); // task t.1
        //             //             t.spawn_fifo(|_| ()); // task t.2
        //             //         });
        //             //     });
        //             // });
        //             // s.spawn_fifo(|s| { // task s.2
        //             // });
        //             // point mid
        //         });
        //
        //         // locks are uncontended now
        //         self.issue_block_to_core(cycle);
        //         drop(enter);
        //
        //         cycle += run_ahead as u64;
        //         self.set_cycle(cycle);
        //
        //         if !self.active() {
        //             finished_kernel = self.finished_kernel();
        //             if finished_kernel.is_some() {
        //                 break;
        //             }
        //         }
        //     }
        //
        //     if let Some(kernel) = finished_kernel {
        //         self.cleanup_finished_kernel(&kernel);
        //     }
        //
        //     log::trace!(
        //         "commands left={} kernels left={}",
        //         self.commands_left(),
        //         self.kernels_left()
        //     );
        // }

        // let clustersx: Vec<_> = self.clusters;

        // let run_core = |(core, core_sim_order): (
        //     Arc<RwLock<crate::Cluster<_>>>,
        //     Arc<Mutex<VecDeque<usize>>>,
        // )| {
        // let run_core = |core: Arc<RwLock<crate::Core<_>>>| {
        //     // c.write().cycle(cycle);
        //     // for c in core {
        //     //     c.write().cycle(cycle);
        //     // }
        //     //
        //     // let mut core_sim_order = core_sim_order.try_lock();
        //     // for core_id in core_sim_order.iter() {
        //     //     // let (_core, interconn_port) = &cores[*core_id];
        //     //     // let mut port = interconn_port.try_lock();
        //     //     // for (dest, fetch, size) in port.drain(..) {
        //     //     // interconn.push(
        //     //     //     *cluster_id,
        //     //     //     dest,
        //     //     //     core::Packet::Fetch(fetch),
        //     //     //     size,
        //     //     // );
        //     //     // }
        //     // }
        //     //
        //     // if use_round_robin {
        //     //     core_sim_order.rotate_left(1);
        //     // }
        // };

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
                                .map(|core| (core.clone(), core.try_read().interconn_port.clone()))
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
                let running_kernels = self.running_kernels.clone();
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
                                let (_core, interconn_port) = &cores[*core_id];
                                let mut port = interconn_port.try_lock();
                                for (dest, fetch, size) in port.drain(..) {
                                    interconn.push(
                                        *cluster_id,
                                        dest,
                                        core::Packet::Fetch(fetch),
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
                            let _ = core_reached[i].1.recv().unwrap();
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
                // self.serial_issue_block_to_core(cycle);
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
    fn serial_issue_block_to_core(&mut self, cycle: u64) {
        // let num_cores = self.cores.len();
        //
        // log::debug!(
        //     "cluster {}: issue block to core for {} cores",
        //     self.cluster_id,
        //     num_cores
        // );
        // let mut num_blocks_issued = 0;
        //
        // let mut block_issue_next_core = self.block_issue_next_core.try_lock();
        //
        // for core_id in 0..num_cores {
        //     let core_id = (core_id + *block_issue_next_core + 1) % num_cores;
        //     // let core = &mut cores[core_id];
        //     // THIS KILLS THE PERFORMANCE
        //     let core = self.cores[core_id].read();
        //
        //     // let kernel: Option<Arc<Kernel>> = if self.config.concurrent_kernel_sm {
        //     //     // always select latest issued kernel
        //     //     // kernel = sim.select_kernel()
        //     //     // sim.select_kernel().map(Arc::clone);
        //     //     unimplemented!("concurrent kernel sm");
        //     // } else {
        //     let mut current_kernel = core.current_kernel.try_lock().clone();
        //     let should_select_new_kernel = if let Some(ref current) = current_kernel {
        //         // if no more blocks left, get new kernel once current block completes
        //         current.no_more_blocks_to_run() && core.not_completed() == 0
        //     } else {
        //         // core was not assigned a kernel yet
        //         true
        //     };
        //
        //     // if let Some(ref current) = current_kernel {
        //     //     log::debug!(
        //     //         "core {}-{}: current kernel {}, more blocks={}, completed={}",
        //     //         self.cluster_id,
        //     //         core_id,
        //     //         current,
        //     //         !current.no_more_blocks_to_run(),
        //     //         core.not_completed() == 0,
        //     //     );
        //     // }
        //
        //     // dbg!(&should_select_new_kernel);
        //     if should_select_new_kernel {
        //         current_kernel = crate::timeit!(self.select_kernel());
        //         // current_kernel = sim.select_kernel();
        //         // if let Some(ref k) = current_kernel {
        //         //     log::debug!("kernel {} bind to core {:?}", kernel, self.id());
        //         //     // core.set_kernel(Arc::clone(k));
        //         // }
        //     }
        //
        //     //     current_kernel
        //     // };
        //
        //     // if let Some(kernel) = kernel {
        //     if let Some(kernel) = current_kernel {
        //         log::debug!(
        //             "core {}-{}: selected kernel {} more blocks={} can issue={}",
        //             self.cluster_id,
        //             core_id,
        //             kernel,
        //             !kernel.no_more_blocks_to_run(),
        //             core.can_issue_block(&kernel),
        //         );
        //
        //         let can_issue = !kernel.no_more_blocks_to_run() && core.can_issue_block(&kernel);
        //         drop(core);
        //         if can_issue {
        //             let mut core = self.cores[core_id].write();
        //             core.issue_block(&kernel);
        //             num_blocks_issued += 1;
        //             *block_issue_next_core = core_id;
        //             break;
        //         }
        //     } else {
        //         log::debug!(
        //             "core {}-{}: selected kernel NULL",
        //             self.cluster_id,
        //             core.core_id,
        //         );
        //     }
        // }
        // num_blocks_issued
    }

    #[tracing::instrument]
    fn serial_cycle(&mut self, cycle: u64) {
        // if false {
        //     let start = Instant::now();
        //     self.issue_block_to_core();
        //     TIMINGS
        //         .lock()
        //         .entry("serial::issue_block_to_core")
        //         .or_default()
        //         .add(start.elapsed());
        // }

        // let start = Instant::now();
        // for cluster in &self.clusters {
        //     cluster.write().interconn_cycle(cycle);
        // }
        // #[cfg(feature = "stats")]
        // {
        //     TIMINGS
        //         .lock()
        //         .entry("serial::interconn_cycle")
        //         .or_default()
        //         .add(start.elapsed());
        // }

        // let start = Instant::now();
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
                    let packet = core::Packet::Fetch(fetch);
                    // fetch.set_return_timestamp(gpu_sim_cycle + gpu_tot_sim_cycle);
                    // , gpu_sim_cycle + gpu_tot_sim_cycle);
                    // drop(fetch);
                    self.interconn
                        .push(device, cluster_id, packet, response_packet_size);
                    // self.partition_replies_in_parallel += 1;
                } else {
                    // self.gpu_stall_icnt2sh += 1;
                }
            }
        }
        // #[cfg(feature = "stats")]
        // {
        //     TIMINGS
        //         .lock()
        //         .entry("serial::subs")
        //         .or_default()
        //         .add(start.elapsed());
        // }

        // let start = Instant::now();
        for (_i, unit) in self.mem_partition_units.iter().enumerate() {
            unit.try_write().simple_dram_cycle();
        }
        // #[cfg(feature = "stats")]
        // {
        //     TIMINGS
        //         .lock()
        //         .entry("serial::dram")
        //         .or_default()
        //         .add(start.elapsed());
        // }

        // let start = Instant::now();
        for (i, mem_sub) in self.mem_sub_partitions.iter().enumerate() {
            // let mut mem_sub = mem_sub.try_borrow_mut().unwrap();
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
            if mem_sub.interconn_to_l2_can_fit(mem_sub_partition::SECTOR_CHUNCK_SIZE as usize) {
                if let Some(core::Packet::Fetch(fetch)) = self.interconn.pop(device) {
                    log::debug!(
                        "got new fetch {} for mem sub partition {} ({})",
                        fetch,
                        i,
                        device
                    );

                    mem_sub.push(fetch, cycle);
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
        // #[cfg(feature = "stats")]
        // {
        //     TIMINGS
        //         .lock()
        //         .entry("serial::l2")
        //         .or_default()
        //         .add(start.elapsed());
        // }

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
