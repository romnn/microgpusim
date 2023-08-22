#![allow(warnings)]

use crate::{
    config, core, engine::cycle::Component, ic, mem_fetch, mem_sub_partition, MockSimulator,
    TIMINGS,
};
use color_eyre::eyre;
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

impl<I> MockSimulator<I>
where
    I: ic::Interconnect<core::Packet> + 'static,
{
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

        let cores_per_thread = self.clusters.len() as f64 / num_threads as f64;
        // prefer less cores
        let cores_per_thread = cores_per_thread.ceil() as usize;
        // todo: tune this
        // let cores: Vec<_> = self.clusters.iter().cloned().collect();
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

        // use std::sync::Semaphore;
        // use parking_lot::Condvar;

        // let core_reached: Vec<_> = vec![Semaphore::new(num_chunks); run_ahead];

        // let core_ = Arc::new((Mutex::new(false), Condvar::new()));
        // let start_core: Vec<_> = Semaphore::new(num_chunks);
        //
        // let core_done: Vec<_> = Semaphore::new(num_chunks);

        let lockstep = false;

        // let (start_serial_tx, start_serial_rx) = crossbeam::channel::bounded(1);
        // let (serial_done_tx, serial_done_rx) = crossbeam::channel::bounded(1);
        //
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

                        for (core_sim_order, cluster_id, cores) in &clusters {
                            // let mut cluster = cluster.read();
                            // let cores_completed = cluster.not_completed() == 0;
                            // let cluster_done = cores_completed && kernels_completed;
                            let start = Instant::now();
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

                            #[cfg(feature = "stats")]
                            {
                                TIMINGS
                                    .lock()
                                    .entry("parallel::cluster")
                                    .or_default()
                                    .add(start.elapsed());
                            }

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
                        std::thread::yield_now();
                    }

                    core_done_tx.send(()).unwrap();
                })
            })
            .collect();

        assert_eq!(core_worker_handles.len(), num_chunks);

        // // crossbeam::thread::scope(|s| { s.spawn(move |s| loop {
        // std::thread::spawn(move || loop {
        //     let Ok(cycle) = start_serial_rx.recv() else {
        //         // println!("cluster {} exited", cluster.try_read().cluster_id);
        //         break;
        //     };
        //
        //     for i in 0..run_ahead {
        //         // wait until all cores are ready for this
        //         println!("waiting for cores to reach barrier {i}");
        //         for _ in 0..num_cores {
        //             // let _ = core_reached_rx[i].recv().unwrap();
        //             let _ = core_reached[i].1.recv().unwrap();
        //         }
        //         println!("all cores reached reached barrier {i}");
        //         log::info!("======== cycle {cycle} ========");
        //         log::info!("");
        //
        //         // collect the core packets pushed to the interconn
        //         for cluster in &self.clusters {
        //             let mut cluster = cluster.write();
        //             for core_id in &cluster.core_sim_order {
        //                 let core = cluster.cores[*core_id].read();
        //                 let mut port = core.interconn_portlock();
        //                 for (dest, fetch, size) in port.drain(..) {
        //                     self.interconn.push(
        //                         core.cluster_id,
        //                         dest,
        //                         core::Packet::Fetch(fetch),
        //                         size,
        //                     );
        //                 }
        //             }
        //
        //             if let config::SchedulingOrder::RoundRobin = self.config.simt_core_sim_order {
        //                 cluster.core_sim_order.rotate_left(1);
        //             }
        //         }
        //
        //         if self.reached_limit(cycle) || !self.active() {
        //             break;
        //         }
        //
        //         self.serial_cycle(cycle + i as u64);
        //         serial_done_tx.send(()).unwrap();
        //     }
        // });
        // // });

        let mut cycle: u64 = 0;
        // loop {
        //     // start serial thread
        //     start_serial_tx.send(cycle).unwrap();
        //
        //     // start all cores
        //     for core_idx in 0..num_cores {
        //         start_core[core_idx].0.send(cycle).unwrap();
        //     }
        //
        //     // wait for all cores to finish
        //     for core_idx in 0..num_cores {
        //         core_done[core_idx].1.recv().unwrap();
        //     }
        //
        //     // wait for serial thread
        //     serial_done_rx.recv().unwrap();
        //
        //     cycle += run_ahead as u64;
        // }

        while (self.commands_left() || self.kernels_left()) && !self.reached_limit(cycle) {
            self.process_commands(cycle);
            self.launch_kernels(cycle);

            let mut finished_kernel = None;
            loop {
                if self.reached_limit(cycle) || !self.active() {
                    break;
                }

                // start all cores
                for core_idx in 0..num_chunks {
                    start_core[core_idx].0.send(cycle).unwrap();
                }

                for i in 0..run_ahead {
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
                }

                // wait for all cores to finish
                for core_idx in 0..num_chunks {
                    core_done[core_idx].1.recv().unwrap();
                }

                // issue new blocks
                let start = Instant::now();
                self.issue_block_to_core();
                #[cfg(feature = "stats")]
                {
                    TIMINGS
                        .lock()
                        .entry("serial::issue_block_to_core")
                        .or_default()
                        .add(start.elapsed());
                }

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

        let start = Instant::now();
        for cluster in &self.clusters {
            cluster.write().interconn_cycle(cycle);
        }
        #[cfg(feature = "stats")]
        {
            TIMINGS
                .lock()
                .entry("serial::interconn_cycle")
                .or_default()
                .add(start.elapsed());
        }

        let start = Instant::now();
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
        #[cfg(feature = "stats")]
        {
            TIMINGS
                .lock()
                .entry("serial::subs")
                .or_default()
                .add(start.elapsed());
        }

        let start = Instant::now();
        for (_i, unit) in self.mem_partition_units.iter().enumerate() {
            unit.try_write().simple_dram_cycle();
        }
        #[cfg(feature = "stats")]
        {
            TIMINGS
                .lock()
                .entry("serial::dram")
                .or_default()
                .add(start.elapsed());
        }

        let start = Instant::now();
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
        #[cfg(feature = "stats")]
        {
            TIMINGS
                .lock()
                .entry("serial::l2")
                .or_default()
                .add(start.elapsed());
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
