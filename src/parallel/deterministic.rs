use crate::sync::{Arc, RwLock};
use crate::{config, engine::cycle::Component, ic, mem_fetch, MockSimulator};
use color_eyre::eyre;

impl<I, MC> MockSimulator<I, MC>
where
    I: ic::Interconnect<ic::Packet<mem_fetch::MemFetch>> + 'static,
    MC: crate::mcu::MemoryController,
{
    pub fn run_to_completion_parallel_deterministic(&mut self) -> eyre::Result<()> {
        crate::TIMINGS.lock().clear();
        let mut cycle: u64 = 0;

        let num_threads = super::get_num_threads()?
            .or(self.config.simulation_threads)
            .unwrap_or_else(num_cpus::get_physical);

        super::rayon_pool(num_threads)?.install(|| {
            eprintln!("parallel (deterministic)");
            eprintln!(
                "\t => launching {num_threads} worker threads for {} cores",
                self.config.total_cores()
            );
            eprintln!();

            let cores: Vec<Vec<Arc<_>>> = self
                .clusters
                .iter()
                .map(|cluster| cluster.cores.clone())
                .collect();

            let mut active_clusters = utils::box_slice![false; self.clusters.len()];

            let log_every = 10_000;
            let mut last_time = std::time::Instant::now();

            while (self.commands_left() || self.kernels_left()) && !self.reached_limit(cycle) {
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

                    crate::timeit!("serial::cycle", self.serial_cycle(cycle));

                    // run cores in any order
                    rayon::scope(|core_scope| {
                        let kernels_completed = self
                            .running_kernels
                            .try_read()
                            .iter()
                            .filter_map(Option::as_ref)
                            .all(|(_, k)| k.no_more_blocks_to_run());

                        for (cluster_id, cluster) in self.clusters.iter().enumerate() {
                            let cores_completed = cluster.not_completed() == 0;
                            let cluster_active = !(cores_completed && kernels_completed);
                            active_clusters[cluster_id] = cluster_active;

                            if !cluster_active {
                                continue;
                            }

                            for core in cores[cluster_id].iter().cloned() {
                                let core: Arc<RwLock<_>> = core;
                                core_scope.spawn(move |_| {
                                    crate::timeit!("core::cycle", core.write().cycle(cycle));
                                });
                            }
                        }
                    });

                    // collect the core packets pushed to the interconn
                    for (cluster_id, active) in active_clusters.iter().enumerate() {
                        if !active {
                            continue;
                        }
                        let cluster = &self.clusters[cluster_id];
                        let mut core_sim_order = cluster.core_sim_order.try_lock();
                        for core_id in &*core_sim_order {
                            let core = cluster.cores[*core_id].try_read();
                            let mut port = core.mem_port.lock();
                            for ic::Packet {
                                data: (dest, fetch, size),
                                time,
                            } in port.buffer.drain(..)
                            {
                                self.interconn.push(
                                    core.cluster_id,
                                    dest,
                                    ic::Packet { data: fetch, time },
                                    size,
                                );
                            }
                        }
                        if let config::SchedulingOrder::RoundRobin = self.config.simt_core_sim_order
                        {
                            core_sim_order.rotate_left(1);
                        }
                    }

                    self.issue_block_to_core(cycle);

                    let mut all_threads_complete = true;
                    if self.config.flush_l1_cache {
                        for cluster in &mut self.clusters {
                            if cluster.not_completed() == 0 {
                                cluster.cache_invalidate();
                            } else {
                                all_threads_complete = false;
                            }
                        }
                    }

                    if self.config.flush_l2_cache {
                        if !self.config.flush_l1_cache {
                            for cluster in &mut self.clusters {
                                if cluster.not_completed() > 0 {
                                    all_threads_complete = false;
                                    break;
                                }
                            }
                        }

                        if let Some(l2_config) = &self.config.data_cache_l2 {
                            if all_threads_complete {
                                log::debug!("flushed L2 caches...");
                                if l2_config.inner.total_lines() > 0 {
                                    for (i, mem_sub) in
                                        self.mem_sub_partitions.iter_mut().enumerate()
                                    {
                                        let mut mem_sub = mem_sub.try_lock();
                                        let num_dirty_lines_flushed = mem_sub.flush_l2();
                                        log::debug!(
                                            "dirty lines flushed from L2 {} is {:?}",
                                            i,
                                            num_dirty_lines_flushed
                                        );
                                    }
                                }
                            }
                        }
                    }

                    cycle += 1;
                    // self.set_cycle(cycle);

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
            Ok::<_, eyre::Report>(())
        })?;

        self.stats.no_kernel.sim.cycles = cycle;
        // self.stats.lock().no_kernel.sim.cycles = cycle;
        if let Some(log_after_cycle) = self.log_after_cycle {
            if log_after_cycle >= cycle {
                eprintln!("WARNING: log after {log_after_cycle} cycles but simulation ended after {cycle} cycles");
            }
        }

        log::info!("exit after {cycle} cycles");
        Ok(())
    }
}
