use crate::{config, ic, mem_fetch, Simulator};
use color_eyre::eyre;

impl<I, MC> Simulator<I, MC>
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

            let mut active_clusters = utils::box_slice![false; self.clusters.len()];

            let log_every = 10_000;
            let mut last_time = std::time::Instant::now();

            while (self.trace.commands_left() || self.kernels_left()) && !self.reached_limit(cycle)
            {
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

                    crate::timeit!("serial::cycle", self.serial_cycle(cycle));

                    let kernels_completed = self.kernel_manager.all_kernels_completed();

                    rayon::scope(|core_scope| {
                        for cluster in self.clusters.iter_mut() {
                            let cores_completed = cluster.num_active_threads() == 0;
                            let cluster_active = !(cores_completed && kernels_completed);
                            active_clusters[cluster.cluster_id] = cluster_active;

                            if !cluster_active {
                                continue;
                            }

                            for core in cluster.cores.iter_mut() {
                                core_scope.spawn(move |_| {
                                    // let mut core = core.try_write();
                                    let mut core = core.try_lock();
                                    crate::timeit!("core::cycle", core.cycle(cycle));
                                });
                            }
                        }
                    });

                    // collect the core packets pushed to the interconn
                    for (cluster_id, active) in active_clusters.iter().enumerate() {
                        if !active {
                            continue;
                        }
                        let cluster = &mut self.clusters[cluster_id];
                        let cluster_id = &cluster.cluster_id;

                        let mut core_sim_order = cluster.core_sim_order.try_lock();
                        for core_id in &*core_sim_order {
                            let core = &cluster.cores[*core_id];
                            let mut core = core.try_lock();
                            for ic::Packet {
                                fetch: (dest, fetch, size),
                                time,
                            } in core.mem_port.buffer.drain(..)
                            {
                                self.interconn.push(
                                    *cluster_id,
                                    dest,
                                    ic::Packet { fetch, time },
                                    size,
                                );
                            }
                        }
                        if let config::SchedulingOrder::RoundRobin = self.config.simt_core_sim_order
                        {
                            core_sim_order.rotate_left(1);
                        }
                    }

                    use crate::BlockIssue;
                    self.block_issuer
                        .issue_blocks_to_core_deterministic(&mut self.kernel_manager, cycle);
                    self.kernel_manager.decrement_launch_latency(1);

                    self.flush_caches(cycle);

                    cycle += 1;

                    if !self.active() {
                        finished_kernel = self.kernel_manager.get_finished_kernel();
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
                    self.trace.commands_left(),
                    self.kernels_left()
                );
            }
            Ok::<_, eyre::Report>(())
        })?;

        self.stats.no_kernel.sim.cycles = cycle;
        if let Some(log_after_cycle) = self.log_after_cycle {
            if log_after_cycle >= cycle {
                eprintln!("WARNING: log after {log_after_cycle} cycles but simulation ended after {cycle} cycles");
            }
        }

        log::info!("exit after {cycle} cycles");
        Ok(())
    }
}
