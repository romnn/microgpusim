#![allow(warnings, clippy::all)]

use crate::kernel_manager::SelectKernel;
use crate::sync::{Arc, Mutex, RwLock};
use crate::{cluster::Cluster, mem_partition_unit::MemoryPartitionUnit};
use crate::{config, core, ic, kernel::Kernel, mem_fetch, mem_sub_partition, Simulator};
use color_eyre::eyre;
use ndarray::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

struct ClusterCycle<'a, I, MC> {
    clusters: &'a mut [Cluster<I, MC>],
}

impl<'a, I, MC> ClusterCycle<'a, I, MC>
where
    I: ic::Interconnect<ic::Packet<mem_fetch::MemFetch>>,
    MC: crate::mcu::MemoryController,
{
    pub fn cycle(&mut self, cycle: u64) {
        for cluster in self.clusters.iter_mut() {
            // Receive memory responses addressed to each cluster and forward to cores
            // perf: blocks on the lock of CORE_INSTR_FETCH_RESPONSE_QUEUE
            // perf: blocks on the lock of CORE_LOAD_STORE_RESPONSE_QUEUE
            // perf: pops from INTERCONN (mem->cluster)
            crate::timeit!("serial::interconn", cluster.interconn_cycle(cycle));
        }
    }
}

struct SerialCycle<'a, I, MC> {
    clusters: &'a mut [Cluster<I, MC>],
    mem_partition_units: &'a mut [MemoryPartitionUnit<MC>],
    interconn: &'a I,
    config: &'a config::GPU,
}

impl<'a, I, MC> SerialCycle<'a, I, MC>
where
    I: ic::Interconnect<ic::Packet<mem_fetch::MemFetch>>,
    MC: crate::mcu::MemoryController,
{
    pub fn cycle(&mut self, cycle: u64) {
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
}

struct MemCycle<'a, I, MC> {
    mem_partition_units: &'a mut [MemoryPartitionUnit<MC>],
    interconn: &'a I,
    config: &'a config::GPU,
}

impl<'a, I, MC> MemCycle<'a, I, MC>
where
    I: ic::Interconnect<ic::Packet<mem_fetch::MemFetch>>,
    MC: crate::mcu::MemoryController,
{
    pub fn cycle(&mut self, cycle: u64) {
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
                        // let mut core = core.try_write();
                        let mut core = core.try_lock();
                        for i in 0..run_ahead {
                            crate::timeit!("core::cycle", core.cycle(cycle + i));

                            // do not enforce ordering of interconnect requests and round robin
                            // core simualation ordering
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

        // self.debug_completed_blocks();

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

                let interconn: &I = &*self.interconn;
                let kernel_manager = &self.kernel_manager;

                let serial_cycle = SerialCycle {
                    clusters: &mut self.clusters,
                    mem_partition_units: &mut self.mem_partition_units,
                    interconn,
                    config: &self.config,
                };
                let serial_cycle = Mutex::new(serial_cycle);
                let serial_cycle_ref = &serial_cycle;

                let fanout = 1;

                rayon::scope_fifo(|wave| {
                    for i in 0..run_ahead {
                        for core in cores.iter() {
                            wave.spawn_fifo(move |_| {
                                for f in 0..fanout {
                                    let mut core = core.lock();
                                    let cluster_id = core.cluster_id;

                                    crate::timeit!("core::cycle", core.cycle(cycle + i + f));

                                    // do not enforce ordering of interconnect
                                    // requests or round robin core simualation
                                    // ordering
                                    {
                                        for ic::Packet {
                                            fetch: (dest, fetch, size),
                                            time,
                                        } in core.mem_port.buffer.drain(..)
                                        {
                                            assert_eq!(time, cycle + i + f);
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
                                        core.maybe_issue_block(kernel_manager, cycle + i + f)
                                    );

                                    if core.global_core_id == 0 {
                                        drop(core);
                                        let mut lock = serial_cycle_ref.lock();
                                        crate::timeit!("serial::cycle", lock.cycle(cycle + i + f));
                                        drop(lock);
                                    }
                                }
                            });
                        }
                    }
                });

                // end of parallel section

                self.kernel_manager
                    .decrement_launch_latency(run_ahead * fanout);

                cycle += run_ahead * fanout;

                self.flush_caches(cycle);

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

        // self.debug_completed_blocks();

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

                if self.reached_limit(cycle) {
                    break;
                }

                rayon::scope(|wave| {
                    wave.spawn(|_| {
                        for i in 0..run_ahead {
                            let start = Instant::now();

                            for cluster in self.clusters.iter_mut() {
                                // Receive memory responses addressed to each cluster and forward to cores
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

                    for core in cores.iter() {
                        wave.spawn(|_| {
                            let mut core = core.try_lock();
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
                            }
                        });
                    }
                });

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

        // self.debug_completed_blocks();

        self.stats.no_kernel.sim.cycles = cycle;
        log::info!("exit after {cycle} cycles");

        Ok(())
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
                        let mut core = core.try_lock();
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
                        let mut core = core.try_lock();
                        for i in 0..run_ahead {
                            crate::timeit!("core::cycle", core.cycle(cycle + i));
                        }
                    });
                }
            }
        });
        Ok(())
    }

    #[tracing::instrument]
    pub fn serial_cycle(&mut self, cycle: u64) {
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
        }

        for partition in self.mem_partition_units.iter_mut() {
            for mem_sub in partition.sub_partitions.iter_mut() {
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
