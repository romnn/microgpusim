#[derive(Clone, Debug, Default)]
#[repr(transparent)]
pub struct Cycle(Arc<CachePadded<atomic::AtomicU64>>);

impl Cycle {
    #[must_use]
    pub fn new(cycle: u64) -> Self {
        Self(Arc::new(CachePadded::new(atomic::AtomicU64::new(cycle))))
    }

    pub fn set(&self, cycle: u64) {
        use std::sync::atomic::Ordering;
        self.0.store(cycle, Ordering::SeqCst);
    }

    #[must_use]
    pub fn get(&self) -> u64 {
        use std::sync::atomic::Ordering;
        self.0.load(Ordering::SeqCst)
    }
}

fn cycle() {
    self.clusters
        .par_iter()
        .for_each(|cluster| cluster.try_write().interconn_cycle(cycle));

    // pop from memory controller to interconnect
    // this can be parallelized, but its not worth it
    // THIS MESSES UP EVERYTHING

    self.mem_sub_partitions
        .par_iter()
        .enumerate()
        .for_each(|(i, mem_sub)| {
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
                    // temp_requests
                    // .push((device, cluster_id, packet, response_packet_size));
                    // self.partition_replies_in_parallel += 1;
                } else {
                    // self.gpu_stall_icnt2sh += 1;
                }
            }
        });

    // we would need to sort by device here, which directly depends on mem sub
    // temp_requests
    // for (device, cluster_id, packet, response_packet_size) in temp_requests {
    //     self.interconn
    //         .push(device, cluster_id, packet, response_packet_size);
    // }

    // this pushes into sub.dram_to_l2_queue and messes up the order
    // also, we race for checking if dram to l2 queue is full, for which ordering does not
    // help
    self.mem_partition_units
        .par_iter()
        .for_each(|partition| partition.try_write().simple_dram_cycle(cycle));

    self.mem_sub_partitions
        .par_iter()
        .enumerate()
        .for_each(|(i, mem_sub)| {
            let mut mem_sub = mem_sub.try_lock();
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

                    mem_sub.push(packet.data, cycle);
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
        });

    // dbg!(self.mem_sub_partitions.len());
    // self.mem_sub_partitions
    //     .par_iter_mut()
    //     .for_each(|mem_sub| mem_sub.try_lock().cache_cycle(cycle));

    // let cores: Vec<_> = self
    //     .clusters
    //     .iter()
    //     .filter(|cluster| {
    //         let cores_completed = cluster.try_read().not_completed() == 0;
    //         !cores_completed || !kernels_completed
    //     })
    //     .flat_map(|cluster| {
    //         let cluster = cluster.try_read();
    //         cluster.cores.clone()
    //     })
    //     .collect();

    // cores.par_iter().for_each(|core| {
    //     crate::timeit!(core.write().cycle(cycle));
    // });
    let mut active_clusters = Vec::new();
    rayon::scope(|core_scope| {
        for cluster_arc in &self.clusters {
            let cluster = cluster_arc.try_read();
            if cluster.not_completed() == 0 && kernels_completed {
                continue;
            }
            active_clusters.push(cluster_arc);
            for core in cluster.cores.iter().cloned() {
                let core: Arc<RwLock<Core<_>>> = core;
                core_scope.spawn(move |_| core.write().cycle(cycle));
            }
        }
    });

    // for cluster in &mut self.clusters {
    for cluster in &active_clusters {
        // let cores_completed = cluster.not_completed() == 0;
        // let kernels_completed = self
        //     .running_kernels
        //     .iter()
        //     .filter_map(std::option::Option::as_ref)
        //     .all(|k| k.no_more_blocks_to_run());
        // if cores_completed && kernels_completed {
        //     continue;
        // }
        // dbg!(&executed_cluster_ids);
        // if !executed_cluster_ids.contains(&cluster.cluster_id) {
        //     continue;
        // }

        let cluster = cluster.try_read();
        // check if cluster was updated

        let mut core_sim_order = cluster.core_sim_order.try_lock();
        for core_id in &*core_sim_order {
            let core = cluster.cores[*core_id].try_read();
            let mut port = core.mem_port.try_lock();
            // let mut port = &mut core.mem_port;
            for ic::Packet {
                data: (dest, fetch, size),
                time: _,
            } in port.buffer.drain(..)
            {
                self.interconn.push(
                    core.cluster_id,
                    dest,
                    ic::Packet {
                        data: fetch,
                        time: cycle,
                    },
                    size,
                );
            }
        }

        if let config::SchedulingOrder::RoundRobin = self.config.simt_core_sim_order {
            core_sim_order.rotate_left(1);
        }
    }

    // for cluster in &mut self.clusters {
    //     let cluster = cluster.try_read();
    //     let cores_completed = cluster.not_completed() == 0;
    //     let kernels_completed = self
    //         .running_kernels
    //         .try_read()
    //         .iter()
    //         .filter_map(std::option::Option::as_ref)
    //         .all(|k| k.no_more_blocks_to_run());
    //
    //     if !(cores_completed && kernels_completed) {
    //         let mut core_sim_order = cluster.core_sim_order.try_lock();
    //         for core_id in core_sim_order.iter() {
    //             let core = cluster.cores[*core_id].try_read();
    //             let mut port = core.mem_port.lock();
    //             for ic::Packet {
    //                 data: (dest, fetch, size),
    //                 time,
    //             } in port.buffer.drain(..)
    //             {
    //                 self.interconn.push(
    //                     core.cluster_id,
    //                     dest,
    //                     ic::Packet { data: fetch, time },
    //                     size,
    //                 );
    //             }
    //         }
    //         if let config::SchedulingOrder::RoundRobin = self.config.simt_core_sim_order {
    //             core_sim_order.rotate_left(1);
    //         }
    //     }
    // }
}
