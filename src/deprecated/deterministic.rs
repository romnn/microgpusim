// let cores: Vec<_> = self
//     .clusters
//     .iter()
//     .flat_map(|cluster| cluster.cores.iter().map(|core| (cluster, core))
//     .cloned()
//     .collect();
// let num_cores = cores.len();

// let num_cores = cores.len();
//
// let (mut start_core_tx, mut start_core_rx) = (Vec::new(), Vec::new());
// let (mut core_done_tx, mut core_done_rx) = (Vec::new(), Vec::new());
// for _ in &cores {
//     let (tx, rx) = crossbeam::channel::bounded(1);
//     start_core_tx.push(tx);
//     start_core_rx.push(rx);
//
//     let (tx, rx) = crossbeam::channel::bounded(1);
//     core_done_tx.push(tx);
//     core_done_rx.push(rx);
// }

// spawn worker threads for core cycles
// let core_worker_handles: Vec<_> = cores
//     .into_iter()
//     .enumerate()
//     .map(|(cluster_idx, cluster)| {
//         let start_core_rx = start_core_rx[cluster_idx].clone();
//         let core_done_tx = core_done_tx[cluster_idx].clone();
//         let running_kernels = self.running_kernels.clone();
//         std::thread::spawn(move || loop {
//             if start_core_rx.recv().is_err() {
//                 // println!("cluster {} exited", cluster.try_read().cluster_id);
//                 break;
//             }
//
//             let kernels_completed = running_kernels
//                 .try_read()
//                 .iter()
//                 .filter_map(std::option::Option::as_ref)
//                 .all(|k| k.no_more_blocks_to_run());
//
//             {
//                 let cluster = cluster.try_read();
//                 let cores_completed = cluster.not_completed() == 0;
//                 if !(cores_completed && kernels_completed) {
//                     for core in &cluster.cores {
//                         let mut core = core.write();
//                         // println!("start core {:?} ({} clusters)", core.id(), num_cores);
//                         core.cycle(cycle);
//                         // println!("done core {:?} ({} clusters)", core.id(), num_cores);
//                     }
//                 }
//             }
//
//             // {
//             //     let mut core = corewrite();
//             //     let core_completed = core.not_completed() == 0;
//             //     if !(core_completed && kernels_completed) {
//             //         core.cycle();
//             //     }
//             // }
//             core_done_tx.send(()).unwrap();
//         })
//     })
//     .collect();

// assert_eq!(core_worker_handles.len(), num_cores);
// // START LIB CYCLE
// for cluster in &mut self.clusters {
//     cluster.try_write().interconn_cycle(cycle);
// }
//
// for (i, mem_sub) in self.mem_sub_partitions.iter().enumerate() {
//     let mut mem_sub = mem_sub.try_lock();
//     if let Some(fetch) = mem_sub.top() {
//         let response_packet_size = if fetch.is_write() {
//             fetch.control_size()
//         } else {
//             fetch.size()
//         };
//         let device = self.config.mem_id_to_device_id(i);
//         if self.interconn.has_buffer(device, response_packet_size) {
//             let mut fetch = mem_sub.pop().unwrap();
//             let cluster_id = fetch.cluster_id;
//             fetch.set_status(mem_fetch::Status::IN_ICNT_TO_SHADER, 0);
//             // fetch.set_return_timestamp(gpu_sim_cycle + gpu_tot_sim_cycle);
//             // , gpu_sim_cycle + gpu_tot_sim_cycle);
//             // drop(fetch);
//             self.interconn.push(
//                 device,
//                 cluster_id,
//                 ic::Packet {
//                     data: fetch,
//                     time: cycle,
//                 },
//                 response_packet_size,
//             );
//             self.partition_replies_in_parallel += 1;
//         } else {
//             // self.gpu_stall_icnt2sh += 1;
//         }
//     }
// }
//
// for (_i, unit) in self.mem_partition_units.iter().enumerate() {
//     unit.try_write().simple_dram_cycle(cycle);
// }
//
// for (i, mem_sub) in self.mem_sub_partitions.iter_mut().enumerate() {
//     // let mut mem_sub = mem_sub.try_borrow_mut().unwrap();
//     let mut mem_sub = mem_sub.try_lock();
//     // move memory request from interconnect into memory partition
//     // (if not backed up)
//     //
//     // Note:This needs to be called in DRAM clock domain if there
//     // is no L2 cache in the system In the worst case, we may need
//     // to push SECTOR_CHUNCK_SIZE requests, so ensure you have enough
//     // buffer for them
//     let device = self.config.mem_id_to_device_id(i);
//
//     // same as full with parameter overload
//     if mem_sub
//         .interconn_to_l2_queue
//         .can_fit(mem_sub_partition::SECTOR_CHUNCK_SIZE as usize)
//     {
//         if let Some(packet) = self.interconn.pop(device) {
//             log::debug!(
//                 "got new fetch {} for mem sub partition {} ({})",
//                 packet.data,
//                 i,
//                 device
//             );
//
//             mem_sub.push(packet.data, cycle);
//             // mem_sub.push(packet.data, packet.time);
//             // self.parallel_mem_partition_reqs += 1;
//         }
//     } else {
//         log::debug!("SKIP sub partition {} ({}): DRAM full stall", i, device);
//         #[cfg(feature = "stats")]
//         {
//             self.stats.lock().stall_dram_full += 1;
//         }
//     }
//     // we borrow all of sub here, which is a problem for the cyclic reference in l2
//     // interface
//     mem_sub.cache_cycle(cycle);
// }
//
// start all cores
// for start_tx in &start_core_tx {
//     start_tx.send(()).unwrap();
// }
//
// // wait for all cores to finish
// for done_rx in &core_done_rx {
//     done_rx.recv().unwrap();
// }
//
// let running_kernels = self.running_kernels.clone();
//// let core_sim_order = cluster.core_sim_order.clone();
// let cores = cluster.cores;
//
// let mut core_sim_order = core_sim_order.try_lock();
// let mut port = core.mem_port.try_lock();
// for ic::Packet {
//     data: (dest, fetch, size),
//     time: _,
// } in port.buffer.drain(..)
// {
//     self.interconn.push(
//         core.cluster_id,
//         dest,
//         ic::Packet {
//             data: fetch,
//             time: cycle,
//         },
//         size,
//     );
// }
// if let config::SchedulingOrder::RoundRobin =
//     self.config.simt_core_sim_order
// {
//     core_sim_order.rotate_left(1);
// }

// for core in &cluster.cores {
//     let mut core = core.write();
//     core.cycle(cycle);
// }
// }
// let cores_completed = cluster.not_completed() == 0;
// let kernels_completed = self
//     .running_kernels
//     .try_read()
//     .iter()
//     .filter_map(std::option::Option::as_ref)
//     .all(|k| k.no_more_blocks_to_run());
//
// if !(cores_completed && kernels_completed) {

// TODO: need shutdown for this
// for h in core_worker_handles {
//     h.join().unwrap();
// }
