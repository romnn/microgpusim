fn write_miss_write_allocate_fetch_on_write(
    &mut self,
    addr: address,
    cache_index: Option<usize>,
    fetch: mem_fetch::MemFetch,
    time: u64,
    events: &mut Vec<cache::Event>,
    probe_status: cache::RequestStatus,
) -> cache::RequestStatus {
    // let super::base::Base { ref cache_config, ref mut tag_array, .. } = self.inner;
    todo!("write_miss_write_allocate_fetch_on_write");
    let super::base::Base {
        ref cache_config, ..
    } = self.inner;
    let block_addr = cache_config.block_addr(addr);
    let mshr_addr = cache_config.mshr_addr(fetch.addr());

    if fetch.access_byte_mask().count_ones() == cache_config.atom_size() as usize {
        // if the request writes to the whole cache line/sector,
        // then write and set cache line modified.
        //
        // no need to send read request to memory or reserve mshr
        if self.inner.miss_queue_full() {
            let stats = self.inner.statslock();
            stats.inc(
                *fetch.access_kind(),
                cache::AccessStat::ReservationFailure(cache::ReservationFailure::MISS_QUEUE_FULL),
                1,
            );
            // cannot handle request this cycle
            return cache::RequestStatus::RESERVATION_FAIL;
        }

        // bool wb = false;
        // evicted_block_info evicted;
        let tag_array::AccessStatus {
            status,
            index,
            writeback,
            evicted,
            ..
        } = self.inner.tag_array.access(block_addr, &fetch, time);
        // , cache_index);
        // , wb, evicted, mf);
        debug_assert_ne!(status, cache::RequestStatus::HIT);
        let block = self.inner.tag_array.get_block_mut(index.unwrap());
        let was_modified_before = block.is_modified();
        block.set_status(cache_block::Status::MODIFIED, fetch.access_sector_mask());
        block.set_byte_mask(fetch.access_byte_mask());
        if status == cache::RequestStatus::HIT_RESERVED {
            block.set_ignore_on_fill(true, fetch.access_sector_mask());
        }
        if !was_modified_before {
            self.inner.tag_array.num_dirty += 1;
            // self.tag_array.inc_dirty();
        }

        if (status != cache::RequestStatus::RESERVATION_FAIL) {
            // If evicted block is modified and not a write-through
            // (already modified lower level)

            if writeback && cache_config.write_policy != config::CacheWritePolicy::WRITE_THROUGH {
                // let writeback_fetch = mem_fetch::MemFetch::new(
                //     fetch.instr,
                //     access,
                //     &*self.config,
                //     if wr {
                //         super::WRITE_PACKET_SIZE
                //     } else {
                //         super::READ_PACKET_SIZE
                //     }
                //     .into(),
                //     0,
                //     0,
                //     0,
                // );

                //     evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
                //     evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
                //     true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
                // NULL);

                // the evicted block may have wrong chip id when
                // advanced L2 hashing  is used,
                // so set the right chip address from the original mf
                // writeback_fetch.set_chip(mf->get_tlx_addr().chip);
                // writeback_fetch.set_parition(mf->get_tlx_addr().sub_partition);
                // self.send_write_request(wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                // time, events);
            }
            todo!("write_miss_write_allocate_fetch_on_write");
            return cache::RequestStatus::MISS;
        }
        return cache::RequestStatus::RESERVATION_FAIL;
    } else {
        todo!("write_miss_write_allocate_fetch_on_write");
        return cache::RequestStatus::RESERVATION_FAIL;
    }
}

fn write_miss_write_allocate_lazy_fetch_on_read(
    &mut self,
    addr: address,
    cache_index: Option<usize>,
    fetch: mem_fetch::MemFetch,
    time: u64,
    events: &mut Vec<cache::Event>,
    probe_status: cache::RequestStatus,
) -> cache::RequestStatus {
    todo!("write_miss_write_allocate_lazy_fetch_on_read");
    cache::RequestStatus::MISS
}

#[cfg(test)]
mod tests {
    use crate::{interconn as ic, mem_fetch};

    #[derive(Debug)]
    struct MockFetchInterconn {}

    impl ic::MemFetchInterface for MockFetchInterconn {
        fn full(&self, _size: u32, _write: bool) -> bool {
            false
        }
        fn push(&self, _fetch: mem_fetch::MemFetch, _time: u64) {}
    }

    fn concat<T>(
        a: impl IntoIterator<Item = T>,
        b: impl IntoIterator<Item = T>,
    ) -> impl Iterator<Item = T> {
        a.into_iter().chain(b.into_iter())
    }

    #[ignore = "todo"]
    #[test]
    fn test_ref_data_l1() {
        // let _control_size = 0;
        // let _warp_id = 0;
        // let _core_id = 0;
        // let _cluster_id = 0;
        // let type_id = bindings::cache_access_logger_types::NORMALS as i32;

        // let l1 = bindings::l1_cache::new(0, interconn, cache_config);
        // let cache_config = bindings::cache_config::new();

        // let mut cache_config = bridge::cache_config::new_cache_config();
        // dbg!(&cache_config.pin_mut().is_streaming());

        // let params = bindings::cache_config_params { disabled: false };
        // let mut cache_config = bridge::cache_config::new_cache_config(params);
        // dbg!(&cache_config.pin_mut().is_streaming());

        // let tag_array = bindings::tag_array::new(cache_config, core_id, type_id);
        // let fetch = bindings::mem_fetch_t::new(
        //     instr,
        //     access,
        //     &config,
        //     control_size,
        //     warp_id,
        //     core_id,
        //     cluster_id,
        // );
        // let status = l1.access(0x00000000, &fetch, vec![]);
        // dbg!(&status);
        // let status = l1.access(0x00000000, &fetch, vec![]);
        // dbg!(&status);
    }

    #[ignore = "todo"]
    #[test]
    fn test_data_l1_full_trace() {
        // let core_id = 0;
        // let cluster_id = 0;
        //
        // let stats = Arc::new(Mutex::new(stats::Cache::default()));
        // let config = Arc::new(config::GPU::default());
        // let cache_config = config.data_cache_l1.clone().unwrap();
        // let interconn = Arc::new(MockFetchInterconn {});
        // let cycle = Cycle::new(0);
        // let _l1 = Data::new(
        //     "l1-data".to_string(),
        //     core_id,
        //     cluster_id,
        //     cycle,
        //     interconn,
        //     stats.clone(),
        //     config,
        //     Arc::clone(&cache_config.inner),
        //     mem_fetch::AccessKind::L1_WR_ALLOC_R,
        //     mem_fetch::AccessKind::L1_WRBK_ACC,
        // );
        //
        // let trace_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        //     .join("test-apps/vectoradd/traces/vectoradd-100-32-trace/");
        // dbg!(&trace_dir);
        // let commands: Vec<Command> =
        //     parse_commands(&trace_dir.join("commands.json")).expect("parse trace commands");
        //
        // dbg!(&commands);
        // let mut kernels: VecDeque<_> = VecDeque::new();
        // for cmd in commands {
        //     match cmd {
        //         Command::MemAlloc { .. } | Command::MemcpyHtoD { .. } => {}
        //         Command::KernelLaunch(launch) => {
        //             let kernel = Kernel::from_trace(launch.clone(), &trace_dir);
        //             kernels.push_back(kernel);
        //         }
        //     }
        // }

        // for kernel in &mut kernels {
        //     let mut block_iter = kernel.next_block_iterlock();
        //     while let Some(block) = block_iter.next() {
        //         dbg!(&block);
        //         let mut trace_pos = kernel.trace_poswrite();
        //         // let mut lock = kernel.trace_iterwrite();
        //         // let trace_iter = lock.take_while_ref(|entry| entry.block_id == block);
        //         while *trace_pos < kernel.trace.len() {
        //             // for trace in trace_iter {
        //             let trace = &kernel.trace[*trace_pos];
        //             if trace.block_id > block.into() {
        //                 break;
        //             }
        //
        //             *trace_pos += 1;
        //
        //             // dbg!(&trace);
        //             let warp_id = trace.warp_id_in_block as usize;
        //             if warp_id != 0 {
        //                 continue;
        //             }
        //             let mut instr =
        //                 instruction::WarpInstruction::from_trace(&kernel, trace.clone());
        //
        //             let mut accesses = instr
        //                 .generate_mem_accesses(&*config)
        //                 .expect("generated acceseses");
        //             // dbg!(&accesses);
        //             for access in &accesses {
        //                 log::debug!(
        //                     "block {} warp {}: {} access {}",
        //                     &block,
        //                     &warp_id,
        //                     if access.is_write { "store" } else { "load" },
        //                     &access.addr
        //                 );
        //             }
        //             assert_eq!(accesses.len(), 1);
        //
        //             let access = accesses.remove(0);
        //             // let access = mem_fetch::MemAccess::from_instr(&instr).unwrap();
        //             let fetch = mem_fetch::MemFetch::new(
        //                 Some(instr),
        //                 access,
        //                 &config,
        //                 control_size,
        //                 warp_id,
        //                 core_id,
        //                 cluster_id,
        //             );
        //             let mut events = Vec::new();
        //             let status = l1.access(fetch.access.addr, fetch, &mut events);
        //             // let status = l1.access(0x00000000, fetch.clone(), None);
        //             dbg!(&status);
        //         }
        //     }
        //     // while let Some(trace_instr) = kernel.trace_iterwrite().next() {
        //     //     // dbg!(&instr);
        //     //     let mut instr = instruction::WarpInstruction::from_trace(&kernel, trace_instr);
        //     //     let mut accesses = instr
        //     //         .generate_mem_accesses(&*config)
        //     //         .expect("generated acceseses");
        //     //     // dbg!(&accesses);
        //     //     assert_eq!(accesses.len(), 1);
        //     //     for access in &accesses {
        //     //         // log::debug!(
        //     //         //     "block {} warp {}: access {}",
        //     //         //     &access.block, &access.warp_id, &access.addr
        //     //         // );
        //     //         // log::debug!("{}", &access);
        //     //     }
        //     //     // continue;
        //     //
        //     //     let access = accesses.remove(0);
        //     //     // let access = mem_fetch::MemAccess::from_instr(&instr).unwrap();
        //     //     let fetch = mem_fetch::MemFetch::new(
        //     //         instr,
        //     //         access,
        //     //         &config,
        //     //         control_size,
        //     //         warp_id,
        //     //         core_id,
        //     //         cluster_id,
        //     //     );
        //     //     let status = l1.access(fetch.access.addr, fetch, None);
        //     //     // let status = l1.access(0x00000000, fetch.clone(), None);
        //     //     dbg!(&status);
        //     // }
        // }

        // let mut stats = STATSlock();
        // dbg!(&statslock());

        // let mut warps: Vec<sched::SchedulerWarp> = Vec::new();
        // for kernel in kernels {
        //     loop {
        //         assert!(!warps.is_empty());
        //         kernel.next_threadblock_traces(&mut warps);
        //         dbg!(&warps);
        //         break;
        //     }
        // }
    }

    #[ignore = "todo"]
    #[test]
    fn test_data_l1_single_access() {
        // let control_size = 0;
        // let warp_id = 0;
        // let core_id = 0;
        // let cluster_id = 0;
        //
        // let stats = Arc::new(Mutex::new(stats::Cache::default()));
        // let config = Arc::new(config::GPU::default());
        // let cache_config = config.data_cache_l1.clone().unwrap();
        // let interconn = Arc::new(MockFetchInterconn {});
        // let cycle = Cycle::new(0);
        //
        // let mut l1 = Data::new(
        //     "l1-data".to_string(),
        //     core_id,
        //     cluster_id,
        //     cycle,
        //     interconn,
        //     stats.clone(),
        //     config.clone(),
        //     Arc::clone(&cache_config.inner),
        //     mem_fetch::AccessKind::L1_WR_ALLOC_R,
        //     mem_fetch::AccessKind::L1_WRBK_ACC,
        // );
        //
        // let trace_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        //     .join("test-apps/vectoradd/traces/vectoradd-100-32-trace/");
        //
        // let launch = trace_model::KernelLaunch {
        //     name: "void vecAdd<float>(float*, float*, float*, int)".into(),
        //     trace_file: "kernel-0-trace".into(),
        //     id: 0,
        //     grid: trace_model::Dim { x: 1, y: 1, z: 1 },
        //     block: trace_model::Dim {
        //         x: 1024,
        //         y: 1,
        //         z: 1,
        //     },
        //     shared_mem_bytes: 0,
        //     num_registers: 8,
        //     binary_version: 61,
        //     stream_id: 0,
        //     shared_mem_base_addr: 140_663_786_045_440,
        //     local_mem_base_addr: 140_663_752_491_008,
        //     nvbit_version: "1.5.5".to_string(),
        // };
        // let kernel = Kernel::from_trace(launch, trace_dir);
        //
        // let trace_instr = trace_model::MemAccessTraceEntry {
        //     cuda_ctx: 0,
        //     sm_id: 0,
        //     kernel_id: 0,
        //     block_id: trace_model::Dim::ZERO,
        //     warp_size: 32,
        //     warp_id_in_sm: 3,
        //     warp_id_in_block: 3,
        //     line_num: 0,
        //     instr_data_width: 4,
        //     instr_opcode: "LDG.E.CG".to_string(),
        //     instr_offset: 176,
        //     instr_idx: 16,
        //     instr_predicate: nvbit_model::Predicate {
        //         num: 0,
        //         is_neg: false,
        //         is_uniform: false,
        //     },
        //     instr_mem_space: nvbit_model::MemorySpace::Global,
        //     instr_is_mem: true,
        //     instr_is_load: true,
        //     instr_is_store: false,
        //     instr_is_extended: true,
        //     active_mask: 15,
        //     // todo: use real values here
        //     dest_regs: [0; 1],
        //     num_dest_regs: 0,
        //     src_regs: [0; 5],
        //     num_src_regs: 0,
        //     addrs: concat(
        //         [
        //             140_663_086_646_144,
        //             140_663_086_646_148,
        //             140_663_086_646_152,
        //             140_663_086_646_156,
        //         ],
        //         [0; 32 - 4],
        //     )
        //     .collect::<Vec<_>>()
        //     .try_into()
        //     .unwrap(),
        // };
        // let mut instr = instruction::WarpInstruction::from_trace(&kernel, &trace_instr);
        // let mut accesses = instr
        //     .generate_mem_accesses(&config)
        //     .expect("generated acceseses");
        // assert_eq!(accesses.len(), 1);
        //
        // let access = accesses.remove(0);
        // // let access = mem_fetch::MemAccess::from_instr(&instr).unwrap();
        // let fetch = mem_fetch::MemFetch::new(
        //     Some(instr),
        //     access,
        //     &config,
        //     control_size,
        //     warp_id,
        //     core_id,
        //     cluster_id,
        // );
        // // let status = l1.access(0x00000000, fetch.clone(), None);
        // let time = 0;
        // let mut events = Vec::new();
        // let status = l1.access(fetch.addr(), fetch.clone(), &mut events, time);
        // dbg!(&status);
        // let mut events = Vec::new();
        // let status = l1.access(fetch.addr(), fetch, &mut events, time);
        // dbg!(&status);
        //
        // // let mut stats = STATSlock();
        // dbg!(&statslock());
    }
}
