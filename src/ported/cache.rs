use super::{address, interconn as ic, mem_fetch, tag_array::TagArray};
use crate::config;
use std::sync::Arc;

#[derive(Debug)]
pub struct TextureL1 {
    id: usize,
    interconn: ic::Interconnect,
}

impl TextureL1 {
    pub fn new(id: usize, interconn: ic::Interconnect) -> Self {
        Self { id, interconn }
    }

    // pub fn new(name: String) -> Self {
    //     Self { name }
    // }
    pub fn cycle(&mut self) {}

    pub fn fill(&self, fetch: &mem_fetch::MemFetch) {}

    pub fn has_free_fill_port(&self) -> bool {
        false
    }
}

#[derive(Debug, Default)]
pub struct ConstL1 {}

impl ConstL1 {
    pub fn cycle(&mut self) {}

    pub fn fill(&self, fetch: &mem_fetch::MemFetch) {}

    pub fn has_free_fill_port(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum CacheBlockState {
    INVALID = 0,
    RESERVED,
    VALID,
    MODIFIED,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum CacheRequestStatus {
    HIT = 0,
    HIT_RESERVED,
    MISS,
    RESERVATION_FAIL,
    SECTOR_MISS,
    MSHR_HIT,
    NUM_CACHE_REQUEST_STATUS,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum CacheReservationFailure {
    /// all line are reserved
    LINE_ALLOC_FAIL = 0,
    /// MISS queue (i.e. interconnect or DRAM) is full
    MISS_QUEUE_FULL,
    MSHR_ENRTY_FAIL,
    MSHR_MERGE_ENRTY_FAIL,
    MSHR_RW_PENDING,
    NUM_CACHE_RESERVATION_FAIL_STATUS,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum CacheEventKind {
    WRITE_BACK_REQUEST_SENT,
    READ_REQUEST_SENT,
    WRITE_REQUEST_SENT,
    WRITE_ALLOCATE_SENT,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct CacheEvent {}

// #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
// pub enum WriteAllocatePolicy {
//     L1_WR_ALLOC_R,
// }
//
// #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
// pub enum WriteBackPolicy{
// }

/// First level data cache in Fermi.
///
/// The cache uses a write-evict (global) or write-back (local) policy
/// at the granularity of individual blocks.
/// (the policy used in fermi according to the CUDA manual)
#[derive(Debug)]
pub struct DataL1<I> {
    core_id: usize,

    // config: Arc<GPUConfig>,
    config: Arc<config::CacheConfig>,

    tag_array: TagArray<usize>,
    fetch_interconn: I,

    /// Specifies type of write allocate request (e.g., L1 or L2)
    write_alloc_type: mem_fetch::AccessKind,
    /// Specifies type of writeback request (e.g., L1 or L2)
    write_back_type: mem_fetch::AccessKind,
}

impl<I> DataL1<I> {
    pub fn new(core_id: usize, fetch_interconn: I, config: Arc<config::CacheConfig>) -> Self {
        let tag_array = TagArray::new(core_id, 0, config.clone());
        Self {
            core_id,
            fetch_interconn,
            config,
            tag_array,
            write_alloc_type: mem_fetch::AccessKind::L1_WR_ALLOC_R,
            write_back_type: mem_fetch::AccessKind::L1_WRBK_ACC,
        }
    }

    pub fn access(
        &self,
        addr: address,
        fetch: mem_fetch::MemFetch,
        events: Vec<CacheEvent>,
    ) -> CacheRequestStatus {
        // data_cache::access(addr, mf, time, events);
        // assert(mf->get_data_size() <= m_config.get_atom_sz());
        let is_write = fetch.is_write();
        let block_addr = self.config.block_addr(addr);

        let (cache_index, probe_status) = self.tag_array.probe(block_addr, &fetch, is_write, true);
        let access_status =
            self.process_tag_probe(is_write, probe_status, addr, cache_index, &fetch, &events);
        // m_stats.inc_stats(mf->get_access_type(),
        //                   m_stats.select_stats_status(probe_status, access_status));
        // m_stats.inc_stats_pw(mf->get_access_type(), m_stats.select_stats_status(
        //                                                 probe_status, access_status));
        access_status
    }

    // A general function that takes the result of a tag_array probe.
    //
    // It performs the correspding functions based on the
    // cache configuration.
    fn process_tag_probe(
        &self,
        is_write: bool,
        probe_status: CacheRequestStatus,
        addr: address,
        cache_index: Option<usize>,
        fetch: &mem_fetch::MemFetch,
        events: &[CacheEvent],
    ) -> CacheRequestStatus {
        // Each function pointer ( m_[rd/wr]_[hit/miss] ) is set in the
        // data_cache constructor to reflect the corresponding cache configuration
        // options. Function pointers were used to avoid many long conditional
        // branches resulting from many cache configuration options.
        let access_status = probe_status;
        if is_write {
            if probe_status == CacheRequestStatus::HIT {
                // access_status = (this->*m_wr_hit)(addr, cache_index, mf, time, events, probe_status);
            } else if probe_status != CacheRequestStatus::RESERVATION_FAIL
                || (probe_status == CacheRequestStatus::RESERVATION_FAIL
                    && self.config.write_allocate_policy
                        == config::CacheWriteAllocatePolicy::NO_WRITE_ALLOCATE)
            {
                // access_status = (this->*m_wr_miss)(addr, cache_index, mf, time, events, probe_status);
            } else {
                // the only reason for reservation fail here is LINE_ALLOC_FAIL
                // (i.e all lines are reserved)
                // m_stats.inc_fail_stats(mf->get_access_type(), LINE_ALLOC_FAIL);
            }
        } else {
            if probe_status == CacheRequestStatus::HIT {
                //     access_status =
                //         (this->*m_rd_hit)(addr, cache_index, mf, time, events, probe_status);
            } else if probe_status != CacheRequestStatus::RESERVATION_FAIL {
                //     access_status =
                //         (this->*m_rd_miss)(addr, cache_index, mf, time, events, probe_status);
            } else {
                //     // the only reason for reservation fail here is LINE_ALLOC_FAIL (i.e all
                //     // lines are reserved)
                //     m_stats.inc_fail_stats(mf->get_access_type(), LINE_ALLOC_FAIL);
            }
        }
        //
        // m_bandwidth_management.use_data_port(mf, access_status, events);
        // return access_status;

        CacheRequestStatus::MISS
    }

    // data_cache(name, config, core_id, type_id, memport, mfcreator, status,
    // , L1_WRBK_ACC, gpu) {}

    /// A general function that takes the result of a tag_array probe
    ///  and performs the correspding functions based on the cache configuration
    ///  The access fucntion calls this function
    // enum cache_request_status process_tag_probe(bool wr,
    //                                             enum cache_request_status status,
    //                                             new_addr_type addr,
    //                                             unsigned cache_index,
    //                                             mem_fetch *mf, unsigned time,
    //                                             std::list<cache_event> &events);

    // /// Sends write request to lower level memory (write or writeback)
    // void send_write_request(mem_fetch *mf, cache_event request, unsigned time,
    //                         std::list<cache_event> &events);
    // void update_m_readable(mem_fetch *mf, unsigned cache_index);
    //
    // // Member Function pointers - Set by configuration options
    // // to the functions below each grouping
    // /******* Write-hit configs *******/
    // enum cache_request_status (data_cache::*m_wr_hit)(
    //     new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    //     std::list<cache_event> &events, enum cache_request_status status);
    // /// Marks block as MODIFIED and updates block LRU
    // enum cache_request_status wr_hit_wb(
    //     new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    //     std::list<cache_event> &events,
    //     enum cache_request_status status);  // write-back
    // enum cache_request_status wr_hit_wt(
    //     new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    //     std::list<cache_event> &events,
    //     enum cache_request_status status);  // write-through
    //
    // /// Marks block as INVALID and sends write request to lower level memory
    // enum cache_request_status wr_hit_we(
    //     new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    //     std::list<cache_event> &events,
    //     enum cache_request_status status);  // write-evict
    // enum cache_request_status wr_hit_global_we_local_wb(
    //     new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    //     std::list<cache_event> &events, enum cache_request_status status);
    // // global write-evict, local write-back
    //
    // /******* Write-miss configs *******/
    // enum cache_request_status (data_cache::*m_wr_miss)(
    //     new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    //     std::list<cache_event> &events, enum cache_request_status status);
    // /// Sends read request, and possible write-back request,
    // //  to lower level memory for a write miss with write-allocate
    // enum cache_request_status wr_miss_wa_naive(
    //     new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    //     std::list<cache_event> &events,
    //     enum cache_request_status
    //         status);  // write-allocate-send-write-and-read-request
    // enum cache_request_status wr_miss_wa_fetch_on_write(
    //     new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    //     std::list<cache_event> &events,
    //     enum cache_request_status
    //         status);  // write-allocate with fetch-on-every-write
    // enum cache_request_status wr_miss_wa_lazy_fetch_on_read(
    //     new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    //     std::list<cache_event> &events,
    //     enum cache_request_status status);  // write-allocate with read-fetch-only
    // enum cache_request_status wr_miss_wa_write_validate(
    //     new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    //     std::list<cache_event> &events,
    //     enum cache_request_status
    //         status);  // write-allocate that writes with no read fetch
    // enum cache_request_status wr_miss_no_wa(
    //     new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    //     std::list<cache_event> &events,
    //     enum cache_request_status status);  // no write-allocate
    //
    // // Currently no separate functions for reads
    // /******* Read-hit configs *******/
    // enum cache_request_status (data_cache::*m_rd_hit)(
    //     new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    //     std::list<cache_event> &events, enum cache_request_status status);
    // enum cache_request_status rd_hit_base(new_addr_type addr,
    //                                       unsigned cache_index, mem_fetch *mf,
    //                                       unsigned time,
    //                                       std::list<cache_event> &events,
    //                                       enum cache_request_status status);
    //
    // /******* Read-miss configs *******/
    // enum cache_request_status (data_cache::*m_rd_miss)(
    //     new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    //     std::list<cache_event> &events, enum cache_request_status status);
    // enum cache_request_status rd_miss_base(new_addr_type addr,
    //                                        unsigned cache_index, mem_fetch *mf,
    //                                        unsigned time,
    //                                        std::list<cache_event> &events,
    //                                        enum cache_request_status status);

    // virtual void init(mem_fetch_allocator *mfcreator) {
    //     m_memfetch_creator = mfcreator;
    //
    //     // Set read hit function
    //     m_rd_hit = &data_cache::rd_hit_base;
    //
    //     // Set read miss function
    //     m_rd_miss = &data_cache::rd_miss_base;
    //
    //     // Set write hit function
    //     switch (m_config.m_write_policy) {
    //       // READ_ONLY is now a separate cache class, config is deprecated
    //       case READ_ONLY:
    //         assert(0 && "Error: Writable Data_cache set as READ_ONLY\n");
    //         break;
    //       case WRITE_BACK:
    //         m_wr_hit = &data_cache::wr_hit_wb;
    //         break;
    //       case WRITE_THROUGH:
    //         m_wr_hit = &data_cache::wr_hit_wt;
    //         break;
    //       case WRITE_EVICT:
    //         m_wr_hit = &data_cache::wr_hit_we;
    //         break;
    //       case LOCAL_WB_GLOBAL_WT:
    //         m_wr_hit = &data_cache::wr_hit_global_we_local_wb;
    //         break;
    //       default:
    //         assert(0 && "Error: Must set valid cache write policy\n");
    //         break;  // Need to set a write hit function
    //     }
    //
    //     // Set write miss function
    //     switch (m_config.m_write_alloc_policy) {
    //       case NO_WRITE_ALLOCATE:
    //         m_wr_miss = &data_cache::wr_miss_no_wa;
    //         break;
    //       case WRITE_ALLOCATE:
    //         m_wr_miss = &data_cache::wr_miss_wa_naive;
    //         break;
    //       case FETCH_ON_WRITE:
    //         m_wr_miss = &data_cache::wr_miss_wa_fetch_on_write;
    //         break;
    //       case LAZY_FETCH_ON_READ:
    //         m_wr_miss = &data_cache::wr_miss_wa_lazy_fetch_on_read;
    //         break;
    //       default:
    //         assert(0 && "Error: Must set valid cache write miss policy\n");
    //         break;  // Need to set a write miss function
    //     }
    //   }

    pub fn cycle(&mut self) {}

    pub fn fill(&self, fetch: &mem_fetch::MemFetch) {}

    pub fn has_free_fill_port(&self) -> bool {
        false
    }

    // virtual enum cache_request_status access(
    //     new_addr_type addr, mem_fetch *mf,
    //    unsigned time,
    //    std::list<cache_event> &events);
}

#[cfg(test)]
mod tests {
    use super::DataL1;
    use crate::config::GPUConfig;
    use crate::ported::{mem_fetch, WarpInstruction};
    use std::sync::{Arc, Mutex};

    struct Interconnect {}

    impl Interconnect {}

    fn concat<T>(
        mut a: impl IntoIterator<Item = T>,
        b: impl IntoIterator<Item = T>,
    ) -> impl Iterator<Item = T> {
        a.into_iter().chain(b.into_iter())
    }

    #[test]
    fn test_data_l1() {
        let config = Arc::new(GPUConfig::default());
        let cache_config = config.data_cache_l1.clone().unwrap();
        let interconn = Interconnect {};
        let l1 = DataL1::new(0, interconn, cache_config);

        let control_size = 0;
        let warp_id = 0;
        let core_id = 0;
        let cluster_id = 0;

        let kernel = crate::ported::KernelInfo::new(trace_model::KernelLaunch {
            name: "void vecAdd<float>(float*, float*, float*, int)".to_string(),
            trace_file: "./test-apps/vectoradd/traces/vectoradd-100-32-trace/kernel-0-trace".into(),
            id: 0,
            grid: nvbit_model::Dim { x: 1, y: 1, z: 1 },
            block: nvbit_model::Dim {
                x: 1024,
                y: 1,
                z: 1,
            },
            shared_mem_bytes: 0,
            num_registers: 8,
            binary_version: 61,
            stream_id: 0,
            shared_mem_base_addr: 140663786045440,
            local_mem_base_addr: 140663752491008,
            nvbit_version: "1.5.5".to_string(),
        });

        let trace_instr = trace_model::MemAccessTraceEntry {
            cuda_ctx: 0,
            kernel_id: 0,
            block_id: nvbit_model::Dim { x: 0, y: 0, z: 0 },
            warp_id: 3,
            line_num: 0,
            instr_data_width: 4,
            instr_opcode: "LDG.E.CG".to_string(),
            instr_offset: 176,
            instr_idx: 16,
            instr_predicate: nvbit_model::Predicate {
                num: 0,
                is_neg: false,
                is_uniform: false,
            },
            instr_mem_space: nvbit_model::MemorySpace::Global,
            instr_is_load: true,
            instr_is_store: false,
            instr_is_extended: true,
            active_mask: 15,
            addrs: concat(
                [
                    140663086646144,
                    140663086646148,
                    140663086646152,
                    140663086646156,
                ],
                [0; 32 - 4],
            )
            .collect::<Vec<_>>()
            .try_into()
            .unwrap(),
        };
        let mut instr = WarpInstruction::from_trace(&kernel, trace_instr);
        dbg!(&instr);
        let accesses = instr.generate_mem_accesses(&*config);
        println!(
            "generated {:?} accesses",
            accesses.map(|a| a.len()).unwrap_or(0)
        );
        let access = mem_fetch::MemAccess::from_instr(&instr).unwrap();
        let fetch = mem_fetch::MemFetch::new(
            instr,
            access,
            &config,
            control_size,
            warp_id,
            core_id,
            cluster_id,
        );
        l1.access(0x00000000, fetch, vec![]);
        assert!(false);
    }
}
