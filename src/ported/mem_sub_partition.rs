use crate::config;

use super::MemFetch;
use std::collections::{HashSet, VecDeque};

pub trait Queue<T> {
    fn new<S: ToString>(name: S, n: usize, queue: usize) -> Self;
    fn enqueue(&mut self, value: T);
    fn dequeue(&mut self) -> Option<T>;
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FifoQueue<T> {
    inner: VecDeque<T>,
}

impl<T> FifoQueue<T> {}

impl<T> Queue<T> for FifoQueue<T> {
    fn new<S: ToString>(name: S, n: usize, queue: usize) -> Self {
        Self {
            inner: VecDeque::new(),
        }
    }

    fn enqueue(&mut self, value: T) {
        self.inner.push_back(value);
    }
    fn dequeue(&mut self) -> Option<T> {
        self.inner.pop_front()
    }
}

// todo: what do we need precisely
// #[derive(Clone, Debug)]
// pub struct MemorySubPartitionConfig {
//     config: super::GPUConfig,
//     enabled: bool,
//     num_mem_sub_partition: usize,
// }

#[derive(Debug)]
pub struct CacheRequestStatus {}

#[derive(Debug)]
pub enum CacheEvent {}

// TODO: make Cache a trait that can be implemented differently for L1 and L2 cache structs ...

#[derive(Clone, Debug)]
pub struct L2Cache {
    name: String,
    // config: MemorySubPartitionConfig,
    config: config::CacheConfig,
}

/// Models second level shared cache with global write-back
/// and write-allocate policies
impl L2Cache {
    pub fn new(
        name: impl Into<String>,
        // todo: what config is needed here
        config: config::CacheConfig,
        // config: MemorySubPartitionConfig,
        // config: MemorySubPartitionConfig,
        core_id: i32,
        kind: i32,
        // memport: mem_fetch_interface,
        // mfcreator: mem_fetch_allocator,
        status: MemFetchStatus,
        // gpu: Sim,
    ) -> Self {
        Self {
            name: name.into(),
            config,
        }
    }

    pub fn access(
        &self,
        addr: super::address,
        mem_fetch: MemFetch,
        time: usize,
        events: Vec<CacheEvent>,
    ) -> CacheRequestStatus {
        // assert!(mf->get_data_size() <= config.get_atom_sz());
        // let is_write = mem_fetch.is_write;
        let block_addr = self.config.block_addr(addr);
        // unsigned cache_index = (unsigned)-1;
        // enum cache_request_status probe_status =
        //       m_tag_array->probe(block_addr, cache_index, mf, mf->is_write(), true);
        // enum cache_request_status access_status =
        //       process_tag_probe(wr, probe_status, addr, cache_index, mf, time, events);
        // m_stats.inc_stats(mf->get_access_type(),
        //                    m_stats.select_stats_status(probe_status, access_status));
        // m_stats.inc_stats_pw(mf->get_access_type(), m_stats.select_stats_status(
        //                                                   probe_status, access_status));
        // return access_status;
        CacheRequestStatus {}
    }

    pub fn flush(&self) {}

    pub fn invalidate(&self) {}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(incorrect_ident_case)]
pub enum MemFetchStatus {
    MEM_FETCH_INITIALIZED,
    IN_L1I_MISS_QUEUE,
    IN_L1D_MISS_QUEUE,
    IN_L1T_MISS_QUEUE,
    IN_L1C_MISS_QUEUE,
    IN_L1TLB_MISS_QUEUE,
    IN_VM_MANAGER_QUEUE,
    IN_ICNT_TO_MEM,
    IN_PARTITION_ROP_DELAY,
    IN_PARTITION_ICNT_TO_L2_QUEUE,
    IN_PARTITION_L2_TO_DRAM_QUEUE,
    IN_PARTITION_DRAM_LATENCY_QUEUE,
    IN_PARTITION_L2_MISS_QUEUE,
    IN_PARTITION_MC_INTERFACE_QUEUE,
    IN_PARTITION_MC_INPUT_QUEU,
    IN_PARTITION_MC_BANK_ARB_QUEUE,
    IN_PARTITION_DRAM,
    IN_PARTITION_MC_RETURNQ,
    IN_PARTITION_DRAM_TO_L2_QUEUE,
    IN_PARTITION_L2_FILL_QUEUE,
    IN_PARTITION_L2_TO_ICNT_QUEUE,
    IN_ICNT_TO_SHADER,
    IN_CLUSTER_TO_SHADER_QUEUE,
    IN_SHADER_LDST_RESPONSE_FIFO,
    IN_SHADER_FETCHED,
    IN_SHADER_L1T_ROB,
    MEM_FETCH_DELETED,
    NUM_MEM_REQ_STAT,
}

#[derive(Debug)]
pub struct MemorySubPartition<Q = FifoQueue<MemFetch>> {
    /// global sub partition ID
    pub id: usize,
    /// memory configuration
    // pub config: MemorySubPartitionConfig,
    pub config: config::GPUConfig,
    /// queues
    interconn_to_l2_queue: Q,
    l2_to_dram_queue: Q,
    dram_to_l2_queue: Q,
    /// L2 cache hit response queue
    l2_to_interconn_queue: Q,

    // class mem_fetch *L2dramout;
    wb_addr: Option<u64>,

    // class memory_stats_t *m_stats;
    request_tracker: HashSet<MemFetch>,

    // This is a cycle offset that has to be applied to the l2 accesses to account
    // for the cudamemcpy read/writes. We want GPGPU-Sim to only count cycles for
    // kernel execution but we want cudamemcpy to go through the L2. Everytime an
    // access is made from cudamemcpy this counter is incremented, and when the l2
    // is accessed (in both cudamemcpyies and otherwise) this value is added to
    // the gpgpu-sim cycle counters.
    memcpy_cycle_offset: usize,
    l2_cache: Option<L2Cache>,
}

impl<Q> MemorySubPartition<Q>
where
    Q: Queue<MemFetch>,
{
    // pub fn new(id: usize, config: MemorySubPartitionConfig) -> Self {
    pub fn new(id: usize, config: config::GPUConfig) -> Self {
        // need to migrate memory config for this
        // assert!(id < config.num_mem_sub_partition);
        // assert!(id < config.numjjjkk);

        // m_L2interface = new L2interface(this);
        // m_mf_allocator = new partition_mf_allocator(config);
        //
        // let l2_cache = if config.data_cache_l2 {
        let l2_cache = match &config.data_cache_l2 {
            Some(l2_config) => Some(L2Cache::new(
                format!("L2_bank_{:03}", id),
                l2_config.clone(),
                -1,
                -1,
                // l2interface,
                // mf_allocator,
                MemFetchStatus::IN_PARTITION_L2_MISS_QUEUE,
                // gpu
            )),
            None => None,
        };

        let interconn_to_l2_queue =
            Q::new("icnt-to-L2", 0, config.dram_partition_queue_interconn_to_l2);
        let l2_to_dram_queue = Q::new("L2-to-dram", 0, config.dram_partition_queue_l2_to_dram);
        let dram_to_l2_queue = Q::new("dram-to-L2", 0, config.dram_partition_queue_dram_to_l2);
        let l2_to_interconn_queue =
            Q::new("L2-to-icnt", 0, config.dram_partition_queue_l2_to_interconn);

        Self {
            id,
            config,
            wb_addr: None,
            memcpy_cycle_offset: 0,
            interconn_to_l2_queue,
            l2_to_dram_queue,
            dram_to_l2_queue,
            l2_to_interconn_queue,
            request_tracker: HashSet::new(),
            l2_cache,
        }
    }

    pub fn flush_l2(&self) -> usize {
        if let Some(l2) = &self.l2_cache {
            l2.flush();
        }
        //  TODO: write the flushed data to the main memory
        0
    }

    pub fn busy(&self) -> bool {
        !self.request_tracker.is_empty()
    }

    pub fn invalidate_l2(&self) {
        if let Some(l2) = &self.l2_cache {
            l2.invalidate();
        }
    }

    // pub fn full(&self) -> bool {
    //     self.interconn_to_l2_queue.full()
    // }
    //
    // pub fn has_available_size(&self, size: usize) -> bool {
    //     self.interconn_to_l2_queue.has_available_size(size)
    // }
}

#[derive(Clone, Debug, Default)]
pub struct MemoryPartitionUnit {
    id: usize,
}

pub type mem_access_sector_mask = u64;

impl MemoryPartitionUnit {
    pub fn new(id: usize, config: config::GPUConfig) -> Self {
        Self { id }
    }

    pub fn handle_memcpy_to_gpu(
        &self,
        addr: super::address,
        global_subpart_id: usize,
        mask: mem_access_sector_mask,
    ) {
        //   unsigned p = global_sub_partition_id_to_local_id(global_subpart_id);
        //   std::string mystring = mask.to_string<char, std::string::traits_type,
        //                                         std::string::allocator_type>();
        //   MEMPART_DPRINTF(
        //       "Copy Engine Request Received For Address=%zx, local_subpart=%u, "
        //       "global_subpart=%u, sector_mask=%s \n",
        //       addr, p, global_subpart_id, mystring.c_str());
        //   m_sub_partition[p]->force_l2_tag_update(
        //       addr, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle, mask);
        // }
    }
}

// class memory_sub_partition {
//  public:
//   memory_sub_partition(unsigned sub_partition_id, const memory_config *config,
//                        class memory_stats_t *stats, class gpgpu_sim *gpu);
//   ~memory_sub_partition();
//
//   unsigned get_id() const { return m_id; }
//
//   bool busy() const;
//
//   void cache_cycle(unsigned cycle);
//
//   bool full() const;
//   bool full(unsigned size) const;
//   void push(class mem_fetch *mf, unsigned long long clock_cycle);
//   class mem_fetch *pop();
//   class mem_fetch *top();
//   void set_done(mem_fetch *mf);
//
//   unsigned flushL2();
//   unsigned invalidateL2();
//
//   // interface to L2_dram_queue
//   bool L2_dram_queue_empty() const;
//   class mem_fetch *L2_dram_queue_top() const;
//   void L2_dram_queue_pop();
//
//   // interface to dram_L2_queue
//   bool dram_L2_queue_full() const;
//   void dram_L2_queue_push(class mem_fetch *mf);
//
//   void visualizer_print(gzFile visualizer_file);
//   void print_cache_stat(unsigned &accesses, unsigned &misses) const;
//   void print(FILE *fp) const;
//
//   void accumulate_L2cache_stats(class cache_stats &l2_stats) const;
//   void get_L2cache_sub_stats(struct cache_sub_stats &css) const;
//
//   // Support for getting per-window L2 stats for AerialVision
//   void get_L2cache_sub_stats_pw(struct cache_sub_stats_pw &css) const;
//   void clear_L2cache_stats_pw();
//
//   void force_l2_tag_update(new_addr_type addr, unsigned time,
//                            mem_access_sector_mask_t mask) {
//     m_L2cache->force_tag_access(addr, m_memcpy_cycle_offset + time, mask);
//     m_memcpy_cycle_offset += 1;
//   }
//
//  private:
//   // data
//   unsigned m_id;  //< the global sub partition ID
//   const memory_config *m_config;
//   class l2_cache *m_L2cache;
//   class L2interface *m_L2interface;
//   class gpgpu_sim *m_gpu;
//   partition_mf_allocator *m_mf_allocator;
//
//   // model delay of ROP units with a fixed latency
//   struct rop_delay_t {
//     unsigned long long ready_cycle;
//     class mem_fetch *req;
//   };
//   std::queue<rop_delay_t> m_rop;
//
//   // these are various FIFOs between units within a memory partition
//   fifo_pipeline<mem_fetch> *m_icnt_L2_queue;
//   fifo_pipeline<mem_fetch> *m_L2_dram_queue;
//   fifo_pipeline<mem_fetch> *m_dram_L2_queue;
//   fifo_pipeline<mem_fetch> *m_L2_icnt_queue;  // L2 cache hit response queue
//
//   class mem_fetch *L2dramout;
//   unsigned long long int wb_addr;
//
//   class memory_stats_t *m_stats;
//
//   std::set<mem_fetch *> m_request_tracker;
//
//   friend class L2interface;
//
//   std::vector<mem_fetch *> breakdown_request_to_sector_requests(mem_fetch *mf);
//
//   // This is a cycle offset that has to be applied to the l2 accesses to account
//   // for the cudamemcpy read/writes. We want GPGPU-Sim to only count cycles for
//   // kernel execution but we want cudamemcpy to go through the L2. Everytime an
//   // access is made from cudamemcpy this counter is incremented, and when the l2
//   // is accessed (in both cudamemcpyies and otherwise) this value is added to
//   // the gpgpu-sim cycle counters.
//   unsigned m_memcpy_cycle_offset;
// };
