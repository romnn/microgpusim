use super::mem_fetch::BitString;
use super::{address, cache, cache::Cache, interconn as ic, l2, mem_fetch, stats::Stats, Packet};
use crate::config::{self, CacheConfig, GPUConfig};
use std::cell::RefCell;
use std::collections::{HashSet, VecDeque};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

pub const MAX_MEMORY_ACCESS_SIZE: usize = 128;
// pub const std::bitset<MAX_MEMORY_ACCESS_SIZE> mem_access_byte_mask_t;
pub const SECTOR_CHUNCK_SIZE: usize = 4; // four sectors
pub const SECTOR_SIZE: usize = 32; // sector is 32 bytes width
                                   // typedef std::bitset<SECTOR_CHUNCK_SIZE> mem_access_sector_mask_t;

pub trait Queue<T> {
    fn new<S: ToString>(name: S, min_size: Option<usize>, max_size: Option<usize>) -> Self;
    fn enqueue(&mut self, value: T);
    fn dequeue(&mut self) -> Option<T>;
    fn first(&self) -> Option<&T>;
    fn full(&self) -> bool;
    fn is_empty(&self) -> bool;
    fn can_fit(&self, n: usize) -> bool;
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FifoQueue<T> {
    inner: VecDeque<T>,
    min_size: Option<usize>,
    max_size: Option<usize>,
}

impl<T> FifoQueue<T> {}

impl<T> Queue<T> for FifoQueue<T> {
    fn new<S: ToString>(name: S, min_size: Option<usize>, max_size: Option<usize>) -> Self {
        Self {
            inner: VecDeque::new(),
            min_size,
            max_size,
        }
    }

    fn enqueue(&mut self, value: T) {
        self.inner.push_back(value);
    }

    fn dequeue(&mut self) -> Option<T> {
        self.inner.pop_front()
    }

    fn first(&self) -> Option<&T> {
        self.inner.get(0)
    }

    fn full(&self) -> bool {
        match self.max_size {
            Some(max) => self.inner.len() >= max,
            None => false,
        }
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn can_fit(&self, n: usize) -> bool {
        // m_max_len && m_length + size - 1 >= m_max_len
        match self.max_size {
            Some(max) => self.inner.len() + n - 1 < max,
            None => true,
        }
    }
}

#[derive()]
pub struct MemorySubPartition<I, Q = FifoQueue<mem_fetch::MemFetch>> {
    pub id: usize,
    // pub cluster_id: usize,
    // pub core_id: usize,
    /// memory configuration
    pub config: Arc<GPUConfig>,
    pub stats: Arc<Mutex<Stats>>,

    /// queues
    interconn_to_l2_queue: Q,
    l2_to_dram_queue: Q,
    dram_to_l2_queue: Q,
    /// L2 cache hit response queue
    l2_to_interconn_queue: Q,

    fetch_interconn: Arc<I>,
    // l2_cache: Option<l2::DataL2<I>>,
    l2_cache: Option<Box<dyn cache::Cache>>, // l2::DataL2<I>>,
    // l2_cache: Option<l2::DataL2<ic::ToyInterconnect<Packet>>>,

    // class mem_fetch *L2dramout;
    wb_addr: Option<u64>,

    // class memory_stats_t *m_stats;
    request_tracker: HashSet<mem_fetch::MemFetch>,

    // This is a cycle offset that has to be applied to the l2 accesses to account
    // for the cudamemcpy read/writes. We want GPGPU-Sim to only count cycles for
    // kernel execution but we want cudamemcpy to go through the L2. Everytime an
    // access is made from cudamemcpy this counter is incremented, and when the l2
    // is accessed (in both cudamemcpyies and otherwise) this value is added to
    // the gpgpu-sim cycle counters.
    memcpy_cycle_offset: usize,
}

impl<I, Q> MemorySubPartition<I, Q>
where
    Q: Queue<mem_fetch::MemFetch>,
    I: ic::MemFetchInterface + 'static,
{
    pub fn new(
        id: usize,
        // cluster_id: usize,
        // core_id: usize,
        fetch_interconn: Arc<I>,
        config: Arc<GPUConfig>,
        stats: Arc<Mutex<Stats>>,
    ) -> Self {
        // need to migrate memory config for this
        // assert!(id < config.num_mem_sub_partition);
        // assert!(id < config.numjjjkk);

        let l2_cache: Option<Box<dyn cache::Cache>> = match &config.data_cache_l2 {
            Some(l2_config) => Some(Box::new(l2::DataL2::new(
                0, // core_id,
                0, // cluster_id,
                fetch_interconn.clone(),
                stats.clone(),
                config.clone(),
                l2_config.clone(),
            ))),
            None => None,
        };

        let interconn_to_l2_queue = Q::new(
            "icnt-to-L2",
            Some(0),
            Some(config.dram_partition_queue_interconn_to_l2),
        );
        let l2_to_dram_queue = Q::new(
            "L2-to-dram",
            Some(0),
            Some(config.dram_partition_queue_l2_to_dram),
        );
        let dram_to_l2_queue = Q::new(
            "dram-to-L2",
            Some(0),
            Some(config.dram_partition_queue_dram_to_l2),
        );
        let l2_to_interconn_queue = Q::new(
            "L2-to-icnt",
            Some(0),
            Some(config.dram_partition_queue_l2_to_interconn),
        );

        Self {
            id,
            // cluster_id,
            // core_id,
            config,
            stats,
            fetch_interconn,
            l2_cache,
            wb_addr: None,
            memcpy_cycle_offset: 0,
            interconn_to_l2_queue,
            l2_to_dram_queue,
            dram_to_l2_queue,
            l2_to_interconn_queue,
            request_tracker: HashSet::new(),
        }
    }

    pub fn push(&self, fetch: mem_fetch::MemFetch) {
        todo!("mem sub partition: push");
    }

    pub fn full(&self, size: usize) -> bool {
        // todo!("mem sub partition: full");
        self.l2_to_interconn_queue.full()
    }

    pub fn busy(&self) -> bool {
        !self.request_tracker.is_empty()
    }

    pub fn flush_l2(&mut self) {
        if let Some(l2) = &mut self.l2_cache {
            l2.flush();
        }
    }

    pub fn invalidate_l2(&mut self) {
        if let Some(l2) = &mut self.l2_cache {
            l2.invalidate();
        }
    }

    pub fn pop(&mut self) -> Option<mem_fetch::MemFetch> {
        use super::AccessKind;
        let fetch = self.l2_to_dram_queue.dequeue()?;
        // self.request_tracker.remove(fetch);
        match fetch.access_kind() {
            AccessKind::L2_WRBK_ACC | AccessKind::L1_WRBK_ACC => None,
            _ => Some(fetch),
        }
        // if (mf && (mf->get_access_type() == L2_WRBK_ACC ||
        //              mf->get_access_type() == L1_WRBK_ACC)) {
        //     delete mf;
        //     mf = NULL;
        //   }
        // f (mf && (mf->get_access_type() == L2_WRBK_ACC ||
        //              mf->get_access_type() == L1_WRBK_ACC)) {
        //     delete mf;
        //     mf = NULL;
        //   }
    }
    // mem_fetch *memory_sub_partition::pop() {
    //   mem_fetch *mf = m_L2_icnt_queue->pop();
    //   m_request_tracker.erase(mf);
    //   if (mf && mf->isatomic()) mf->do_atomic();
    //   if (mf && (mf->get_access_type() == L2_WRBK_ACC ||
    //              mf->get_access_type() == L1_WRBK_ACC)) {
    //     delete mf;
    //     mf = NULL;
    //   }
    //   return mf;
    // }
    //

    pub fn top(&self) -> Option<&mem_fetch::MemFetch> {
        use super::AccessKind;
        let fetch = self.l2_to_dram_queue.first()?;
        match fetch.access_kind() {
            AccessKind::L2_WRBK_ACC | AccessKind::L1_WRBK_ACC => {
                // self.l2_to_dram_queue.dequeue();
                // self.request_tracker.remove(fetch);
                None
            }
            _ => Some(fetch),
        }
        // if (mf && (mf->get_access_type() == L2_WRBK_ACC ||
        //            mf->get_access_type() == L1_WRBK_ACC)) {
        //   m_L2_icnt_queue->pop();
        //   m_request_tracker.erase(mf);
        //   delete mf;
        //   mf = NULL;
        // }
        // return mf;
    }

    // pub fn full(&self) -> bool {
    //     self.interconn_to_l2_queue.full()
    // }
    //
    // pub fn has_available_size(&self, size: usize) -> bool {
    //     self.interconn_to_l2_queue.has_available_size(size)
    // }

    pub fn set_done(&self, fetch: &mem_fetch::MemFetch) {
        todo!("mem sub partition: set done");
    }

    pub fn dram_l2_queue_push(&mut self, fetch: &mem_fetch::MemFetch) {
        todo!("mem sub partition: dram l2 queue push");
    }

    pub fn dram_l2_queue_full(&self) -> bool {
        todo!("mem sub partition: dram l2 queue full");
    }

    pub fn cache_cycle(&mut self, cycle: usize) {
        use config::CacheWriteAllocatePolicy;
        use mem_fetch::{AccessKind, Status};

        // L2 fill responses
        // if let Some(l2_config) = self.config.data_cache_l2 {
        if let Some(l2_cache) = &mut self.l2_cache {
            // todo: move config into l2
            let l2_config = self.config.data_cache_l2.as_ref().unwrap();
            // if !l2_config.disabled {}
            let queue_full = self.l2_to_interconn_queue.full();
            if l2_cache.has_ready_accesses() && !queue_full {
                let mut fetch = l2_cache.next_access().unwrap();
                // Don't pass write allocate read request back to upper level cache
                if fetch.access_kind() != &AccessKind::L2_WR_ALLOC_R {
                    // fetch.set_reply();
                    fetch.set_status(Status::IN_PARTITION_L2_TO_ICNT_QUEUE, 0);
                    // m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
                    self.l2_to_interconn_queue.enqueue(fetch);
                } else {
                    if l2_config.write_allocate_policy == CacheWriteAllocatePolicy::FETCH_ON_WRITE {
                        // mem_fetch *original_wr_mf = mf->get_original_wr_mf();
                        // assert(original_wr_mf);
                        // original_wr_mf->set_reply();
                        // original_wr_mf->set_status(
                        //     IN_PARTITION_L2_TO_ICNT_QUEUE,
                        //     m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
                        // self.l2_to_interconn_queue.push(original_wr_mf);
                    }
                    // self.request_tracker.remove(fetch);
                    // delete mf;
                }
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TempDRAM {}

impl TempDRAM {
    pub fn return_queue_pop(&mut self) -> Option<mem_fetch::MemFetch> {
        // TODO
        None
    }
    pub fn return_queue_top(&self) -> Option<&mem_fetch::MemFetch> {
        // TODO
        None
    }
}

#[derive()]
pub struct MemoryPartitionUnit<I> {
    id: usize,
    // cluster_id: usize,
    // core_id: usize,
    fetch_interconn: Arc<I>,
    pub sub_partitions: Vec<Rc<RefCell<MemorySubPartition<I>>>>,
    dram: TempDRAM,

    config: Arc<GPUConfig>,
    stats: Arc<Mutex<Stats>>,
}

impl<I> MemoryPartitionUnit<I>
where
    I: ic::MemFetchInterface + 'static,
{
    pub fn new(
        id: usize,
        // cluster_id: usize,
        // core_id: usize,
        fetch_interconn: Arc<I>,
        config: Arc<GPUConfig>,
        stats: Arc<Mutex<Stats>>,
    ) -> Self {
        let num_sub_partitions = config.num_sub_partition_per_memory_channel;
        let sub_partitions: Vec<_> = (0..num_sub_partitions)
            .map(|i| {
                let sub_id = id * num_sub_partitions + i;
                let sub = MemorySubPartition::new(
                    sub_id,
                    // cluster_id,
                    // core_id,
                    fetch_interconn.clone(),
                    config.clone(),
                    stats.clone(),
                );
                Rc::new(RefCell::new(sub))
            })
            .collect();

        Self {
            id,
            // cluster_id,
            // core_id,
            fetch_interconn,
            config,
            stats,
            dram: TempDRAM::default(),
            sub_partitions,
        }
    }

    // pub fn sub_partition(&self, p: usize) -> {
    //     self.sub_partitions[
    // }

    fn global_sub_partition_id_to_local_id(&self, global_sub_partition_id: usize) -> usize {
        let mut local_id = global_sub_partition_id;
        local_id -= self.id * self.config.num_sub_partition_per_memory_channel;
        local_id
    }

    pub fn handle_memcpy_to_gpu(
        &self,
        addr: address,
        global_subpart_id: usize,
        mask: mem_fetch::MemAccessSectorMask,
    ) {
        let p = self.global_sub_partition_id_to_local_id(global_subpart_id);
        // println!(
        //       "copy engine request received for address={}, local_subpart={}, global_subpart={}, sector_mask={}",
        //       addr, p, global_subpart_id, mask.to_bit_string());

        // self.mem_sub_partititon[p].force_l2_tag_update(addr, mask);
        // m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle, mask);
    }

    pub fn cache_cycle(&mut self, cycle: usize) {
        // todo!("mem partition unit: cache_cycle");
        // for p < m_config->m_n_sub_partition_per_memory_channel
        for mem_sub in self.sub_partitions.iter_mut() {
            mem_sub.borrow_mut().cache_cycle(cycle);
        }
    }

    pub fn simple_dram_model_cycle(&mut self) {
        todo!("mem partition unit: simple_dram_model_cycle");
        // for p < m_config->m_n_sub_partition_per_memory_channel
    }

    pub fn dram_cycle(&mut self) {
        use mem_fetch::{AccessKind, Status};
        // todo!("mem partition unit: dram_cycle");
        // TODO
        return;

        // pop completed memory request from dram and push it to
        // dram-to-L2 queue of the original sub partition
        if let Some(return_fetch) = self.dram.return_queue_top() {
            let dest_global_spid = return_fetch.sub_partition_id() as usize;
            let dest_spid = self.global_sub_partition_id_to_local_id(dest_global_spid);
            let mem_sub = self.sub_partitions[dest_spid].borrow();
            debug_assert_eq!(mem_sub.id, dest_global_spid);
            if !mem_sub.dram_l2_queue_full() {
                if return_fetch.access_kind() == &AccessKind::L1_WRBK_ACC {
                    mem_sub.set_done(return_fetch);
                    // delete mf_return;
                } else {
                    mem_sub.dram_l2_queue_push(return_fetch);
                    return_fetch.set_status(Status::IN_PARTITION_DRAM_TO_L2_QUEUE, 0);
                    // m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
                    // m_arbitration_metadata.return_credit(dest_spid);
                    println!(
                        "mem_fetch request {:?} return from dram to sub partition {}",
                        return_fetch, dest_spid
                    );
                }
                self.dram.return_queue_pop();
            }
        } else {
            self.dram.return_queue_pop();
        }
    }
}
