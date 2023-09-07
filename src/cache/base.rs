use crate::sync::{Arc, Mutex};
use crate::{
    address, cache, config, interconn as ic, mem_fetch,
    mem_sub_partition::SECTOR_SIZE,
    mshr::{self, MSHR},
    tag_array,
};
use cache::block::Block;
use console::style;
use std::collections::{HashMap, VecDeque};
use tag_array::{Access, CacheAddressTranslation};

#[derive(Debug)]
struct PendingRequest {
    valid: bool,
    block_addr: address,
    addr: address,
    cache_index: usize,
    data_size: u32,
    // this variable is used when a load request generates multiple load
    // transactions For example, a read request from non-sector L1 request sends
    // a request to sector L2
    pending_reads: usize,
}

impl PendingRequest {}

/// Base cache
///
/// Implements common functions for `read_only_cache` and `data_cache`
/// Each subclass implements its own 'access' function
#[derive()]
pub struct Base<CC> {
    pub name: String,
    pub core_id: usize,
    pub cluster_id: usize,

    pub stats: Arc<Mutex<stats::Cache>>,
    pub cache_controller: CC,
    pub cache_config: cache::Config,

    pub miss_queue: VecDeque<mem_fetch::MemFetch>,
    pub miss_queue_status: mem_fetch::Status,
    pub mshrs: mshr::Table<mem_fetch::MemFetch>,
    pub tag_array: tag_array::TagArray<cache::block::Line, CC>,
    pending: HashMap<mem_fetch::MemFetch, PendingRequest>,
    top_port: Option<ic::Port<mem_fetch::MemFetch>>,
    pub bandwidth: super::bandwidth::Manager,
}

impl<CC> std::fmt::Debug for Base<CC> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Base")
            .field("name", &self.name)
            .field("core_id", &self.core_id)
            .field("cluster_id", &self.cluster_id)
            .field("miss_queue", &self.miss_queue)
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct Builder<CC> {
    pub name: String,
    pub core_id: usize,
    pub cluster_id: usize,
    pub stats: Arc<Mutex<stats::Cache>>,
    pub cache_controller: CC,
    pub cache_config: Arc<config::Cache>,
}

impl<CC> Builder<CC>
where
    CC: Clone,
{
    #[must_use]
    pub fn build(self) -> Base<CC> {
        let cache_config = self.cache_config;
        let tag_array =
            tag_array::TagArray::new(cache_config.clone(), self.cache_controller.clone());

        debug_assert!(matches!(
            cache_config.mshr_kind,
            mshr::Kind::ASSOC | mshr::Kind::SECTOR_ASSOC
        ));
        let mshrs = mshr::Table::new(cache_config.mshr_entries, cache_config.mshr_max_merge);

        let bandwidth = super::bandwidth::Manager::new(cache_config.clone());

        let cache_config = cache::Config::from(&*cache_config);

        let miss_queue = VecDeque::with_capacity(cache_config.miss_queue_size);

        Base {
            name: self.name,
            core_id: self.core_id,
            cluster_id: self.cluster_id,
            tag_array,
            mshrs,
            top_port: None,
            stats: self.stats,
            cache_config,
            cache_controller: self.cache_controller,
            bandwidth,
            pending: HashMap::new(),
            miss_queue,
            miss_queue_status: mem_fetch::Status::INITIALIZED,
        }
    }
}

impl<CC> Base<CC>
where
    CC: CacheAddressTranslation,
{
    /// Read miss handler.
    ///
    /// Check MSHR hit or MSHR available
    pub fn send_read_request(
        &mut self,
        addr: address,
        block_addr: u64,
        cache_index: usize,
        mut fetch: mem_fetch::MemFetch,
        time: u64,
        events: &mut Vec<super::event::Event>,
        read_only: bool,
        write_allocate: bool,
    ) -> (bool, bool, Option<tag_array::EvictedBlockInfo>) {
        let mut should_miss = false;
        let mut writeback = false;
        let mut evicted = None;

        let mshr_addr = self.cache_controller.mshr_addr(fetch.addr());
        let mshr_hit = self.mshrs.get(mshr_addr).is_some();
        let mshr_full = self.mshrs.full(mshr_addr);

        log::debug!(
            "{}::baseline_cache::send_read_request({}) (addr={}, block={}, mshr_addr={}, mshr_hit={}, mshr_full={}, miss_queue_full={})",
            &self.name, fetch, addr, block_addr, &mshr_addr, mshr_hit, mshr_full, self.miss_queue_full(),
        );

        if mshr_hit && !mshr_full {
            if read_only {
                let _ = self.tag_array.access(block_addr, &fetch, time);
            } else {
                tag_array::AccessStatus {
                    writeback,
                    evicted,
                    ..
                } = self.tag_array.access(block_addr, &fetch, time);
            }

            self.mshrs.add(mshr_addr, fetch.clone());
            let mut stats = self.stats.lock();
            stats.inc(
                *fetch.access_kind(),
                super::AccessStat::Status(super::RequestStatus::MSHR_HIT),
                1,
            );

            should_miss = true;
        } else if !mshr_hit && !mshr_full && !self.miss_queue_full() {
            if read_only {
                let _ = self.tag_array.access(block_addr, &fetch, time);
            } else {
                tag_array::AccessStatus {
                    writeback,
                    evicted,
                    ..
                } = self.tag_array.access(block_addr, &fetch, time);
            }

            let is_sector_cache = self.cache_config.mshr_kind == mshr::Kind::SECTOR_ASSOC;
            self.pending.insert(
                fetch.clone(),
                PendingRequest {
                    valid: true,
                    block_addr: mshr_addr,
                    addr: fetch.addr(),
                    cache_index,
                    data_size: fetch.data_size(),
                    pending_reads: if is_sector_cache {
                        self.cache_config.line_size / SECTOR_SIZE
                    } else {
                        0
                    } as usize,
                },
            );

            // change address to mshr block address
            fetch.access.req_size_bytes = self.cache_config.atom_size;
            fetch.access.addr = mshr_addr;

            self.mshrs.add(mshr_addr, fetch.clone());
            self.miss_queue.push_back(fetch.clone());
            fetch.set_status(self.miss_queue_status, time);
            if !write_allocate {
                events.push(super::event::Event::ReadRequestSent);
            }

            should_miss = true;
        } else if mshr_hit && mshr_full {
            self.stats.lock().inc(
                *fetch.access_kind(),
                super::AccessStat::ReservationFailure(
                    super::ReservationFailure::MSHR_MERGE_ENTRY_FAIL,
                ),
                1,
            );
        } else if !mshr_hit && mshr_full {
            self.stats.lock().inc(
                *fetch.access_kind(),
                super::AccessStat::ReservationFailure(super::ReservationFailure::MSHR_ENTRY_FAIL),
                1,
            );
        } else {
            panic!(
                "mshr_hit={} mshr_full={} miss_queue_full={}",
                mshr_hit,
                mshr_full,
                self.miss_queue_full()
            );
        }
        (should_miss, writeback, evicted)
    }
}

impl<CC> crate::engine::cycle::Component for Base<CC> {
    /// Sends next request to top memory in the memory hierarchy.
    fn cycle(&mut self, cycle: u64) {
        let Some(ref top_level_memory_port) = self.top_port else {
            // panic!("missing top port");
            return;
        };

        log::debug!(
            "{}::baseline cache::cycle miss queue={:?}",
            self.name,
            style(
                self.miss_queue
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
            )
            .blue(),
        );
        if let Some(fetch) = self.miss_queue.front() {
            let mut top_level_memory_port = top_level_memory_port.lock();
            let packet_size = if fetch.is_write() {
                fetch.size()
            } else {
                fetch.control_size()
            };
            if top_level_memory_port.can_send(&[packet_size]) {
                let fetch = self.miss_queue.pop_front().unwrap();
                log::debug!(
                    "{}::baseline cache::memport::push({}, data size={}, control size={})",
                    &self.name,
                    fetch.addr(),
                    fetch.data_size(),
                    fetch.control_size(),
                );
                top_level_memory_port.send(ic::Packet {
                    data: fetch,
                    time: cycle,
                });
            }
        }
        // let _data_port_busy = !self.has_free_data_port();
        // let _fill_port_busy = !self.has_free_fill_port();
        // m_stats.sample_cache_port_utility(data_port_busy, fill_port_busy);
        self.bandwidth.replenish_port_bandwidth();
    }
}

impl<CC> Base<CC> {
    /// Checks whether this request can be handled in this cycle.
    ///
    /// `n` equals the number of misses to be handled in this cycle.
    #[must_use]
    pub fn miss_queue_can_fit(&self, n: usize) -> bool {
        self.miss_queue.len() + n < self.cache_config.miss_queue_size
    }

    /// Checks whether the miss queue is full.
    ///
    /// This leads to misses not being handled in this cycle.
    #[must_use]
    pub fn miss_queue_full(&self) -> bool {
        self.miss_queue.len() >= self.cache_config.miss_queue_size
    }

    /// Checks if fetch is waiting to be filled by lower memory level
    #[must_use]
    pub fn waiting_for_fill(&self, fetch: &mem_fetch::MemFetch) -> bool {
        self.pending.contains_key(fetch)
    }

    /// Are any (accepted) accesses that had to wait for memory now ready?
    ///
    /// Note: does not include accesses that "HIT"
    #[must_use]
    pub fn has_ready_accesses(&self) -> bool {
        self.mshrs.has_ready_accesses()
    }

    #[must_use]
    pub fn ready_accesses(&self) -> Option<&VecDeque<mem_fetch::MemFetch>> {
        self.mshrs.ready_accesses()
    }

    /// Pop next ready access
    ///
    /// Note: does not include accesses that "HIT"
    pub fn next_access(&mut self) -> Option<mem_fetch::MemFetch> {
        self.mshrs.next_access()
    }

    #[inline]
    pub fn set_top_port(&mut self, port: ic::Port<mem_fetch::MemFetch>) {
        self.top_port = Some(port);
    }
}

impl<CC> Base<CC>
where
    CC: tag_array::CacheAddressTranslation,
{
    /// Flush all entries in cache
    pub fn flush(&mut self) -> usize {
        self.tag_array.flush()
    }

    /// Invalidate all entries in cache
    pub fn invalidate(&mut self) {
        self.tag_array.invalidate();
    }

    /// Interface for response from lower memory level.
    ///
    /// bandwidth restictions should be modeled in the caller.
    pub fn fill(&mut self, mut fetch: mem_fetch::MemFetch, time: u64) {
        if self.cache_config.mshr_kind == mshr::Kind::SECTOR_ASSOC {
            let original_fetch = fetch.original_fetch.as_ref().unwrap();
            let pending = self.pending.get_mut(original_fetch).unwrap();
            pending.pending_reads -= 1;
            todo!("sector assoc cache");
        }

        let pending = self.pending.remove(&fetch).unwrap();
        self.bandwidth.use_fill_port(&fetch);

        debug_assert!(pending.valid);
        // fetch.data_size = pending.data_size();
        fetch.access.req_size_bytes = pending.data_size;
        fetch.access.addr = pending.addr;

        match self.cache_config.allocate_policy {
            cache::config::AllocatePolicy::ON_MISS => {
                self.tag_array.fill_on_miss(
                    pending.cache_index,
                    fetch.addr(),
                    &fetch.access.sector_mask,
                    &fetch.access.byte_mask,
                    time,
                );
            }
            cache::config::AllocatePolicy::ON_FILL => {
                self.tag_array.fill_on_fill(
                    pending.block_addr,
                    &fetch.access.sector_mask,
                    &fetch.access.byte_mask,
                    fetch.is_write(),
                    time,
                );
            }
            other @ cache::config::AllocatePolicy::STREAMING => {
                unimplemented!("cache allocate policy {:?} is not implemented", other)
            }
        }

        let access_sector_mask = fetch.access.sector_mask;
        let access_byte_mask = fetch.access.byte_mask;

        let has_atomic = self
            .mshrs
            .mark_ready(pending.block_addr, fetch)
            .unwrap_or(false);

        if has_atomic {
            debug_assert!(
                self.cache_config.allocate_policy == cache::config::AllocatePolicy::ON_MISS
            );
            let block = self.tag_array.get_block_mut(pending.cache_index);
            // mark line as dirty for atomic operation
            let was_modified_before = block.is_modified();
            block.set_status(super::block::Status::MODIFIED, &access_sector_mask);
            block.set_byte_mask(&access_byte_mask);
            if !was_modified_before {
                self.tag_array.num_dirty += 1;
            }
        }
    }
}

impl<CC> super::Bandwidth for Base<CC> {
    fn has_free_data_port(&self) -> bool {
        self.bandwidth.has_free_data_port()
    }

    fn has_free_fill_port(&self) -> bool {
        self.bandwidth.has_free_fill_port()
    }
}

#[cfg(test)]
mod tests {
    // use super::Base;
    // use crate::{config, interconn as ic, Cycle, FromConfig, Packet};
    //
    // use crate::sync::{Arc, Mutex};

    // #[ignore = "todo"]
    // #[test]
    // fn base_cache_init() {
    //     let core_id = 0;
    //     let cluster_id = 0;
    //     let config = Arc::new(config::GPUConfig::default());
    //     let cache_stats = Arc::new(Mutex::new(stats::Cache::default()));
    //     let cache_config = config.data_cache_l1.clone().unwrap();
    //
    //     let stats = Arc::new(Mutex::new(stats::Stats::from_config(&config)));
    //     let interconn: Arc<ic::ToyInterconnect<Packet>> = Arc::new(ic::ToyInterconnect::new(0, 0));
    //     let port = Arc::new(ic::CoreMemoryInterface {
    //         interconn,
    //         interconn_port,
    //         cluster_id: 0,
    //         stats,
    //         config: config.clone(),
    //     });
    //
    //     let cycle = Cycle::new(0);
    //     let base = Base::new(
    //         "base cache".to_string(),
    //         core_id,
    //         cluster_id,
    //         cycle,
    //         port,
    //         cache_stats,
    //         config,
    //         Arc::clone(&cache_config.inner),
    //     );
    //     dbg!(&base);
    //     assert!(false);
    // }
}
