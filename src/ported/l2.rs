use super::{address, cache, interconn as ic, l1, mem_fetch, stats::CacheStats};
use crate::config;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Generic data cache.
///
/// todo: move this to cache as its generic
#[derive(Debug)]
pub struct DataL2<I> {
    pub inner: l1::Data<I>,
}

impl<I> DataL2<I>
where
    I: ic::MemFetchInterface,
{
    pub fn new(
        name: String,
        core_id: usize,
        cluster_id: usize,
        fetch_interconn: Arc<I>,
        stats: Arc<Mutex<CacheStats>>,
        config: Arc<config::GPUConfig>,
        cache_config: Arc<config::CacheConfig>,
    ) -> Self {
        let inner = l1::Data::new(
            name,
            core_id,
            cluster_id,
            fetch_interconn,
            stats,
            config,
            cache_config,
            mem_fetch::AccessKind::L2_WR_ALLOC_R,
            mem_fetch::AccessKind::L2_WRBK_ACC,
        );
        Self { inner }
    }
}

impl<I> cache::Component for DataL2<I>
where
    I: ic::MemFetchInterface,
{
    fn cycle(&mut self) {
        self.inner.cycle()
    }
}

impl<I> cache::Cache for DataL2<I>
where
    I: ic::MemFetchInterface,
{
    fn stats(&self) -> &Arc<Mutex<CacheStats>> {
        &self.inner.inner.stats
    }

    fn write_allocate_policy(&self) -> config::CacheWriteAllocatePolicy {
        self.inner.write_allocate_policy()
    }

    fn has_ready_accesses(&self) -> bool {
        self.inner.has_ready_accesses()
    }

    fn ready_accesses(&self) -> Option<&VecDeque<mem_fetch::MemFetch>> {
        self.inner.ready_accesses()
    }

    fn next_access(&mut self) -> Option<mem_fetch::MemFetch> {
        self.inner.next_access()
    }

    /// Access read only cache.
    ///
    /// returns `RequestStatus::RESERVATION_FAIL` if
    /// request could not be accepted (for any reason)
    fn access(
        &mut self,
        addr: address,
        fetch: mem_fetch::MemFetch,
        events: &mut Vec<cache::Event>,
    ) -> cache::RequestStatus {
        self.inner.access(addr, fetch, events)
    }

    fn waiting_for_fill(&self, fetch: &mem_fetch::MemFetch) -> bool {
        self.inner.waiting_for_fill(fetch)
    }

    // fn fill(&mut self, fetch: &mut mem_fetch::MemFetch) {
    fn fill(&mut self, fetch: mem_fetch::MemFetch) {
        self.inner.fill(fetch)
    }
}

impl<I> cache::CacheBandwidth for DataL2<I>
where
    I: ic::MemFetchInterface,
{
    fn has_free_data_port(&self) -> bool {
        self.inner.has_free_data_port()
    }

    fn has_free_fill_port(&self) -> bool {
        self.inner.has_free_fill_port()
    }
}
