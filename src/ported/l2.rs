use crate::config;
use crate::ported::{self, address, cache, interconn as ic, l1, mem_fetch};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Generic data cache.
///
/// todo: move this to cache as its generic
#[derive(Debug)]
pub struct DataL2<I> {
    pub inner: l1::Data<I>,
    pub cache_config: Arc<config::L2DCacheConfig>,
}

impl<I> DataL2<I>
where
    I: ic::MemFetchInterface,
{
    pub fn new(
        name: String,
        core_id: usize,
        cluster_id: usize,
        cycle: ported::Cycle,
        fetch_interconn: Arc<I>,
        stats: Arc<Mutex<stats::Cache>>,
        config: Arc<config::GPUConfig>,
        cache_config: Arc<config::L2DCacheConfig>,
    ) -> Self {
        let inner = l1::Data::new(
            name,
            core_id,
            cluster_id,
            cycle,
            fetch_interconn,
            stats,
            config,
            cache_config.inner.clone(),
            mem_fetch::AccessKind::L2_WR_ALLOC_R,
            mem_fetch::AccessKind::L2_WRBK_ACC,
        );
        Self {
            inner,
            cache_config,
        }
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
    I: ic::MemFetchInterface + 'static,
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn stats(&self) -> &Arc<Mutex<stats::Cache>> {
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

    // This is a gapping hole we are poking in the system to quickly handle
    // filling the cache on cudamemcopies. We don't care about anything other than
    // L2 state after the memcopy - so just force the tag array to act as though
    // something is read or written without doing anything else.
    fn force_tag_access(&mut self, addr: address, time: u64, mask: mem_fetch::MemAccessSectorMask) {
        // todo!("cache: invalidate");

        // use bitvec::{array::BitArray, field::BitField, BitArr};
        // let byte_mask: mem_fetch::MemAccessByteMask = !bitvec::array::BitArray::ZERO;
        let byte_mask: mem_fetch::MemAccessByteMask = bitvec::array::BitArray::ZERO;
        // let access = mem_fetch::MemAccess {};
        // let fetch = mem_fetch::MemFetch {};
        self.inner
            .inner
            .tag_array
            .populate_memcopy(addr, mask, byte_mask, true, time);
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
        time: u64,
    ) -> cache::RequestStatus {
        self.inner.access(addr, fetch, events, time)
    }

    fn waiting_for_fill(&self, fetch: &mem_fetch::MemFetch) -> bool {
        self.inner.waiting_for_fill(fetch)
    }

    fn fill(&mut self, fetch: mem_fetch::MemFetch, time: u64) {
        self.inner.fill(fetch, time)
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
