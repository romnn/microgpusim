use crate::sync::{Arc, Mutex};
use crate::{address, cache, config, interconn as ic, mcu, mem_fetch, tag_array};
use mem_fetch::access::Kind as AccessKind;
use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct L2CacheController<MC, CC>
where
    MC: std::fmt::Debug,
    CC: std::fmt::Debug,
{
    // set_index_function: crate::set_index::linear::SetIndex,
    // config: cache::Config,
    memory_controller: MC,
    cache_controller: CC,
    // inner: crate::tag_array::Pascal,
    // inner: crate::tag_array::Pascal,
}

// impl Pascal {
//     pub fn new(config: cache::Config) -> Self {
//         Self {
//             config,
//             set_index_function: crate::set_index::linear::SetIndex::default(),
//         }
//     }
// }

impl<MC, CC> crate::tag_array::CacheAddressTranslation for L2CacheController<MC, CC>
where
    MC: mcu::MemoryController,
    CC: tag_array::CacheAddressTranslation,
{
    #[inline]
    fn tag(&self, addr: address) -> address {
        // todo!();
        self.cache_controller.tag(addr)
    }

    #[inline]
    fn block_addr(&self, addr: address) -> address {
        // todo!();
        self.cache_controller.block_addr(addr)
    }

    #[inline]
    fn set_index(&self, addr: address) -> u64 {
        // todo!();
        // let partition_addr = addr;
        let partition_addr = self.memory_controller.memory_partition_address(addr);
        println!("partition address for addr {} is {}", addr, partition_addr);
        self.cache_controller.set_index(partition_addr)
    }

    #[inline]
    fn mshr_addr(&self, addr: address) -> address {
        // todo!();
        self.cache_controller.mshr_addr(addr)
    }
}

/// Generic data cache.
#[derive(Debug)]
#[allow(clippy::module_name_repetitions)]
// pub struct DataL2<I> {
pub struct DataL2 {
    pub inner: super::data::Data<
        mcu::MemoryControllerUnit,
        L2CacheController<mcu::MemoryControllerUnit, tag_array::Pascal>,
    >,
    // pub inner: super::data::Data<I>,
    pub cache_config: Arc<config::L2DCache>,
}

impl DataL2
// impl<I> DataL2<I>
// where
//     // I: ic::MemFetchInterface,
//     I: crate::fifo::Queue<mem_fetch::MemFetch>,
{
    pub fn new(
        name: String,
        core_id: usize,
        cluster_id: usize,
        // fetch_interconn: Arc<I>,
        stats: Arc<Mutex<stats::Cache>>,
        config: Arc<config::GPU>,
        cache_config: Arc<config::L2DCache>,
    ) -> Self {
        // let memory_controller = config.address_mapping().clone();
        let mem_controller = mcu::MemoryControllerUnit::new(&*config).unwrap();
        let cache_controller = L2CacheController {
            memory_controller: mem_controller.clone(),
            cache_controller: tag_array::Pascal::new(
                cache::Config::from(cache_config.inner.as_ref()).into(),
            ),
        };
        let inner = super::data::Builder {
            name,
            core_id,
            cluster_id,
            stats,
            config,
            cache_controller,
            mem_controller,
            cache_config: cache_config.inner.clone(),
            write_alloc_type: AccessKind::L2_WR_ALLOC_R,
            write_back_type: AccessKind::L2_WRBK_ACC,
        }
        .build();
        // TODO: crate a builder for data cache and base cache
        // inner.inner.addr_translation = L2CacheController {
        //     // inner: mmeory_controller,
        // };
        Self {
            inner,
            cache_config,
        }
    }

    #[inline]
    pub fn set_top_port(&mut self, port: ic::Port<mem_fetch::MemFetch>) {
        self.inner.set_top_port(port);
    }
}

impl crate::engine::cycle::Component for DataL2
// impl<I> crate::engine::cycle::Component for DataL2<I>
// where
//     I: ic::MemFetchInterface,
{
    fn cycle(&mut self, cycle: u64) {
        self.inner.cycle(cycle);
    }
}

impl super::Cache for DataL2
// impl<I> super::Cache for DataL2<I>
// where
//     I: ic::MemFetchInterface + 'static,
{
    #[inline]
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    #[inline]
    fn stats(&self) -> &Arc<Mutex<stats::Cache>> {
        &self.inner.inner.stats
    }

    #[inline]
    fn write_allocate_policy(&self) -> cache::config::WriteAllocatePolicy {
        self.inner.write_allocate_policy()
    }

    #[inline]
    fn has_ready_accesses(&self) -> bool {
        self.inner.has_ready_accesses()
    }

    #[inline]
    fn ready_accesses(&self) -> Option<&VecDeque<mem_fetch::MemFetch>> {
        self.inner.ready_accesses()
    }

    #[inline]
    fn next_access(&mut self) -> Option<mem_fetch::MemFetch> {
        self.inner.next_access()
    }

    // This is a gapping hole we are poking in the system to quickly handle
    // filling the cache on cudamemcopies. We don't care about anything other than
    // L2 state after the memcopy - so just force the tag array to act as though
    // something is read or written without doing anything else.
    #[inline]
    fn force_tag_access(&mut self, addr: address, time: u64, sector_mask: &mem_fetch::SectorMask) {
        let byte_mask = mem_fetch::ByteMask::ZERO;
        let is_write = true;
        self.inner
            .inner
            .tag_array
            .fill_on_fill(addr, sector_mask, &byte_mask, is_write, time);
    }

    /// Access read only cache.
    ///
    /// returns `RequestStatus::RESERVATION_FAIL` if
    /// request could not be accepted (for any reason)
    #[inline]
    fn access(
        &mut self,
        addr: address,
        fetch: mem_fetch::MemFetch,
        events: &mut Vec<super::event::Event>,
        time: u64,
    ) -> super::RequestStatus {
        self.inner.access(addr, fetch, events, time)
    }

    #[inline]
    fn waiting_for_fill(&self, fetch: &mem_fetch::MemFetch) -> bool {
        self.inner.waiting_for_fill(fetch)
    }

    #[inline]
    fn fill(&mut self, fetch: mem_fetch::MemFetch, time: u64) {
        self.inner.fill(fetch, time);
    }

    #[inline]
    fn flush(&mut self) -> usize {
        self.inner.flush()
    }

    #[inline]
    fn invalidate(&mut self) {
        self.inner.invalidate()
    }
}

impl super::Bandwidth for DataL2
// impl<I> super::Bandwidth for DataL2<I>
// where
//     I: ic::MemFetchInterface,
{
    fn has_free_data_port(&self) -> bool {
        self.inner.has_free_data_port()
    }

    fn has_free_fill_port(&self) -> bool {
        self.inner.has_free_fill_port()
    }
}
