use crate::sync::{Arc, Mutex};
use crate::{address, cache, config, interconn as ic, mcu, mem_fetch};
use color_eyre::eyre;
use mem_fetch::access::Kind as AccessKind;
use std::collections::VecDeque;

#[allow(clippy::module_name_repetitions)]
#[derive(Clone)]
pub struct L2DataCacheController<MC, CC> {
    accelsim_compat: bool,
    memory_controller: MC,
    cache_controller: CC,
}

impl<MC, CC> cache::CacheController for L2DataCacheController<MC, CC>
where
    MC: mcu::MemoryController,
    CC: cache::CacheController,
{
    // #[inline]
    fn tag(&self, addr: address) -> address {
        self.cache_controller.tag(addr)
    }

    // #[inline]
    fn block_addr(&self, addr: address) -> address {
        self.cache_controller.block_addr(addr)
    }

    // #[inline]
    fn set_index(&self, addr: address) -> u64 {
        let partition_addr = if true || self.accelsim_compat {
            self.memory_controller.memory_partition_address(addr)
        } else {
            addr
        };
        // println!("partition address for addr {} is {}", addr, partition_addr);
        self.cache_controller.set_index(partition_addr)
    }

    // #[inline]
    fn set_bank(&self, addr: address) -> u64 {
        self.cache_controller.set_bank(addr)
    }

    // #[inline]
    fn mshr_addr(&self, addr: address) -> address {
        self.cache_controller.mshr_addr(addr)
    }
}

/// Generic data cache.
#[allow(clippy::module_name_repetitions)]
pub struct DataL2 {
    pub sub_partition_id: usize,
    pub partition_id: usize,
    pub cache_config: Arc<config::L2DCache>,
    pub inner: super::data::Data<
        mcu::MemoryControllerUnit,
        L2DataCacheController<
            mcu::MemoryControllerUnit,
            cache::controller::pascal::DataCacheController,
        >,
        stats::cache::PerKernel,
    >,
}

impl DataL2 {
    pub fn new(
        name: String,
        sub_partition_id: usize,
        partition_id: usize,
        stats: Arc<Mutex<stats::cache::PerKernel>>,
        config: Arc<config::GPU>,
        cache_config: Arc<config::L2DCache>,
    ) -> Self {
        let mem_controller = mcu::MemoryControllerUnit::new(&config).unwrap();
        let default_cache_controller = cache::controller::pascal::DataCacheController::new(
            cache::Config::new(cache_config.inner.as_ref(), config.accelsim_compat),
        );
        let cache_controller = L2DataCacheController {
            accelsim_compat: config.accelsim_compat,
            memory_controller: mem_controller.clone(),
            cache_controller: default_cache_controller,
        };
        let inner = super::data::Builder {
            name,
            stats,
            config,
            cache_controller,
            mem_controller,
            cache_config: cache_config.inner.clone(),
            write_alloc_type: AccessKind::L2_WR_ALLOC_R,
            write_back_type: AccessKind::L2_WRBK_ACC,
        }
        .build();
        Self {
            inner,
            sub_partition_id,
            partition_id,
            cache_config,
        }
    }

    // #[inline]
    pub fn set_top_port(&mut self, port: ic::Port<mem_fetch::MemFetch>) {
        self.inner.set_top_port(port);
    }
}

impl crate::engine::cycle::Component for DataL2 {
    fn cycle(&mut self, cycle: u64) {
        self.inner.cycle(cycle);
    }
}

impl super::Cache<stats::cache::PerKernel> for DataL2 {
    // #[inline]
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    // #[inline]
    fn per_kernel_stats(&self) -> &Arc<Mutex<stats::cache::PerKernel>> {
        &self.inner.inner.stats
    }

    fn controller(&self) -> &dyn cache::CacheController {
        &self.inner.inner.cache_controller
    }

    fn write_state(
        &self,
        csv_writer: &mut csv::Writer<std::io::BufWriter<std::fs::File>>,
    ) -> eyre::Result<()> {
        self.inner.inner.tag_array.write_state(csv_writer)
    }

    // #[inline]
    fn write_allocate_policy(&self) -> cache::config::WriteAllocatePolicy {
        self.inner.write_allocate_policy()
    }

    // #[inline]
    fn has_ready_accesses(&self) -> bool {
        self.inner.has_ready_accesses()
    }

    // #[inline]
    fn ready_accesses(&self) -> Option<&VecDeque<mem_fetch::MemFetch>> {
        self.inner.ready_accesses()
    }

    // #[inline]
    fn next_access(&mut self) -> Option<mem_fetch::MemFetch> {
        self.inner.next_access()
    }

    // This is a gapping hole we are poking in the system to quickly handle
    // filling the cache on cudamemcopies. We don't care about anything other than
    // L2 state after the memcopy - so just force the tag array to act as though
    // something is read or written without doing anything else.
    // #[inline]
    // fn force_tag_access(&mut self, addr: address, time: u64, sector_mask: &mem_fetch::SectorMask) {
    //     let byte_mask = mem_fetch::ByteMask::ZERO;
    //     let is_write = true;
    //     self.inner
    //         .inner
    //         .tag_array
    //         .fill_on_fill(addr, sector_mask, &byte_mask, is_write, allocation_id, time);
    // }

    /// Access read only cache.
    ///
    /// returns `RequestStatus::RESERVATION_FAIL` if
    /// request could not be accepted (for any reason)
    // #[inline]
    fn access(
        &mut self,
        addr: address,
        fetch: mem_fetch::MemFetch,
        events: &mut Vec<super::event::Event>,
        time: u64,
    ) -> super::RequestStatus {
        self.inner.access(addr, fetch, events, time)
    }

    // #[inline]
    fn waiting_for_fill(&self, fetch: &mem_fetch::MemFetch) -> bool {
        self.inner.waiting_for_fill(fetch)
    }

    // #[inline]
    fn fill(&mut self, fetch: mem_fetch::MemFetch, time: u64) {
        self.inner.fill(fetch, time);
    }

    // #[inline]
    fn flush(&mut self) -> usize {
        self.inner.flush()
    }

    // #[inline]
    fn invalidate(&mut self) {
        self.inner.invalidate();
    }

    fn num_used_lines(&self) -> usize {
        self.inner.inner.tag_array.num_used_lines()
    }

    fn num_total_lines(&self) -> usize {
        self.inner.inner.tag_array.num_total_lines()
    }
}

impl super::Bandwidth for DataL2 {
    fn has_free_data_port(&self) -> bool {
        self.inner.has_free_data_port()
    }

    fn has_free_fill_port(&self) -> bool {
        self.inner.has_free_fill_port()
    }
}

#[cfg(test)]
mod tests {
    use crate::cache::CacheController;
    use color_eyre::eyre;

    #[test]
    fn test_l2d_set_index() -> eyre::Result<()> {
        let accelsim_compat = false;
        let config = crate::config::GPU::default();
        let l2_cache_config = &config.data_cache_l2.as_ref().unwrap().inner;

        // create l2 data cache controller
        let memory_controller = crate::mcu::MemoryControllerUnit::new(&config)?;
        let cache_controller = crate::cache::controller::pascal::DataCacheController::new(
            crate::cache::Config::new(l2_cache_config.as_ref(), accelsim_compat),
        );
        let l2_cache_controller = super::L2DataCacheController {
            accelsim_compat: false,
            memory_controller,
            cache_controller,
        };

        let block_addr = 34_887_082_112;
        assert_eq!(l2_cache_controller.set_index(block_addr), 1);
        Ok(())
    }
}
