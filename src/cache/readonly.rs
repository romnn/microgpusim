use crate::sync::{Arc, Mutex};
use crate::{
    address, cache, config, interconn as ic, mem_fetch,
    tag_array::{self, Access},
};
use cache::CacheController;
use color_eyre::eyre;
use std::collections::VecDeque;

pub struct ReadOnly {
    inner: cache::base::Base<
        cache::block::Line,
        // cache::controller::pascal::DataCacheController,
        cache::controller::pascal::L1DataCacheController,
        stats::cache::PerKernel,
    >,
}

impl ReadOnly {
    pub fn new(
        id: usize,
        name: String,
        kind: cache::base::Kind,
        stats: Arc<Mutex<stats::cache::PerKernel>>,
        readonly_cache_config: Arc<config::Cache>,
        accelsim_compat: bool,
    ) -> Self {
        let cache_config = cache::config::Config::new(&*readonly_cache_config, accelsim_compat);
        let cache_controller = cache::controller::pascal::L1DataCacheController::new(
            cache_config,
            // this is totally a hack and not a nice solution
            &crate::config::L1DCache {
                inner: Arc::clone(&readonly_cache_config),
                l1_latency: 1,
                l1_hit_latency: 1,
                l1_banks_byte_interleaving: 1,
                l1_banks: 1,
            },
            accelsim_compat,
        );
        let inner = cache::base::Builder {
            name,
            id,
            kind,
            stats,
            cache_controller,
            cache_config: readonly_cache_config,
            accelsim_compat,
        }
        .build();
        Self { inner }
    }

    // #[inline]
    pub fn set_top_port(&mut self, port: ic::Port<mem_fetch::MemFetch>) {
        self.inner.set_top_port(port);
    }
}

impl crate::engine::cycle::Component for ReadOnly {
    fn cycle(&mut self, cycle: u64) {
        self.inner.cycle(cycle);
    }
}

impl cache::Bandwidth for ReadOnly {
    // #[inline]
    fn has_free_data_port(&self) -> bool {
        self.inner.has_free_data_port()
    }

    // #[inline]
    fn has_free_fill_port(&self) -> bool {
        self.inner.has_free_data_port()
    }
}

impl cache::Cache<stats::cache::PerKernel> for ReadOnly {
    // #[inline]
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    // #[inline]
    fn per_kernel_stats(&self) -> &Arc<Mutex<stats::cache::PerKernel>> {
        &self.inner.stats
    }

    fn controller(&self) -> &dyn cache::CacheController {
        &self.inner.cache_controller
    }

    fn write_state(
        &self,
        csv_writer: &mut csv::Writer<std::io::BufWriter<std::fs::File>>,
    ) -> eyre::Result<()> {
        self.inner.tag_array.write_state(csv_writer)
    }

    // #[inline]
    fn has_ready_accesses(&self) -> bool {
        self.inner.has_ready_accesses()
    }

    // #[inline]
    fn next_access(&mut self) -> Option<mem_fetch::MemFetch> {
        self.inner.next_access()
    }

    // #[inline]
    fn ready_accesses(&self) -> Option<&VecDeque<mem_fetch::MemFetch>> {
        self.inner.ready_accesses()
    }

    /// Access read only cache.
    ///
    /// returns `RequestStatus::RESERVATION_FAIL` if
    /// request could not be accepted (for any reason)
    // #[inline]
    fn access(
        &mut self,
        addr: address,
        fetch: mem_fetch::MemFetch,
        events: &mut Vec<cache::Event>,
        time: u64,
    ) -> cache::RequestStatus {
        let cache::base::Base {
            ref cache_config,
            ref cache_controller,
            ref mut tag_array,
            ..
        } = self.inner;
        debug_assert!(fetch.data_size() <= cache_config.atom_size);
        debug_assert_eq!(
            cache_config.write_policy,
            cache::config::WritePolicy::READ_ONLY
        );
        debug_assert!(!fetch.is_write());
        let block_addr = cache_controller.block_addr(addr);

        log::debug!(
            "{}::readonly_cache::access({fetch}, warp = {}, size = {}, block = {block_addr}, time = {time}))",
            self.inner.name,
            fetch.warp_id,
            fetch.data_size(),
        );

        let is_probe = false;

        let probe = tag_array.probe(block_addr, &fetch, fetch.is_write(), is_probe);
        let probe_status =
            probe.map_or(cache::RequestStatus::RESERVATION_FAIL, |(_, status)| status);

        let mut access_status = cache::RequestStatus::RESERVATION_FAIL;

        log::info!(
            "{}::access({}) => probe status={:?} access status={:?}",
            self.inner.name,
            &fetch,
            probe_status,
            access_status
        );

        match probe {
            None => {
                let mut stats = self.inner.stats.lock();
                let kernel_stats = stats.get_mut(fetch.kernel_launch_id());
                kernel_stats.inc(
                    fetch.allocation_id(),
                    fetch.access_kind(),
                    cache::AccessStat::ReservationFailure(
                        cache::ReservationFailure::LINE_ALLOC_FAIL,
                    ),
                    1,
                );
            }
            Some((_, cache::RequestStatus::HIT)) => {
                // update LRU state
                let tag_array::AccessStatus { status, .. } =
                    tag_array.access(block_addr, &fetch, time);
                access_status = status;
            }
            Some((cache_index, _probe_status)) => {
                if self.inner.miss_queue_full() {
                    access_status = cache::RequestStatus::RESERVATION_FAIL;

                    let mut stats = self.inner.stats.lock();
                    let kernel_stats = stats.get_mut(fetch.kernel_launch_id());
                    kernel_stats.inc(
                        fetch.allocation_id(),
                        fetch.access_kind(),
                        cache::AccessStat::ReservationFailure(
                            cache::ReservationFailure::MISS_QUEUE_FULL,
                        ),
                        1,
                    );
                } else {
                    let (should_miss, _evicted) = self.inner.send_read_request(
                        addr,
                        block_addr,
                        cache_index,
                        fetch.clone(),
                        time,
                        events,
                        true,
                        false,
                    );
                    if should_miss {
                        access_status = cache::RequestStatus::MISS;
                    } else {
                        access_status = cache::RequestStatus::RESERVATION_FAIL;
                    }
                }
            }
        }

        let mut stats = self.inner.stats.lock();
        let kernel_stats = stats.get_mut(fetch.kernel_launch_id());
        let access_stat = if self.inner.cache_config.accelsim_compat {
            cache::select_status_accelsim_compat(probe_status, access_status)
        } else {
            cache::select_status(probe_status, access_status)
        };
        kernel_stats.inc(
            fetch.allocation_id(),
            fetch.access_kind(),
            cache::AccessStat::Status(access_stat),
            1,
        );
        access_status
    }

    // #[inline]
    fn fill(&mut self, fetch: mem_fetch::MemFetch, time: u64) {
        self.inner.fill(fetch, time);
    }

    fn waiting_for_fill(&self, _fetch: &mem_fetch::MemFetch) -> bool {
        false
    }

    fn write_allocate_policy(&self) -> cache::config::WriteAllocatePolicy {
        cache::config::WriteAllocatePolicy::NO_WRITE_ALLOCATE
    }

    fn invalidate(&mut self) {
        self.inner.invalidate();
    }

    fn invalidate_addr(&mut self, addr: address) {
        self.inner.invalidate_addr(addr);
    }

    fn flush(&mut self) -> usize {
        self.inner.flush()
    }

    fn num_used_lines(&self) -> usize {
        self.inner.tag_array.num_used_lines()
    }

    fn num_used_bytes(&self) -> u64 {
        self.inner.tag_array.num_used_lines() as u64 * self.inner.cache_config.line_size as u64
    }

    fn num_total_lines(&self) -> usize {
        self.inner.tag_array.num_total_lines()
    }
}

#[cfg(test)]
mod tests {
    use crate::config;

    #[ignore = "todo"]
    #[test]
    fn test_read_only_cache() {
        // todo: compare accelsim::read_only_cache and readonly
        let _config = config::GPU::default().data_cache_l1.unwrap();
    }
}
