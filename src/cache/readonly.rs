use crate::sync::{Arc, Mutex};
use crate::{
    address, cache, config, interconn as ic, mcu, mem_fetch,
    tag_array::{self, Access, CacheAddressTranslation},
};
use std::collections::VecDeque;

#[derive(Debug)]
pub struct ReadOnly {
    inner: cache::base::Base<mcu::MemoryControllerUnit>,
}

// impl<I> ReadOnly<I> {
impl ReadOnly {
    pub fn new(
        name: String,
        core_id: usize,
        cluster_id: usize,
        stats: Arc<Mutex<stats::Cache>>,
        config: Arc<config::GPU>,
        cache_config: Arc<config::Cache>,
    ) -> Self {
        let inner = cache::base::Builder {
            name,
            core_id,
            cluster_id,
            stats,
            mem_controller: mcu::MemoryControllerUnit::new(&*config).unwrap(),
            cache_config,
        }
        .build();
        Self { inner }
    }

    #[inline]
    pub fn set_top_port(&mut self, port: ic::Port<mem_fetch::MemFetch>) {
        self.inner.set_top_port(port);
    }
}

impl crate::engine::cycle::Component for ReadOnly
// impl<I> crate::engine::cycle::Component for ReadOnly<I>
// where
//     I: ic::MemFetchInterface,
{
    fn cycle(&mut self, cycle: u64) {
        self.inner.cycle(cycle);
    }
}

impl cache::Bandwidth for ReadOnly
// impl<I> cache::Bandwidth for ReadOnly<I>
// where
// I: ic::MemFetchInterface,
{
    #[inline]
    fn has_free_data_port(&self) -> bool {
        self.inner.has_free_data_port()
    }

    #[inline]
    fn has_free_fill_port(&self) -> bool {
        self.inner.has_free_data_port()
    }
}

impl cache::Cache for ReadOnly
// impl<I> cache::Cache for ReadOnly<I>
// where
//     I: ic::MemFetchInterface + 'static,
{
    #[inline]
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    #[inline]
    fn stats(&self) -> &Arc<Mutex<stats::Cache>> {
        &self.inner.stats
    }

    #[inline]
    fn has_ready_accesses(&self) -> bool {
        self.inner.has_ready_accesses()
    }

    #[inline]
    fn next_access(&mut self) -> Option<mem_fetch::MemFetch> {
        self.inner.next_access()
    }

    #[inline]
    fn ready_accesses(&self) -> Option<&VecDeque<mem_fetch::MemFetch>> {
        self.inner.ready_accesses()
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
        events: &mut Vec<cache::Event>,
        time: u64,
    ) -> cache::RequestStatus {
        let cache::base::Base {
            ref cache_config,
            ref addr_translation,
            ref mut tag_array,
            ..
        } = self.inner;
        debug_assert!(fetch.data_size() <= cache_config.atom_size);
        debug_assert_eq!(
            cache_config.write_policy,
            cache::config::WritePolicy::READ_ONLY
        );
        debug_assert!(!fetch.is_write());
        let block_addr = addr_translation.block_addr(addr);

        log::debug!(
            "{}::readonly_cache::access({addr}, write = {}, data size = {}, control size = {}, block = {block_addr}, time={})",
            self.inner.name,
            fetch.is_write(),
            fetch.data_size(),
            fetch.control_size(),
            time,
        );

        let is_probe = false;
        // let (cache_index, probe_status) =
        //     tag_array.probe(block_addr, &fetch, fetch.is_write(), is_probe);

        let probe = tag_array.probe(block_addr, &fetch, fetch.is_write(), is_probe);
        let probe_status = probe
            .map(|(_, status)| status)
            .unwrap_or(cache::RequestStatus::RESERVATION_FAIL);

        let mut status = cache::RequestStatus::RESERVATION_FAIL;

        match probe {
            None => {
                self.inner.stats.lock().inc(
                    *fetch.access_kind(),
                    cache::AccessStat::ReservationFailure(
                        cache::ReservationFailure::LINE_ALLOC_FAIL,
                    ),
                    1,
                );
            }
            Some((_, cache::RequestStatus::HIT)) => {
                // update LRU state
                tag_array::AccessStatus { status, .. } = tag_array.access(block_addr, &fetch, time);
            }
            Some((cache_index, probe_status)) => {
                if self.inner.miss_queue_full() {
                    status = cache::RequestStatus::RESERVATION_FAIL;

                    self.inner.stats.lock().inc(
                        *fetch.access_kind(),
                        cache::AccessStat::ReservationFailure(
                            cache::ReservationFailure::MISS_QUEUE_FULL,
                        ),
                        1,
                    );
                } else {
                    let (should_miss, _writeback, _evicted) = self.inner.send_read_request(
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
                        status = cache::RequestStatus::MISS;
                    } else {
                        status = cache::RequestStatus::RESERVATION_FAIL;
                    }
                }
            }
        }

        // if probe_status == Status::HIT {
        //     // update LRU state
        //     tag_array::AccessStatus { status, .. } = tag_array.access(block_addr, &fetch, time);
        // } else if probe_status != Status::RESERVATION_FAIL {
        //     if self.inner.miss_queue_full() {
        //         status = Status::RESERVATION_FAIL;
        //
        //         self.inner.stats.lock().inc(
        //             *fetch.access_kind(),
        //             cache::AccessStat::ReservationFailure(
        //                 cache::ReservationFailure::MISS_QUEUE_FULL,
        //             ),
        //             1,
        //         );
        //     } else {
        //         let (should_miss, _writeback, _evicted) = self.inner.send_read_request(
        //             addr,
        //             block_addr,
        //             cache_index.unwrap(),
        //             fetch.clone(),
        //             time,
        //             events,
        //             true,
        //             false,
        //         );
        //         if should_miss {
        //             status = Status::MISS;
        //         } else {
        //             status = Status::RESERVATION_FAIL;
        //         }
        //     }
        // } else {
        //     self.inner.stats.lock().inc(
        //         *fetch.access_kind(),
        //         cache::AccessStat::ReservationFailure(cache::ReservationFailure::LINE_ALLOC_FAIL),
        //         1,
        //     );
        // }
        self.inner.stats.lock().inc(
            *fetch.access_kind(),
            cache::AccessStat::Status(select_status(probe_status, status)),
            1,
        );
        status
    }

    #[inline]
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
        self.inner.invalidate()
    }

    fn flush(&mut self) -> usize {
        self.inner.flush()
    }
}

/// This function selects how the cache access outcome should be counted.
///
/// `HIT_RESERVED` is considered as a MISS in the cores, however, it should be
/// counted as a `HIT_RESERVED` in the caches.
#[inline]
fn select_status(
    probe: cache::RequestStatus,
    access: cache::RequestStatus,
) -> cache::RequestStatus {
    use cache::RequestStatus;
    match probe {
        RequestStatus::HIT_RESERVED if access != RequestStatus::RESERVATION_FAIL => probe,
        RequestStatus::SECTOR_MISS if access == RequestStatus::MISS => probe,
        _ => access,
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
