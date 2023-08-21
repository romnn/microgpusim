use crate::{address, cache, config, interconn as ic, mem_fetch, tag_array};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

#[derive(Debug)]
pub struct ReadOnly<I> {
    inner: cache::base::Base<I>,
}

impl<I> ReadOnly<I> {
    pub fn new(
        name: String,
        core_id: usize,
        cluster_id: usize,
        mem_port: Arc<I>,
        stats: Arc<Mutex<stats::Cache>>,
        config: Arc<config::GPU>,
        cache_config: Arc<config::Cache>,
    ) -> Self {
        let inner = cache::base::Base::new(
            name,
            core_id,
            cluster_id,
            mem_port,
            stats,
            config,
            cache_config,
        );
        Self { inner }
    }
}

impl<I> crate::engine::cycle::Component for ReadOnly<I>
where
    I: ic::MemFetchInterface,
{
    fn cycle(&mut self, cycle: u64) {
        self.inner.cycle(cycle);
    }
}

impl<I> cache::Bandwidth for ReadOnly<I>
where
    I: ic::MemFetchInterface,
{
    fn has_free_data_port(&self) -> bool {
        self.inner.has_free_data_port()
    }

    fn has_free_fill_port(&self) -> bool {
        self.inner.has_free_data_port()
    }
}

impl<I> cache::Cache for ReadOnly<I>
where
    I: ic::MemFetchInterface + 'static,
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn stats(&self) -> &Arc<Mutex<stats::Cache>> {
        &self.inner.stats
    }

    fn has_ready_accesses(&self) -> bool {
        self.inner.has_ready_accesses()
    }

    fn next_access(&mut self) -> Option<mem_fetch::MemFetch> {
        self.inner.next_access()
    }

    fn ready_accesses(&self) -> Option<&VecDeque<mem_fetch::MemFetch>> {
        self.inner.ready_accesses()
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
        use cache::RequestStatus as Status;

        let cache::base::Base {
            ref cache_config,
            ref mut tag_array,
            ..
        } = self.inner;
        debug_assert!(fetch.data_size() <= cache_config.atom_size());
        debug_assert_eq!(
            cache_config.write_policy,
            config::CacheWritePolicy::READ_ONLY
        );
        debug_assert!(!fetch.is_write());
        let block_addr = cache_config.block_addr(addr);

        log::debug!(
            "{}::readonly_cache::access({addr}, write = {}, data size = {}, control size = {}, block = {block_addr}, time={})",
            self.inner.name,
            fetch.is_write(),
            fetch.data_size(),
            fetch.control_size(),
            time,
        );

        let is_probe = false;
        let (cache_index, probe_status) =
            tag_array.probe(block_addr, &fetch, fetch.is_write(), is_probe);
        let mut status = Status::RESERVATION_FAIL;

        if probe_status == Status::HIT {
            // update LRU state
            tag_array::AccessStatus { status, .. } = tag_array.access(block_addr, &fetch, time);
        } else if probe_status != Status::RESERVATION_FAIL {
            if self.inner.miss_queue_full() {
                status = Status::RESERVATION_FAIL;
                self.inner.stats.lock().unwrap().inc(
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
                    cache_index.unwrap(),
                    fetch.clone(),
                    time,
                    events,
                    true,
                    false,
                );
                if should_miss {
                    status = Status::MISS;
                } else {
                    status = Status::RESERVATION_FAIL;
                }
            }
        } else {
            self.inner.stats.lock().unwrap().inc(
                *fetch.access_kind(),
                cache::AccessStat::ReservationFailure(cache::ReservationFailure::LINE_ALLOC_FAIL),
                1,
            );
        }
        self.inner.stats.lock().unwrap().inc(
            *fetch.access_kind(),
            cache::AccessStat::Status(select_status(probe_status, status)),
            1,
        );
        status
    }

    fn fill(&mut self, fetch: mem_fetch::MemFetch, time: u64) {
        self.inner.fill(fetch, time);
    }
}

/// This function selects how the cache access outcome should be counted.
///
/// `HIT_RESERVED` is considered as a MISS in the cores, however, it should be
/// counted as a `HIT_RESERVED` in the caches.
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
