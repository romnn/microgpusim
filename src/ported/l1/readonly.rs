use super::base;
use crate::config;
use crate::ported::{self, address, cache, interconn as ic, mem_fetch, tag_array};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

#[derive(Debug)]
pub struct ReadOnly<I> {
    inner: base::Base<I>,
}

impl<I> ReadOnly<I> {
    pub fn new(
        name: String,
        core_id: usize,
        cluster_id: usize,
        cycle: ported::Cycle,
        mem_port: Arc<I>,
        stats: Arc<Mutex<stats::Cache>>,
        config: Arc<config::GPUConfig>,
        cache_config: Arc<config::CacheConfig>,
    ) -> Self {
        let inner = base::Base::new(
            name,
            core_id,
            cluster_id,
            cycle,
            mem_port,
            stats,
            config,
            cache_config,
        );
        Self { inner }
    }

    // pub fn access_ready(&self) -> bool {
    //     todo!("readonly: access_ready");
    // }
}

impl<I> cache::Component for ReadOnly<I>
where
    // I: ic::MemPort,
    I: ic::MemFetchInterface,
    // I: ic::Interconnect<crate::ported::core::Packet> + 'static,
{
    fn cycle(&mut self) {
        self.inner.cycle()
    }
}

impl<I> cache::CacheBandwidth for ReadOnly<I>
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
        // fn ready_access_iter(&self) -> () {
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
    ) -> cache::RequestStatus {
        use cache::RequestStatus as Status;

        let base::Base {
            ref cache_config,
            ref config,
            ref mut tag_array,
            ..
        } = self.inner;
        debug_assert!(fetch.data_size <= cache_config.atom_size());
        debug_assert_eq!(
            cache_config.write_policy,
            config::CacheWritePolicy::READ_ONLY
        );
        debug_assert!(!fetch.is_write());
        let block_addr = cache_config.block_addr(addr);

        log::debug!(
            "{}::readonly_cache::access({addr}, write = {}, data size = {}, control size = {}, block = {block_addr})",
            self.inner.name,
            fetch.is_write(),
            fetch.data_size,
            fetch.control_size,
        );

        let is_probe = false;
        let (cache_index, probe_status) =
            tag_array.probe(block_addr, &fetch, fetch.is_write(), is_probe);
        let mut status = Status::RESERVATION_FAIL;

        let time = self.inner.cycle.get();
        if probe_status == Status::HIT {
            // update LRU state
            tag_array::AccessStatus { status, .. } = tag_array.access(block_addr, &fetch, time);
        } else if probe_status != Status::RESERVATION_FAIL {
            if !self.inner.miss_queue_full() {
                let (should_miss, writeback, evicted) = self.inner.send_read_request(
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
            } else {
                status = Status::RESERVATION_FAIL;
                let mut stats = self.inner.stats.lock().unwrap();
                stats.inc(
                    *fetch.access_kind(),
                    cache::AccessStat::ReservationFailure(
                        cache::ReservationFailure::MISS_QUEUE_FULL,
                    ),
                    1,
                );
            }
        } else {
            let mut stats = self.inner.stats.lock().unwrap();
            stats.inc(
                *fetch.access_kind(),
                cache::AccessStat::ReservationFailure(cache::ReservationFailure::LINE_ALLOC_FAIL),
                1,
            );
        }
        let mut stats = self.inner.stats.lock().unwrap();
        stats.inc(
            *fetch.access_kind(),
            cache::AccessStat::Status(select_status(probe_status, status)),
            1,
        );
        status
    }

    fn fill(&mut self, fetch: mem_fetch::MemFetch) {
        self.inner.fill(fetch);
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
    use super::ReadOnly;
    use crate::config::GPUConfig;

    #[ignore = "todo"]
    #[test]
    fn test_read_only_cache() {
        // todo: compare accelsim::read_only_cache and readonly
        let config = GPUConfig::default().data_cache_l1.unwrap();
        assert!(false);
    }
}
