use super::{address, cache, interconn as ic, l1, mem_fetch, stats::Stats};
use crate::config;
use std::sync::{Arc, Mutex};

/// Generic data cache.
///
/// todo: move this to cache as its generic
#[derive(Debug)]
pub struct DataL2<I> {
    inner: l1::Data<I>,
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
        stats: Arc<Mutex<Stats>>,
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
    fn write_allocate_policy(&self) -> config::CacheWriteAllocatePolicy {
        self.inner.write_allocate_policy()
    }

    fn has_ready_accesses(&self) -> bool {
        self.inner.has_ready_accesses()
    }

    /// Access read only cache.
    ///
    /// returns `RequestStatus::RESERVATION_FAIL` if
    /// request could not be accepted (for any reason)
    fn access(
        &mut self,
        addr: address,
        fetch: mem_fetch::MemFetch,
        events: Option<&mut Vec<cache::Event>>,
    ) -> cache::RequestStatus {
        self.inner.access(addr, fetch, events)
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
        self.inner.has_free_data_port()
    }
}

// impl<I> cache::CacheBandwidth for DataL2<I>

//     /// Both the L1 and L2 currently use the same access function.
//     ///
//     /// Differentiation between the two caches is done through configuration
//     /// of caching policies.
//     /// Both the L1 and L2 override this function to provide a means of
//     /// performing actions specific to each cache when such actions are
//     /// implemnted.
//     fn access(
//         &mut self,
//         addr: address,
//         fetch: mem_fetch::MemFetch,
//         events: Option<&mut Vec<cache::Event>>,
//     ) -> cache::RequestStatus {
//         cache::RequestStatus::MISS
//     }
//
//     fn fill(&self, fetch: &mem_fetch::MemFetch) {
//         todo!("data cache: fill");
//     }
// }
//
// impl cache::CacheBandwidth for DataCache {
//     fn has_free_fill_port(&self) -> bool {
//         todo!("data cache: has_free_fill_port");
//         false
//     }
// }

// TODO (roman) temp disable l2
// /// Models second level shared cache.
// ///
// /// Uses global write-back and write-allocate policies by default.
// #[derive(Debug)]
// pub struct Data<I> {
//     inner: DataCache,
//     interconn: I,
// }
//
// impl<I> Data<I>
// where
//     I: ic::MemPort,
// {
//     pub fn new(
//         core_id: usize,
//         fetch_interconn: I,
//         stats: Arc<Mutex<Stats>>,
//         config: Arc<config::GPUConfig>,
//         cache_config: Arc<config::CacheConfig>,
//     ) -> Self {
//         Self {
//             inner: DataCache {},
//             interconn: fetch_interconn,
//         }
//     }
// }
//
// impl<I> cache::Component for Data<I> {}
//
// impl<I> cache::Cache for Data<I>
// where
//     I: ic::MemPort,
// {
//     // The l2 cache access function calls the base data_cache access
//     // implementation.  When the L2 needs to diverge from L1, L2 specific
//     // changes should be made here.
//     fn access(
//         &mut self,
//         addr: address,
//         fetch: mem_fetch::MemFetch,
//         events: Option<&mut Vec<cache::Event>>,
//     ) -> cache::RequestStatus {
//         self.inner.access(addr, fetch, events)
//     }
//
//     fn fill(&self, fetch: &mem_fetch::MemFetch) {
//         todo!("l2: fill");
//     }
// }

//     fn has_free_fill_port(&self) -> bool {
//         todo!("l2: has_free_fill_port");
//         false
//     }
// }

// class l2_cache : public data_cache {
//  public:
//   l2_cache(const char *name, cache_config &config, int core_id, int type_id,
//            mem_fetch_interface *memport, mem_fetch_allocator *mfcreator,
//            enum mem_fetch_status status, class gpgpu_sim *gpu)
//       : data_cache(name, config, core_id, type_id, memport, mfcreator, status,
//                    L2_WR_ALLOC_R, L2_WRBK_ACC, gpu) {}
//
//   virtual ~l2_cache() {}
//
//   virtual enum cache_request_status access(new_addr_type addr, mem_fetch *mf,
//                                            unsigned time,
//                                            std::list<cache_event> &events);
// };
//
