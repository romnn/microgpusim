#[derive(Debug)]
pub struct ReadOnly {}

impl ReadOnly {
    pub fn new() -> Self {
        Self {}
    }

    pub fn access_ready(&self) -> bool {
        todo!("readonly: access_ready");
        false
    }
}

// #[derive(Debug)]
// pub struct Readonly<I> {
//     inner: Data,
//     interconn: I,
// }
//
// impl<I> cache::Cache for Readonly<I> {
//     /// Access read only cache.
//     ///
//     /// returns RequestStatus::RESERVATION_FAIL if request could not be
//     /// accepted (for any reason)
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
//         todo!("readonly: fill");
//     }
//
//     fn has_free_fill_port(&self) -> bool {
//         todo!("readonly: has_free_fill_port");
//         false
//     }
// }

// read_only_cache(const char *name, cache_config &config, int core_id,
//                   int type_id, mem_fetch_interface *memport,
//                   enum mem_fetch_status status)
//       : baseline_cache(name, config, core_id, type_id, memport, status)
