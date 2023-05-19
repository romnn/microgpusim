use crate::ported::cache;

/// Baseline cache
/// Implements common functions for read_only_cache and data_cache
/// Each subclass implements its own 'access' function
#[derive(Debug)]
pub struct Baseline {
    // cache_config &m_config;
    // tag_array *m_tag_array;
    // mshr_table m_mshrs;
    // std::list<mem_fetch *> m_miss_queue;
    // enum mem_fetch_status m_miss_queue_status;
    // mem_fetch_interface *m_memport;
}

impl Baseline {
    pub fn new() -> Self {
        //     m_config(config),
        //     m_tag_array(new tag_array(config, core_id, type_id)),
        //     m_mshrs(config.m_mshr_entries, config.m_mshr_max_merge),
        //     m_bandwidth_management(config) {
        //
        //     m_name = name;
        // debug_assert!(config.m_mshr_type == ASSOC || config.m_mshr_type == SECTOR_ASSOC);
        // m_memport = memport;
        // m_miss_queue_status = status;
        Self {}
    }
}

impl cache::Component for Baseline {
    /// Sends next request to lower level of memory
    fn cycle(&mut self) {
        println!("baseline cache: cycle");
        dbg!(&self.miss_queue.len());
        if let Some(fetch) = self.miss_queue.front() {
            dbg!(&fetch);
            if !self.mem_port.full(fetch.data_size, fetch.is_write()) {
                if let Some(fetch) = self.miss_queue.pop_front() {
                    self.mem_port.push(fetch);
                }
            }
        }
        // bool data_port_busy = !m_bandwidth_management.data_port_free();
        // bool fill_port_busy = !m_bandwidth_management.fill_port_free();
        // m_stats.sample_cache_port_utility(data_port_busy, fill_port_busy);
        // m_bandwidth_management.replenish_port_bandwidth();
    }
}

impl cache::Cache for Baseline {
    /// Interface for response from lower memory level.
    ///
    /// bandwidth restictions should be modeled in the caller.
    fn fill(&self, fetch: &mem_fetch::MemFetch) {
        if self.cache_config.mshr_kind == config::MshrKind::SECTOR_ASSOC {
            // debug_assert!(fetch.get_original_mf());
            // extra_mf_fields_lookup::iterator e =
            //     m_extra_mf_fields.find(mf->get_original_mf());
            // assert(e != m_extra_mf_fields.end());
            // e->second.pending_read--;
            //
            // if (e->second.pending_read > 0) {
            //   // wait for the other requests to come back
            //   delete mf;
            //   return;
            // } else {
            //   mem_fetch *temp = mf;
            //   mf = mf->get_original_mf();
            //   delete temp;
            // }
        }
        // extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
        //   assert(e != m_extra_mf_fields.end());
        //   assert(e->second.m_valid);
        //   mf->set_data_size(e->second.m_data_size);
        //   mf->set_addr(e->second.m_addr);
        //   if (m_config.m_alloc_policy == ON_MISS)
        //     m_tag_array->fill(e->second.m_cache_index, time, mf);
        //   else if (m_config.m_alloc_policy == ON_FILL) {
        //     m_tag_array->fill(e->second.m_block_addr, time, mf, mf->is_write());
        //   } else
        //     abort();
        //   bool has_atomic = false;
        //   m_mshrs.mark_ready(e->second.m_block_addr, has_atomic);
        //   if (has_atomic) {
        //     assert(m_config.m_alloc_policy == ON_MISS);
        //     cache_block_t *block = m_tag_array->get_block(e->second.m_cache_index);
        //     if (!block->is_modified_line()) {
        //       m_tag_array->inc_dirty();
        //     }
        //     block->set_status(MODIFIED,
        //                       mf->get_access_sector_mask());  // mark line as dirty for
        //                                                       // atomic operation
        //     block->set_byte_mask(mf);
        //   }
        //   m_extra_mf_fields.erase(mf);
        //   m_bandwidth_management.use_fill_port(mf);
        todo!("l1: fill");
    }
}

// impl cache::CacheBandwidth for Baseline {
//     fn has_free_data_port(&self) -> bool {
//         self.bandwidth_management.has_free_data_port()
//     }
//
//     fn has_free_fill_port(&self) -> bool {
//         self.bandwidth_management.has_free_fill_port()
//     }
// }
