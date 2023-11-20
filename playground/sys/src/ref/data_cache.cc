#include "data_cache.hpp"
#include "cache_reservation_fail_reason.hpp"
#include "trace_gpgpu_sim.hpp"

/// Sends write request to lower level memory (write or writeback)
void data_cache::send_write_request(mem_fetch *mf, cache_event request,
                                    unsigned time,
                                    std::list<cache_event> &events) {
  logger->debug("data_cache::send_write_request({})", mem_fetch_ptr(mf));
  events.push_back(request);
  m_miss_queue.push_back(mf);
  mf->set_status(m_miss_queue_status, time);
}

void data_cache::update_m_readable(mem_fetch *mf, unsigned cache_index) {
  cache_block_t *block = m_tag_array->get_block(cache_index);
  for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; i++) {
    if (mf->get_access_sector_mask().test(i)) {
      bool all_set = true;
      for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
        // If any bit in the byte mask (within the sector) is not set,
        // the sector is unreadble
        if (!block->get_dirty_byte_mask().test(k)) {
          all_set = false;
          break;
        }
      }
      if (all_set) block->set_m_readable(true, mf->get_access_sector_mask());
    }
  }
}

/****** Write-hit functions (Set by config file) ******/

/// Write-back hit: Mark block as modified
cache_request_status data_cache::wr_hit_wb(new_addr_type addr,
                                           unsigned cache_index, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events,
                                           enum cache_request_status status) {
  assert(addr == mf->get_addr());
  new_addr_type block_addr = m_config.block_addr(addr);
  logger->debug(
      "handling WRITE HIT WRITE BACK for {} (block_addr={}, cache_idx={})",
      mem_fetch_ptr(mf), block_addr, cache_index);

  m_tag_array->access(block_addr, time, cache_index, mf);  // update LRU state
  cache_block_t *block = m_tag_array->get_block(cache_index);
  if (!block->is_modified_line()) {
    m_tag_array->inc_dirty();
  }
  block->set_status(MODIFIED, mf->get_access_sector_mask());
  block->set_byte_mask(mf);
  update_m_readable(mf, cache_index);

  return HIT;
}

/// Write-through hit: Directly send request to lower level memory
cache_request_status data_cache::wr_hit_wt(new_addr_type addr,
                                           unsigned cache_index, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events,
                                           enum cache_request_status status) {
  if (miss_queue_full(0)) {
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
    return RESERVATION_FAIL;  // cannot handle request this cycle
  }

  new_addr_type block_addr = m_config.block_addr(addr);
  m_tag_array->access(block_addr, time, cache_index, mf);  // update LRU state
  cache_block_t *block = m_tag_array->get_block(cache_index);
  if (!block->is_modified_line()) {
    m_tag_array->inc_dirty();
  }
  block->set_status(MODIFIED, mf->get_access_sector_mask());
  block->set_byte_mask(mf);
  update_m_readable(mf, cache_index);

  // generate a write-through
  send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);

  return HIT;
}

/// Write-evict hit: Send request to lower level memory and invalidate
/// corresponding block
cache_request_status data_cache::wr_hit_we(new_addr_type addr,
                                           unsigned cache_index, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events,
                                           enum cache_request_status status) {
  if (miss_queue_full(0)) {
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
    return RESERVATION_FAIL;  // cannot handle request this cycle
  }

  // generate a write-through/evict
  cache_block_t *block = m_tag_array->get_block(cache_index);
  send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);

  // Invalidate block
  block->set_status(INVALID, mf->get_access_sector_mask());

  return HIT;
}

/// Global write-evict, local write-back: Useful for private caches
enum cache_request_status data_cache::wr_hit_global_we_local_wb(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  bool evict = (mf->get_access_type() ==
                GLOBAL_ACC_W);  // evict a line that hits on global memory write
  if (evict)
    return wr_hit_we(addr, cache_index, mf, time, events,
                     status);  // Write-evict
  else
    return wr_hit_wb(addr, cache_index, mf, time, events,
                     status);  // Write-back
}

/****** Write-miss functions (Set by config file) ******/

/// Write-allocate miss: Send write request to lower level memory
// and send a read request for the same block
enum cache_request_status data_cache::wr_miss_wa_naive(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  new_addr_type block_addr = m_config.block_addr(addr);
  new_addr_type mshr_addr = m_config.mshr_addr(mf->get_addr());

  assert(addr == mf->get_addr());

  // Write allocate, maximum 3 requests (write miss, read request, write back
  // request) Conservatively ensure the worst-case request can be handled this
  // cycle
  bool mshr_hit = m_mshrs.probe(mshr_addr);
  bool mshr_avail = !m_mshrs.full(mshr_addr);
  logger->debug(
      "handling write miss for {} (block addr={} mshr addr={}, mshr hit={} "
      "mshr avail={}, miss_queue_full={})",
      mem_fetch_ptr(mf), block_addr, mshr_addr, mshr_hit, mshr_avail,
      miss_queue_full(2));

  if (miss_queue_full(2) ||
      (!(mshr_hit && mshr_avail) &&
       !(!mshr_hit && mshr_avail &&
         (m_miss_queue.size() < m_config.m_miss_queue_size)))) {
    // check what is the exactly the failure reason
    if (miss_queue_full(2))
      m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
    else if (mshr_hit && !mshr_avail)
      m_stats.inc_fail_stats(mf->get_access_type(), MSHR_MERGE_ENRTY_FAIL);
    else if (!mshr_hit && !mshr_avail)
      m_stats.inc_fail_stats(mf->get_access_type(), MSHR_ENRTY_FAIL);
    else
      assert(0);

    logger->debug("handling write miss for {}: RESERVATION FAIL",
                  mem_fetch_ptr(mf));

    return RESERVATION_FAIL;
  }

  send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);
  // Tries to send write allocate request, returns true on success and false on
  // failure
  // if(!send_write_allocate(mf, addr, block_addr, cache_index, time, events))
  //    return RESERVATION_FAIL;

  mem_access_t *new_access =
      new mem_access_t(m_wr_alloc_type, mf->get_addr(), m_config.get_atom_sz(),
                       false,  // Now performing a read
                       mf->get_access_warp_mask(), mf->get_access_byte_mask(),
                       mf->get_access_sector_mask(), m_gpu->gpgpu_ctx);
  // keep the original alloc
  new_access->set_alloc_start_addr(mf->get_alloc_start_addr());
  new_access->set_alloc_id(mf->get_alloc_id());

  mem_fetch *new_fetch =
      new mem_fetch(*new_access, NULL, mf->get_ctrl_size(), mf->get_wid(),
                    mf->get_sid(), mf->get_tpc(), mf->get_mem_config(),
                    m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle);

  bool do_miss = false;
  bool wb = false;
  evicted_block_info evicted;

  // Send read request resulting from write miss
  send_read_request(addr, block_addr, cache_index, new_fetch, time, do_miss, wb,
                    evicted, events, false, true);

  events.push_back(cache_event(WRITE_ALLOCATE_SENT));

  if (do_miss) {
    // If evicted block is modified and not a write-through
    // (already modified lower level)
    logger->debug("evicted block: {}", evicted);
    // throw std::runtime_error("has evicted block");
    if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
      assert(status ==
             MISS);  // SECTOR_MISS and HIT_RESERVED should not send write back
      mem_fetch *wb = m_memfetch_creator->alloc(
          evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
          evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
          true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
          NULL);
      wb->set_alloc_id(evicted.m_allocation_id);
      wb->set_alloc_start_addr(evicted.m_allocation_start_addr);

      // the evicted block may have wrong chip id when advanced L2 hashing  is
      // used, so set the right chip address from the original mf
      wb->set_chip(mf->get_tlx_addr().chip);
      wb->set_partition(mf->get_tlx_addr().sub_partition);
      send_write_request(wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                         time, events);
    }
    return MISS;
  }

  return RESERVATION_FAIL;
}

enum cache_request_status data_cache::wr_miss_wa_fetch_on_write(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  new_addr_type block_addr = m_config.block_addr(addr);
  new_addr_type mshr_addr = m_config.mshr_addr(mf->get_addr());

  if (mf->get_access_byte_mask().count() == m_config.get_atom_sz()) {
    // if the request writes to the whole cache line/sector, then, write and set
    // cache line Modified. and no need to send read request to memory or
    // reserve mshr

    if (miss_queue_full(0)) {
      m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
      return RESERVATION_FAIL;  // cannot handle request this cycle
    }

    bool wb = false;
    evicted_block_info evicted;

    cache_request_status status =
        m_tag_array->access(block_addr, time, cache_index, wb, evicted, mf);
    assert(status != HIT);
    cache_block_t *block = m_tag_array->get_block(cache_index);
    if (!block->is_modified_line()) {
      m_tag_array->inc_dirty();
    }
    block->set_status(MODIFIED, mf->get_access_sector_mask());
    block->set_byte_mask(mf);
    if (status == HIT_RESERVED)
      block->set_ignore_on_fill(true, mf->get_access_sector_mask());

    if (status != RESERVATION_FAIL) {
      // If evicted block is modified and not a write-through
      // (already modified lower level)
      if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
        mem_fetch *wb = m_memfetch_creator->alloc(
            evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
            evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
            true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
            NULL);
        wb->set_alloc_id(evicted.m_allocation_id);
        wb->set_alloc_start_addr(evicted.m_allocation_start_addr);

        // the evicted block may have wrong chip id when advanced L2 hashing  is
        // used, so set the right chip address from the original mf
        wb->set_chip(mf->get_tlx_addr().chip);
        wb->set_partition(mf->get_tlx_addr().sub_partition);
        send_write_request(wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                           time, events);
      }
      return MISS;
    }
    return RESERVATION_FAIL;
  } else {
    bool mshr_hit = m_mshrs.probe(mshr_addr);
    bool mshr_avail = !m_mshrs.full(mshr_addr);
    if (miss_queue_full(1) ||
        (!(mshr_hit && mshr_avail) &&
         !(!mshr_hit && mshr_avail &&
           (m_miss_queue.size() < m_config.m_miss_queue_size)))) {
      // check what is the exactly the failure reason
      if (miss_queue_full(1))
        m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
      else if (mshr_hit && !mshr_avail)
        m_stats.inc_fail_stats(mf->get_access_type(), MSHR_MERGE_ENRTY_FAIL);
      else if (!mshr_hit && !mshr_avail)
        m_stats.inc_fail_stats(mf->get_access_type(), MSHR_ENRTY_FAIL);
      else
        assert(0);

      return RESERVATION_FAIL;
    }

    // prevent Write - Read - Write in pending mshr
    // allowing another write will override the value of the first write, and
    // the pending read request will read incorrect result from the second write
    if (m_mshrs.probe(mshr_addr) &&
        m_mshrs.is_read_after_write_pending(mshr_addr) && mf->is_write()) {
      // assert(0);
      m_stats.inc_fail_stats(mf->get_access_type(), MSHR_RW_PENDING);
      return RESERVATION_FAIL;
    }

    mem_access_t *new_access = new mem_access_t(
        m_wr_alloc_type, mf->get_addr(), m_config.get_atom_sz(),
        false,  // Now performing a read
        mf->get_access_warp_mask(), mf->get_access_byte_mask(),
        mf->get_access_sector_mask(), m_gpu->gpgpu_ctx);

    // keep the original alloc start addr
    new_access->set_alloc_start_addr(mf->get_alloc_start_addr());
    new_access->set_alloc_id(mf->get_alloc_id());

    mem_fetch *new_fetch = new mem_fetch(
        *new_access, NULL, mf->get_ctrl_size(), mf->get_wid(), mf->get_sid(),
        mf->get_tpc(), mf->get_mem_config(),
        m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, NULL, mf);

    new_addr_type block_addr = m_config.block_addr(addr);
    bool do_miss = false;
    bool wb = false;
    evicted_block_info evicted;
    send_read_request(addr, block_addr, cache_index, new_fetch, time, do_miss,
                      wb, evicted, events, false, true);

    cache_block_t *block = m_tag_array->get_block(cache_index);
    block->set_modified_on_fill(true, mf->get_access_sector_mask());
    block->set_byte_mask_on_fill(true);

    events.push_back(cache_event(WRITE_ALLOCATE_SENT));

    if (do_miss) {
      // If evicted block is modified and not a write-through
      // (already modified lower level)
      if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
        mem_fetch *wb = m_memfetch_creator->alloc(
            evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
            evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
            true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
            NULL);
        wb->set_alloc_id(evicted.m_allocation_id);
        wb->set_alloc_start_addr(evicted.m_allocation_start_addr);

        // the evicted block may have wrong chip id when advanced L2 hashing  is
        // used, so set the right chip address from the original mf
        wb->set_chip(mf->get_tlx_addr().chip);
        wb->set_partition(mf->get_tlx_addr().sub_partition);
        send_write_request(wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                           time, events);
      }
      return MISS;
    }
    return RESERVATION_FAIL;
  }
}

enum cache_request_status data_cache::wr_miss_wa_lazy_fetch_on_read(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  new_addr_type block_addr = m_config.block_addr(addr);

  // if the request writes to the whole cache line/sector, then, write and set
  // cache line Modified. and no need to send read request to memory or reserve
  // mshr

  if (miss_queue_full(0)) {
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
    return RESERVATION_FAIL;  // cannot handle request this cycle
  }

  if (m_config.m_write_policy == WRITE_THROUGH) {
    send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);
  }

  bool wb = false;
  evicted_block_info evicted;

  cache_request_status m_status =
      m_tag_array->access(block_addr, time, cache_index, wb, evicted, mf);
  assert(m_status != HIT);
  cache_block_t *block = m_tag_array->get_block(cache_index);
  if (!block->is_modified_line()) {
    m_tag_array->inc_dirty();
  }
  block->set_status(MODIFIED, mf->get_access_sector_mask());
  block->set_byte_mask(mf);
  if (m_status == HIT_RESERVED) {
    block->set_ignore_on_fill(true, mf->get_access_sector_mask());
    block->set_modified_on_fill(true, mf->get_access_sector_mask());
    block->set_byte_mask_on_fill(true);
  }

  if (mf->get_access_byte_mask().count() == m_config.get_atom_sz()) {
    block->set_m_readable(true, mf->get_access_sector_mask());
  } else {
    block->set_m_readable(false, mf->get_access_sector_mask());
    if (m_status == HIT_RESERVED)
      block->set_readable_on_fill(true, mf->get_access_sector_mask());
  }
  update_m_readable(mf, cache_index);

  if (m_status != RESERVATION_FAIL) {
    // If evicted block is modified and not a write-through
    // (already modified lower level)
    if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
      mem_fetch *wb = m_memfetch_creator->alloc(
          evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
          evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
          true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
          NULL);
      wb->set_alloc_id(evicted.m_allocation_id);
      wb->set_alloc_start_addr(evicted.m_allocation_start_addr);

      // the evicted block may have wrong chip id when advanced L2 hashing  is
      // used, so set the right chip address from the original mf
      wb->set_chip(mf->get_tlx_addr().chip);
      wb->set_partition(mf->get_tlx_addr().sub_partition);
      send_write_request(wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                         time, events);
    }
    return MISS;
  }
  return RESERVATION_FAIL;
}

/// No write-allocate miss: Simply send write request to lower level memory
enum cache_request_status data_cache::wr_miss_no_wa(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  assert(addr == mf->get_addr());
  logger->debug(
      "handling WRITE MISS NO WRITE ALLOCATE for {} (miss_queue_full={})",
      mem_fetch_ptr(mf), miss_queue_full(0));

  if (miss_queue_full(0)) {
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
    return RESERVATION_FAIL;  // cannot handle request this cycle
  }

  // on miss, generate write through (no write buffering -- too many threads for
  // that)
  send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);

  return MISS;
}

/****** Read hit functions (Set by config file) ******/

/// Baseline read hit: Update LRU status of block.
// Special case for atomic instructions -> Mark block as modified
enum cache_request_status data_cache::rd_hit_base(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  new_addr_type block_addr = m_config.block_addr(addr);
  m_tag_array->access(block_addr, time, cache_index, mf);
  // Atomics treated as global read/write requests - Perform read, mark line as
  // MODIFIED
  if (mf->isatomic()) {
    assert(mf->get_access_type() == GLOBAL_ACC_R);
    cache_block_t *block = m_tag_array->get_block(cache_index);
    if (!block->is_modified_line()) {
      m_tag_array->inc_dirty();
    }
    block->set_status(MODIFIED,
                      mf->get_access_sector_mask());  // mark line as
    block->set_byte_mask(mf);
  }
  return HIT;
}

/****** Read miss functions (Set by config file) ******/

/// Baseline read miss: Send read request to lower level memory,
// perform write-back as necessary
enum cache_request_status data_cache::rd_miss_base(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
  if (miss_queue_full(1)) {
    // cannot handle request this cycle
    // (might need to generate two requests)
    m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL);
    return RESERVATION_FAIL;
  }

  new_addr_type block_addr = m_config.block_addr(addr);
  bool do_miss = false;
  bool wb = false;
  evicted_block_info evicted;
  send_read_request(addr, block_addr, cache_index, mf, time, do_miss, wb,
                    evicted, events, false, false);

  logger->trace(
      "handling READ MISS for {} (should miss={}, writeback={}, writeback "
      "policy={}, events: [{}])",
      mem_fetch_ptr(mf), do_miss, wb,
      write_policy_t_str[m_config.m_write_policy], fmt::join(events, ", "));

  if (do_miss) {
    // If evicted block is modified and not a write-through
    // (already modified lower level)
    if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
      mem_fetch *wb = m_memfetch_creator->alloc(
          evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
          evicted.m_byte_mask, evicted.m_sector_mask, evicted.m_modified_size,
          true, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
          NULL);
      wb->set_alloc_id(evicted.m_allocation_id);
      wb->set_alloc_start_addr(evicted.m_allocation_start_addr);
      // logger->trace("evicted: allocation start addr={}",
      //               evicted.m_allocation_start_addr);

      // the evicted block may have wrong chip id when advanced L2 hashing  is
      // used, so set the right chip address from the original mf
      wb->set_chip(mf->get_tlx_addr().chip);
      wb->set_partition(mf->get_tlx_addr().sub_partition);
      logger->trace("handling READ MISS for {}: => sending writeback {}",
                    mem_fetch_ptr(mf), mem_fetch_ptr(wb));

      send_write_request(wb, WRITE_BACK_REQUEST_SENT, time, events);
    }
    return MISS;
  }
  return RESERVATION_FAIL;
}

//! A general function that takes the result of a tag_array probe
//  and performs the correspding functions based on the cache configuration
//  The access fucntion calls this function
enum cache_request_status data_cache::process_tag_probe(
    bool wr, enum cache_request_status probe_status, new_addr_type addr,
    unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events) {
  // Each function pointer ( m_[rd/wr]_[hit/miss] ) is set in the
  // data_cache constructor to reflect the corresponding cache configuration
  // options. Function pointers were used to avoid many long conditional
  // branches resulting from many cache configuration options.
  cache_request_status access_status = probe_status;
  if (wr) {  // Write
    if (probe_status == HIT) {
      access_status =
          (this->*m_wr_hit)(addr, cache_index, mf, time, events, probe_status);
    } else if ((probe_status != RESERVATION_FAIL) ||
               (probe_status == RESERVATION_FAIL &&
                m_config.m_write_alloc_policy == NO_WRITE_ALLOCATE)) {
      access_status =
          (this->*m_wr_miss)(addr, cache_index, mf, time, events, probe_status);
    } else {
      // the only reason for reservation fail here is LINE_ALLOC_FAIL (i.e all
      // lines are reserved)
      m_stats.inc_fail_stats(mf->get_access_type(), LINE_ALLOC_FAIL);
    }
  } else {  // Read
    if (probe_status == HIT) {
      access_status =
          (this->*m_rd_hit)(addr, cache_index, mf, time, events, probe_status);
    } else if (probe_status != RESERVATION_FAIL) {
      access_status =
          (this->*m_rd_miss)(addr, cache_index, mf, time, events, probe_status);
    } else {
      // the only reason for reservation fail here is LINE_ALLOC_FAIL (i.e all
      // lines are reserved)
      m_stats.inc_fail_stats(mf->get_access_type(), LINE_ALLOC_FAIL);
    }
  }

  m_bandwidth_management.use_data_port(mf, access_status, events);
  return access_status;
}

// Both the L1 and L2 currently use the same access function.
// Differentiation between the two caches is done through configuration
// of caching policies.
// Both the L1 and L2 override this function to provide a means of
// performing actions specific to each cache when such actions are implemnted.
enum cache_request_status data_cache::access(new_addr_type addr, mem_fetch *mf,
                                             unsigned time,
                                             std::list<cache_event> &events) {
  // ROMAN: just checking
  assert(addr == mf->get_addr());

  assert(mf->get_data_size() <= m_config.get_atom_sz());
  bool wr = mf->get_is_write();
  new_addr_type block_addr = m_config.block_addr(addr);
  unsigned cache_index = (unsigned)-1;

  logger->debug(
      "data_cache::access({}, write = {}, size = {}, block = {}, time = {})",
      mem_fetch_ptr(mf), wr, mf->get_data_size(), block_addr, time);
  enum cache_request_status probe_status =
      m_tag_array->probe(block_addr, cache_index, mf, mf->is_write(), true);

  enum cache_request_status access_status =
      process_tag_probe(wr, probe_status, addr, cache_index, mf, time, events);
  logger->debug("data_cache::access({}) => probe status={} access status={}",
                mem_fetch_ptr(mf), cache_request_status_str[probe_status],
                cache_request_status_str[access_status]);

  if (m_gpu->gpgpu_ctx->accelsim_compat_mode) {
    // printf(
    //     "data cache access(%lu) time=%d cache index=%d block addr=%lu probe "
    //     "status=%s access status=%s\n",
    //     addr, time, cache_index, block_addr,
    //     cache_request_status_str[probe_status],
    //     cache_request_status_str[access_status]);
  }

  if (mf != NULL) {
    const warp_inst_t inst = mf->get_inst();
    // fmt::println(
    //     "{} fetch {} [{}] addr={} probe status={} access status={}", name(),
    //     mem_fetch_ptr(mf), _memory_space_t_str[inst.space.get_type()],
    //     mf->get_relative_byte_addr(), cache_request_status_str[probe_status],
    //     cache_request_status_str[access_status]);
  }
  m_stats.inc_stats(mf->get_access_type(),
                    m_stats.select_stats_status(probe_status, access_status));
  m_stats.inc_stats_pw(mf->get_access_type(), m_stats.select_stats_status(
                                                  probe_status, access_status));
  return access_status;
}
