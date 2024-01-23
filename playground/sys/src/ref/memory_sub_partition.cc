#include "memory_sub_partition.hpp"

#include <iostream>
#include <list>

#include "io.hpp"
#include "mem_fetch.hpp"
#include "cache_sub_stats.hpp"
#include "l2_cache_trace.hpp"
#include "l2_interface.hpp"
#include "memory_stats.hpp"
#include "partition_mf_allocator.hpp"
#include "trace_gpgpu_sim.hpp"

memory_sub_partition::memory_sub_partition(unsigned sub_partition_id,
                                           const memory_config *config,
                                           class memory_stats_t *stats,
                                           class trace_gpgpu_sim *gpu) {
  logger = gpu->logger;
  m_id = sub_partition_id;
  m_config = config;
  m_stats = stats;
  m_gpu = gpu;
  m_memcpy_cycle_offset = 0;

  assert(m_id < m_config->m_n_mem_sub_partition);

  char L2c_name[32];
  snprintf(L2c_name, 32, "L2_bank_%03d", m_id);
  m_L2interface = new L2interface(this);
  m_mf_allocator = new partition_mf_allocator(config);

  if (!m_config->m_L2_config.disabled())
    m_L2cache =
        new l2_cache(L2c_name, m_config->m_L2_config, -1, -1, m_L2interface,
                     m_mf_allocator, IN_PARTITION_L2_MISS_QUEUE, logger, gpu);

  unsigned int icnt_L2;
  unsigned int L2_dram;
  unsigned int dram_L2;
  unsigned int L2_icnt;
  sscanf(m_config->gpgpu_L2_queue_config, "%u:%u:%u:%u", &icnt_L2, &L2_dram,
         &dram_L2, &L2_icnt);
  m_icnt_L2_queue = new fifo_pipeline<mem_fetch>("icnt-to-L2", 0, icnt_L2);
  m_L2_dram_queue = new fifo_pipeline<mem_fetch>("L2-to-dram", 0, L2_dram);
  m_dram_L2_queue = new fifo_pipeline<mem_fetch>("dram-to-L2", 0, dram_L2);
  m_L2_icnt_queue = new fifo_pipeline<mem_fetch>("L2-to-icnt", 0, L2_icnt);
  wb_addr = -1;
}

memory_sub_partition::~memory_sub_partition() {
  delete m_icnt_L2_queue;
  delete m_L2_dram_queue;
  delete m_dram_L2_queue;
  delete m_L2_icnt_queue;
  delete m_L2cache;
  delete m_L2interface;
}

std::ostream &operator<<(std::ostream &os, const rop_delay_t &delay) {
  os << delay.req;
  return os;
}

void memory_sub_partition::cache_cycle(unsigned cycle) {
  unsigned before = m_rop.size();
  logger->debug(
      "=> memory sub partition[{}] cache cycle {} rop queue=[{}] icnt to l2 "
      "queue=[{}] l2 to icnt queue=[{}] l2 to dram queue=[{}]",
      m_id, cycle, fmt::join(queue_to_vector(m_rop), ", "),
      fmt::join(m_icnt_L2_queue->to_vector(), ", "),
      fmt::join(m_L2_icnt_queue->to_vector(), ", "),
      fmt::join(m_L2_dram_queue->to_vector(), ", "));

  // make sure that printing the rop queue emptied a copy only
  assert(m_rop.size() == before);

  // L2 fill responses
  if (!m_config->m_L2_config.disabled()) {
    logger->debug(
        "=> memory sub partition[{}] cache cycle {} l2 cache ready "
        "accesses=[{}] l2 to icnt queue full={}",
        m_id, cycle, fmt::join(m_L2cache->ready_accesses(), ", "),
        m_L2_icnt_queue->full());
    if (m_L2cache->access_ready() && !m_L2_icnt_queue->full()) {
      mem_fetch *mf = m_L2cache->next_access();
      // l2 access is ready to go to interconnect
      if (mf->get_access_type() !=
          L2_WR_ALLOC_R) {  // Don't pass write allocate read request back
                            // to upper level cache
        mf->set_reply();
        mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                       m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        m_L2_icnt_queue->push(mf);
      } else {
        if (m_config->m_L2_config.m_write_alloc_policy == FETCH_ON_WRITE) {
          assert(0 && "fetch on write: l2 to icnt queue");
          mem_fetch *original_wr_mf = mf->get_original_wr_mf();
          assert(original_wr_mf);
          original_wr_mf->set_reply();
          original_wr_mf->set_status(
              IN_PARTITION_L2_TO_ICNT_QUEUE,
              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
          m_L2_icnt_queue->push(original_wr_mf);
        }
        m_request_tracker.erase(mf);
        delete mf;
      }
    }
  }

  // DRAM to L2 (texture) and icnt (not texture)
  if (!m_dram_L2_queue->empty()) {
    mem_fetch *mf = m_dram_L2_queue->top();
    if (!m_config->m_L2_config.disabled() && m_L2cache->waiting_for_fill(mf)) {
      if (m_L2cache->fill_port_free()) {
        mf->set_status(IN_PARTITION_L2_FILL_QUEUE,
                       m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);

        logger->debug("filling L2 with {}", mem_fetch_ptr(mf));
        m_L2cache->fill(mf, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                                m_memcpy_cycle_offset);
        m_dram_L2_queue->pop();
      } else {
        logger->debug("skip filling L2 with {}: no free fill port",
                      mem_fetch_ptr(mf));
      }
    } else if (!m_L2_icnt_queue->full()) {
      if (mf->is_write() && mf->get_type() == WRITE_ACK)
        mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                       m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      logger->debug("pushing {} to interconn queue", mem_fetch_ptr(mf));
      m_L2_icnt_queue->push(mf);
      m_dram_L2_queue->pop();
    } else {
      logger->debug(
          "skip pushing {} to interconn queue: l2 to interconn queue full",
          mem_fetch_ptr(mf));
    }
  }

  // prior L2 misses inserted into m_L2_dram_queue here
  if (!m_config->m_L2_config.disabled()) m_L2cache->cycle();

  // new L2 texture accesses and/or non-texture accesses
  if (!m_L2_dram_queue->full() && !m_icnt_L2_queue->empty()) {
    mem_fetch *mf = m_icnt_L2_queue->top();

    if (!m_config->m_L2_config.disabled() &&
        ((m_config->m_L2_texure_only && mf->istexture()) ||
         (!m_config->m_L2_texure_only))) {
      // L2 is enabled and access is for L2
      bool output_full = m_L2_icnt_queue->full();
      bool port_free = m_L2cache->data_port_free();
      if (!output_full && port_free) {
        std::list<cache_event> events;
        enum cache_request_status status =
            m_L2cache->access(mf->get_addr(), mf,
                              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                                  m_memcpy_cycle_offset,
                              events);
        bool write_sent = was_write_sent(events);
        bool read_sent = was_read_sent(events);
        logger->debug("probing L2 cache address={}, status={}", mf->get_addr(),
                      get_cache_request_status_str(status));

        if (status == HIT) {
          if (!write_sent) {
            // L2 cache replies
            assert(!read_sent);
            if (mf->get_access_type() == L1_WRBK_ACC) {
              m_request_tracker.erase(mf);
              delete mf;
            } else {
              mf->set_reply();
              mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                             m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
              m_L2_icnt_queue->push(mf);
            }
            m_icnt_L2_queue->pop();
          } else {
            assert(write_sent);
            m_icnt_L2_queue->pop();
          }
        } else if (status != RESERVATION_FAIL) {
          if (mf->is_write() &&
              (m_config->m_L2_config.m_write_alloc_policy == FETCH_ON_WRITE ||
               m_config->m_L2_config.m_write_alloc_policy ==
                   LAZY_FETCH_ON_READ) &&
              !was_writeallocate_sent(events)) {
            if (mf->get_access_type() == L1_WRBK_ACC) {
              m_request_tracker.erase(mf);
              delete mf;
            } else {
              // throw std::runtime_error("l2 to interconn queue push");
              mf->set_reply();
              mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                             m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
              m_L2_icnt_queue->push(mf);
            }
          }
          // L2 cache accepted request
          m_icnt_L2_queue->pop();
        } else {
          assert(!write_sent);
          assert(!read_sent);
          // L2 cache lock-up: will try again next cycle
        }
      }
    } else {
      // L2 is disabled or non-texture access to texture-only L2
      mf->set_status(IN_PARTITION_L2_TO_DRAM_QUEUE,
                     m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      throw std::runtime_error(
          "pushing from l2 to dram: l2 disabled or non "
          "texture access to texture only l2");
      m_L2_dram_queue->push(mf);
      m_icnt_L2_queue->pop();
    }
  }

  // ROP delay queue
  if (!m_rop.empty() && (cycle >= m_rop.front().ready_cycle) &&
      !m_icnt_L2_queue->full()) {
    mem_fetch *mf = m_rop.front().req;
    m_rop.pop();
    logger->debug("POP FROM ROP: {}", mem_fetch_ptr(mf));
    m_icnt_L2_queue->push(mf);
    mf->set_status(IN_PARTITION_ICNT_TO_L2_QUEUE,
                   m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
  }
}

// Interconn to L2
bool memory_sub_partition::full() const { return m_icnt_L2_queue->full(); }

// Interconn to L2
bool memory_sub_partition::full(unsigned size) const {
  return m_icnt_L2_queue->is_available_size(size);
}

// L2 to DRAM
bool memory_sub_partition::L2_dram_queue_empty() const {
  return m_L2_dram_queue->empty();
}

// L2 to DRAM
class mem_fetch *memory_sub_partition::L2_dram_queue_top() const {
  return m_L2_dram_queue->top();
}

// L2 to DRAM
void memory_sub_partition::L2_dram_queue_pop() { m_L2_dram_queue->pop(); }

// DRAM back to L2
bool memory_sub_partition::dram_L2_queue_full() const {
  return m_dram_L2_queue->full();
}

// DRAM back to L2
void memory_sub_partition::dram_L2_queue_push(class mem_fetch *mf) {
  assert(mf->is_reply());
  m_dram_L2_queue->push(mf);
}

void memory_sub_partition::print_cache_stat(unsigned &accesses,
                                            unsigned &misses) const {
  FILE *fp = stdout;
  if (!m_config->m_L2_config.disabled()) m_L2cache->print(fp, accesses, misses);
}

// void memory_sub_partition::print(FILE *fp) const {
//   if (!m_request_tracker.empty()) {
//     fprintf(fp, "Memory Sub Parition %u: %lu pending memory requests:\n",
//     m_id,
//             m_request_tracker.size());
//     for (std::set<mem_fetch *>::const_iterator r = m_request_tracker.begin();
//          r != m_request_tracker.end(); ++r) {
//       mem_fetch *mf = *r;
//       if (mf) {
//         // mf->print(fp);
//         std::stringstream buffer;
//         buffer << mf;
//         fprintf(fp, "%s", buffer.str().c_str());
//         // (std::ostream &)fp << mf;
//       } else {
//         fprintf(fp, " <NULL mem_fetch?>\n");
//       }
//     }
//   }
//   if (!m_config->m_L2_config.disabled()) m_L2cache->display_state(fp);
// }

unsigned memory_sub_partition::flushL2() {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->flush();
  }
  return 0;  // TODO: write the flushed data to the main memory
}

unsigned memory_sub_partition::invalidateL2() {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->invalidate();
  }
  return 0;
}

bool memory_sub_partition::busy() const { return !m_request_tracker.empty(); }

std::vector<mem_fetch *>
memory_sub_partition::breakdown_request_to_sector_requests(mem_fetch *mf) {
  logger->trace(
      "breakdown to sector requests for {} with data size {} sector "
      "mask={}",
      mem_fetch_ptr(mf), mf->get_data_size(),
      mask_to_string(mf->get_access_sector_mask()));

  std::vector<mem_fetch *> result;
  mem_access_sector_mask_t sector_mask = mf->get_access_sector_mask();
  if (mf->get_data_size() == SECTOR_SIZE &&
      mf->get_access_sector_mask().count() == 1) {
    result.push_back(mf);
  } else if (mf->get_data_size() == MAX_MEMORY_ACCESS_SIZE) {
    // break down every sector
    mem_access_byte_mask_t mask;
    for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; i++) {
      for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
        mask.set(k);
      }
      mem_fetch *n_mf = m_mf_allocator->alloc(
          mf->get_addr() + SECTOR_SIZE * i, mf->get_access_type(),
          mf->get_access_warp_mask(), mf->get_access_byte_mask() & mask,
          std::bitset<SECTOR_CHUNCK_SIZE>().set(i), SECTOR_SIZE, mf->is_write(),
          m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, mf->get_wid(),
          mf->get_sid(), mf->get_tpc(), mf);

      n_mf->set_alloc_start_addr(mf->get_alloc_start_addr());
      n_mf->set_alloc_id(mf->get_alloc_id());

      result.push_back(n_mf);
    }
    // This is for constant cache
  } else if (mf->get_data_size() == 64 &&
             (mf->get_access_sector_mask().all() ||
              mf->get_access_sector_mask().none())) {
    unsigned start;
    if (mf->get_addr() % MAX_MEMORY_ACCESS_SIZE == 0)
      start = 0;
    else
      start = 2;
    mem_access_byte_mask_t mask;
    for (unsigned i = start; i < start + 2; i++) {
      for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
        mask.set(k);
      }
      mem_fetch *n_mf = m_mf_allocator->alloc(
          mf->get_addr(), mf->get_access_type(), mf->get_access_warp_mask(),
          mf->get_access_byte_mask() & mask,
          std::bitset<SECTOR_CHUNCK_SIZE>().set(i), SECTOR_SIZE, mf->is_write(),
          m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, mf->get_wid(),
          mf->get_sid(), mf->get_tpc(), mf);

      n_mf->set_alloc_start_addr(mf->get_alloc_start_addr());
      n_mf->set_alloc_id(mf->get_alloc_id());

      result.push_back(n_mf);
    }
  } else {
    for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; i++) {
      if (sector_mask.test(i)) {
        mem_access_byte_mask_t mask;
        for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
          mask.set(k);
        }
        mem_fetch *n_mf = m_mf_allocator->alloc(
            mf->get_addr() + SECTOR_SIZE * i, mf->get_access_type(),
            mf->get_access_warp_mask(), mf->get_access_byte_mask() & mask,
            std::bitset<SECTOR_CHUNCK_SIZE>().set(i), SECTOR_SIZE,
            mf->is_write(), m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle,
            mf->get_wid(), mf->get_sid(), mf->get_tpc(), mf);

        n_mf->set_alloc_start_addr(mf->get_alloc_start_addr());
        n_mf->set_alloc_id(mf->get_alloc_id());

        result.push_back(n_mf);
      }
    }
  }
  logger->trace("sector requests for {}: [{}]", mem_fetch_ptr(mf),
                fmt::join(result, ","));
  if (result.size() == 0) assert(0 && "no mf sent");
  return result;
}

void memory_sub_partition::push(mem_fetch *m_req, unsigned long long cycle) {
  if (m_req) {
    m_stats->memlatstat_icnt2mem_pop(m_req);
    std::vector<mem_fetch *> reqs;
    if (m_config->m_L2_config.m_cache_type == SECTOR) {
      reqs = breakdown_request_to_sector_requests(m_req);
    } else {
      reqs.push_back(m_req);
    }

    assert(reqs.size() == 1);

    for (unsigned i = 0; i < reqs.size(); ++i) {
      mem_fetch *req = reqs[i];
      m_request_tracker.insert(req);
      if (req->istexture()) {
        m_icnt_L2_queue->push(req);
        req->set_status(IN_PARTITION_ICNT_TO_L2_QUEUE,
                        m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      } else {
        rop_delay_t r;
        r.req = req;
        r.ready_cycle = cycle + m_config->rop_latency;
        logger->debug("PUSH TO ROP: {}", mem_fetch_ptr(req));
        m_rop.push(r);
        req->set_status(IN_PARTITION_ROP_DELAY,
                        m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      }
    }
  }
}

mem_fetch *memory_sub_partition::pop() {
  mem_fetch *mf = m_L2_icnt_queue->pop();
  m_request_tracker.erase(mf);
  if (mf && mf->isatomic()) mf->do_atomic();
  if (mf && (mf->get_access_type() == L2_WRBK_ACC ||
             mf->get_access_type() == L1_WRBK_ACC)) {
    delete mf;
    mf = NULL;
  }
  return mf;
}

mem_fetch *memory_sub_partition::top() {
  mem_fetch *mf = m_L2_icnt_queue->top();
  if (mf && (mf->get_access_type() == L2_WRBK_ACC ||
             mf->get_access_type() == L1_WRBK_ACC)) {
    m_L2_icnt_queue->pop();
    m_request_tracker.erase(mf);
    delete mf;
    mf = NULL;
  }
  return mf;
}

void memory_sub_partition::set_done(mem_fetch *mf) {
  m_request_tracker.erase(mf);
}

void memory_sub_partition::accumulate_L2cache_stats(
    class cache_stats &l2_stats) const {
  if (!m_config->m_L2_config.disabled()) {
    l2_stats += m_L2cache->get_stats();
  }
}

void memory_sub_partition::get_L2cache_sub_stats(
    struct cache_sub_stats &css) const {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->get_sub_stats(css);
  }
}

void memory_sub_partition::get_L2cache_sub_stats_pw(
    struct cache_sub_stats_pw &css) const {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->get_sub_stats_pw(css);
  }
}

void memory_sub_partition::clear_L2cache_stats_pw() {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->clear_pw();
  }
}

// void memory_sub_partition::visualizer_print(gzFile visualizer_file) {
//   // Support for L2 AerialVision stats
//   // Per-sub-partition stats would be trivial to extend from this
//   cache_sub_stats_pw temp_sub_stats;
//   get_L2cache_sub_stats_pw(temp_sub_stats);
//
//   m_stats->L2_read_miss += temp_sub_stats.read_misses;
//   m_stats->L2_write_miss += temp_sub_stats.write_misses;
//   m_stats->L2_read_hit += temp_sub_stats.read_hits;
//   m_stats->L2_write_hit += temp_sub_stats.write_hits;
//
//   clear_L2cache_stats_pw();
// }
