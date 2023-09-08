#include "memory_partition_unit.hpp"

#include <iostream>
#include <iomanip>

#include "io.hpp"
#include "l2_cache_trace.hpp"
#include "memory_sub_partition.hpp"
#include "trace_gpgpu_sim.hpp"

memory_partition_unit::memory_partition_unit(unsigned partition_id,
                                             const memory_config *config,
                                             class memory_stats_t *stats,
                                             class trace_gpgpu_sim *gpu)
    : logger(gpu->logger),
      m_id(partition_id),
      m_config(config),
      m_stats(stats),
      m_arbitration_metadata(config, gpu->logger),
      m_gpu(gpu) {
  m_dram = new dram_t(m_id, m_config, m_stats, this, gpu);

  m_sub_partition = new memory_sub_partition
      *[m_config->m_n_sub_partition_per_memory_channel];
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    unsigned sub_partition_id =
        m_id * m_config->m_n_sub_partition_per_memory_channel + p;
    m_sub_partition[p] =
        new memory_sub_partition(sub_partition_id, m_config, stats, gpu);
  }
}

void memory_partition_unit::handle_memcpy_to_gpu(
    size_t addr, unsigned global_subpart_id, mem_access_sector_mask_t mask) {
  unsigned p = global_sub_partition_id_to_local_id(global_subpart_id);
  std::string mystring = mask.to_string<char, std::string::traits_type,
                                        std::string::allocator_type>();
  logger->trace(
      "copy engine request received for address={}, local_subpart={}, "
      "global_subpart={}, sector_mask={}",
      addr, p, global_subpart_id, mystring.c_str());

  MEMPART_DPRINTF(
      "Copy Engine Request Received For Address=%zx, local_subpart=%u, "
      "global_subpart=%u, sector_mask=%s \n",
      addr, p, global_subpart_id, mystring.c_str());
  m_sub_partition[p]->force_l2_tag_update(
      addr, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle, mask);
}

memory_partition_unit::~memory_partition_unit() {
  delete m_dram;
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    delete m_sub_partition[p];
  }
  delete[] m_sub_partition;
}

memory_partition_unit::arbitration_metadata::arbitration_metadata(
    const memory_config *config, std::shared_ptr<spdlog::logger> logger)
    : logger(logger),
      m_last_borrower(config->m_n_sub_partition_per_memory_channel - 1),
      m_private_credit(config->m_n_sub_partition_per_memory_channel, 0),
      m_shared_credit(0) {
  // each sub partition get at least 1 credit for forward progress
  // the rest is shared among with other partitions
  m_private_credit_limit = 1;
  m_shared_credit_limit = config->gpgpu_frfcfs_dram_sched_queue_size +
                          config->gpgpu_dram_return_queue_size -
                          (config->m_n_sub_partition_per_memory_channel - 1);
  if (config->seperate_write_queue_enabled)
    m_shared_credit_limit += config->gpgpu_frfcfs_dram_write_queue_size;
  if (config->gpgpu_frfcfs_dram_sched_queue_size == 0 or
      config->gpgpu_dram_return_queue_size == 0) {
    m_shared_credit_limit =
        0;  // no limit if either of the queue has no limit in size
  }
  assert(m_shared_credit_limit >= 0);
}

bool memory_partition_unit::arbitration_metadata::has_credits(
    int inner_sub_partition_id) const {
  int spid = inner_sub_partition_id;
  if (m_private_credit[spid] < m_private_credit_limit) {
    return true;
  } else if (m_shared_credit_limit == 0 ||
             m_shared_credit < m_shared_credit_limit) {
    return true;
  } else {
    return false;
  }
}

void memory_partition_unit::arbitration_metadata::borrow_credit(
    int inner_sub_partition_id) {
  int spid = inner_sub_partition_id;
  // int private_before = m_private_credit[spid];
  // int shared_before = m_shared_credit;
  if (m_private_credit[spid] < m_private_credit_limit) {
    m_private_credit[spid] += 1;
  } else if (m_shared_credit_limit == 0 ||
             m_shared_credit < m_shared_credit_limit) {
    m_shared_credit += 1;
  } else {
    assert(0 && "DRAM arbitration error: Borrowing from depleted credit!");
  }
  // logger->trace(
  //     "arbitration: borrow from spid {}: private credit={}/{} (was {}), "
  //     "shared_credit={}/{} (was {}), last borrower is now {}",
  //     spid, m_private_credit[spid], m_private_credit_limit, private_before,
  //     m_shared_credit, m_shared_credit_limit, shared_before, spid);

  m_last_borrower = spid;
}

void memory_partition_unit::arbitration_metadata::return_credit(
    int inner_sub_partition_id) {
  int spid = inner_sub_partition_id;
  // int private_before = m_private_credit[spid];
  // int shared_before = m_shared_credit;
  if (m_private_credit[spid] > 0) {
    m_private_credit[spid] -= 1;
  } else {
    m_shared_credit -= 1;
  }
  // logger->trace(
  //     "arbitration: return credit to spid {}: private credit={}/{} (was {}),
  //     " "shared_credit={}/{} (was {})", spid, m_private_credit[spid],
  //     m_private_credit_limit, private_before, m_shared_credit,
  //     m_shared_credit_limit, shared_before);

  assert((m_shared_credit >= 0) &&
         "DRAM arbitration error: Returning more than available credits!");
}

void memory_partition_unit::arbitration_metadata::print(FILE *fp) const {
  fprintf(fp, "private_credit = ");
  for (unsigned p = 0; p < m_private_credit.size(); p++) {
    fprintf(fp, "%d ", m_private_credit[p]);
  }
  fprintf(fp, "(limit = %d)\n", m_private_credit_limit);
  fprintf(fp, "shared_credit = %d (limit = %d)\n", m_shared_credit,
          m_shared_credit_limit);
}

bool memory_partition_unit::busy() const {
  bool busy = false;
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    if (m_sub_partition[p]->busy()) {
      busy = true;
    }
  }
  return busy;
}

void memory_partition_unit::cache_cycle(unsigned cycle) {
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    m_sub_partition[p]->cache_cycle(cycle);
  }
}

// void memory_partition_unit::visualizer_print(gzFile visualizer_file) const {
//   m_dram->visualizer_print(visualizer_file);
//   for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
//        p++) {
//     m_sub_partition[p]->visualizer_print(visualizer_file);
//   }
// }

// determine whether a given subpartition can issue to DRAM
bool memory_partition_unit::can_issue_to_dram(int inner_sub_partition_id) {
  int spid = inner_sub_partition_id;
  bool sub_partition_contention = m_sub_partition[spid]->dram_L2_queue_full();
  bool has_dram_resource = m_arbitration_metadata.has_credits(spid);

  MEMPART_DPRINTF(
      "sub partition %d sub_partition_contention=%c has_dram_resource=%c\n",
      spid, (sub_partition_contention) ? 'T' : 'F',
      (has_dram_resource) ? 'T' : 'F');

  return (has_dram_resource && !sub_partition_contention);
}

int memory_partition_unit::global_sub_partition_id_to_local_id(
    int global_sub_partition_id) const {
  return (global_sub_partition_id -
          m_id * m_config->m_n_sub_partition_per_memory_channel);
}

void memory_partition_unit::simple_dram_model_cycle() {
  // pop completed memory request from dram and push it to dram-to-L2 queue
  // of the original sub partition
  if (!m_dram_latency_queue.empty() &&
      ((m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) >=
       m_dram_latency_queue.front().ready_cycle)) {
    mem_fetch *mf_return = m_dram_latency_queue.front().req;

    if (mf_return->get_access_type() != L1_WRBK_ACC &&
        mf_return->get_access_type() != L2_WRBK_ACC) {
      // ROMAN: added this for compatibility, otherwise we do not get dram
      // metrics
      m_stats->memlatstat_dram_access(mf_return);

      mf_return->set_reply();
      logger->debug("got {} fetch return from dram latency queue (write={})",
                    mem_fetch_ptr(mf_return), mf_return->is_write());

      unsigned dest_global_spid = mf_return->get_sub_partition_id();
      int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
      assert(m_sub_partition[dest_spid]->get_id() == dest_global_spid);

      if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {
        if (mf_return->get_access_type() == L1_WRBK_ACC) {
          m_sub_partition[dest_spid]->set_done(mf_return);
          delete mf_return;
        } else {
          m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
          mf_return->set_status(
              IN_PARTITION_DRAM_TO_L2_QUEUE,
              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
          m_arbitration_metadata.return_credit(dest_spid);
          MEMPART_DPRINTF(
              "mem_fetch request %p return from dram to sub partition %d\n",
              mf_return, dest_spid);
        }
        m_dram_latency_queue.pop_front();
      } else {
        // throw std::runtime_error("fyi: simple dram model stall");
      }

    } else {
      logger->debug(
          "DROPPING {} fetch return from dram latency queue (write={})",
          mem_fetch_ptr(mf_return), mf_return->is_write());

      this->set_done(mf_return);
      delete mf_return;
      m_dram_latency_queue.pop_front();
    }
  }

  // mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
  // if( !m_dram->full(mf->is_write()) ) {
  // L2->DRAM queue to DRAM latency queue
  // Arbitrate among multiple L2 subpartitions
  int last_issued_partition = m_arbitration_metadata.last_borrower();
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    int spid = (p + last_issued_partition + 1) %
               m_config->m_n_sub_partition_per_memory_channel;

    logger->debug("checking sub partition[{}]:", spid);
    logger->debug("\t icnt to l2 queue = {}",
                  *(m_sub_partition[spid]->m_icnt_L2_queue));
    logger->debug("\t l2 to icnt queue = {}",
                  *(m_sub_partition[spid]->m_L2_icnt_queue));
    logger->debug("\t l2 to dram queue = {}",
                  *(m_sub_partition[spid]->m_L2_dram_queue));
    logger->debug("\t dram to l2 queue = {}",
                  *(m_sub_partition[spid]->m_dram_L2_queue));
    logger->debug("\t dram latency queue = ({:<3})[{}]",
                  m_dram_latency_queue.size(),
                  fmt::join(m_dram_latency_queue, ","));

    bool can_issue_to_dram_now = can_issue_to_dram(spid);
    logger->debug("\t can issue to dram={} dram to l2 queue full={}",
                  can_issue_to_dram_now,
                  m_sub_partition[spid]->dram_L2_queue_full());

    logger->debug("");

    if (!m_sub_partition[spid]->L2_dram_queue_empty() &&
        can_issue_to_dram_now) {
      mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
      if (m_dram->full(mf->is_write())) break;

      m_sub_partition[spid]->L2_dram_queue_pop();
      logger->debug("issue mem_fetch {} from sub partition {} to DRAM",
                    mem_fetch_ptr(mf), spid);
      dram_delay_t d;
      d.req = mf;
      d.ready_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                      m_config->dram_latency;
      m_dram_latency_queue.push_back(d);
      mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE,
                     m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      m_arbitration_metadata.borrow_credit(spid);
      break;  // the DRAM should only accept one request per cycle
    }
  }
  //}
}

void memory_partition_unit::dram_cycle() {
  // pop completed memory request from dram and push it to dram-to-L2 queue
  // of the original sub partition
  // WARN: NON BOX
  mem_fetch *mf_return = m_dram->return_queue_top();
  if (mf_return) {
    unsigned dest_global_spid = mf_return->get_sub_partition_id();
    int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
    assert(m_sub_partition[dest_spid]->get_id() == dest_global_spid);
    if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {
      if (mf_return->get_access_type() == L1_WRBK_ACC) {
        m_sub_partition[dest_spid]->set_done(mf_return);
        delete mf_return;
      } else {
        m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
        mf_return->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE,
                              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        m_arbitration_metadata.return_credit(dest_spid);
        MEMPART_DPRINTF(
            "mem_fetch request %p return from dram to sub partition %d\n",
            mf_return, dest_spid);
      }
      m_dram->return_queue_pop();
    }
  } else {
    m_dram->return_queue_pop();
  }

  m_dram->cycle();
  m_dram->dram_log(stdout, SAMPLELOG);

  // mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
  // if( !m_dram->full(mf->is_write()) ) {

  // L2->DRAM queue to DRAM latency queue
  // Arbitrate among multiple L2 subpartitions
  int last_issued_partition = m_arbitration_metadata.last_borrower();
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    int spid = (p + last_issued_partition + 1) %
               m_config->m_n_sub_partition_per_memory_channel;
    if (!m_sub_partition[spid]->L2_dram_queue_empty() &&
        can_issue_to_dram(spid)) {
      mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
      if (m_dram->full(mf->is_write())) break;

      m_sub_partition[spid]->L2_dram_queue_pop();
      MEMPART_DPRINTF(
          "Issue mem_fetch request %p from sub partition %d to dram\n", mf,
          spid);
      dram_delay_t d;
      d.req = mf;
      d.ready_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                      m_config->dram_latency;
      m_dram_latency_queue.push_back(d);
      mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE,
                     m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      m_arbitration_metadata.borrow_credit(spid);
      break;  // the DRAM should only accept one request per cycle
    }
  }
  //}

  if (!m_dram_latency_queue.empty() &&
      ((m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) >=
       m_dram_latency_queue.front().ready_cycle) &&
      !m_dram->full(m_dram_latency_queue.front().req->is_write())) {
    mem_fetch *mf = m_dram_latency_queue.front().req;
    m_dram_latency_queue.pop_front();
    m_dram->push(mf);
  }
}

void memory_partition_unit::set_done(mem_fetch *mf) {
  unsigned global_spid = mf->get_sub_partition_id();
  int spid = global_sub_partition_id_to_local_id(global_spid);
  assert(m_sub_partition[spid]->get_id() == global_spid);
  if (mf->get_access_type() == L1_WRBK_ACC ||
      mf->get_access_type() == L2_WRBK_ACC) {
    m_arbitration_metadata.return_credit(spid);
    logger->trace("mem_fetch request {} return from dram to sub partition {}",
                  mem_fetch_ptr(mf), spid);
  }
  m_sub_partition[spid]->set_done(mf);
}

void memory_partition_unit::set_dram_power_stats(
    unsigned &n_cmd, unsigned &n_activity, unsigned &n_nop, unsigned &n_act,
    unsigned &n_pre, unsigned &n_rd, unsigned &n_wr, unsigned &n_wr_WB,
    unsigned &n_req) const {
  m_dram->set_dram_power_stats(n_cmd, n_activity, n_nop, n_act, n_pre, n_rd,
                               n_wr, n_wr_WB, n_req);
}

// void memory_partition_unit::print(FILE *fp) const {
//   fprintf(fp, "Memory Partition %u: \n", m_id);
//   for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
//        p++) {
//     m_sub_partition[p]->print(fp);
//   }
//   fprintf(fp, "In Dram Latency Queue (total = %zd): \n",
//           m_dram_latency_queue.size());
//   for (std::list<dram_delay_t>::const_iterator mf_dlq =
//            m_dram_latency_queue.begin();
//        mf_dlq != m_dram_latency_queue.end(); ++mf_dlq) {
//     mem_fetch *mf = mf_dlq->req;
//     fprintf(fp, "Ready @ %llu - ", mf_dlq->ready_cycle);
//     if (mf) {
//       std::stringstream buffer;
//       buffer << mf;
//       fprintf(fp, "%s", buffer.str().c_str());
//
//       // (std::ostream &)fp << mf;
//       // mf->print(fp);
//     } else {
//       fprintf(fp, " <NULL mem_fetch?>\n");
//     }
//   }
//   m_dram->print(fp);
// }

std::ostream &operator<<(std::ostream &os,
                         const memory_partition_unit::dram_delay_t &delay) {
  os << delay.req;
  return os;
}
