#include "trace_simt_core_cluster.hpp"

#include "cache_sub_stats.hpp"
#include "icnt_wrapper.hpp"
#include "memory_stats.hpp"
#include "shader_core_config.hpp"
#include "trace_gpgpu_sim.hpp"
#include "trace_shader_core_ctx.hpp"

void trace_simt_core_cluster::create_shader_core_ctx() {
  m_core = new trace_shader_core_ctx *[m_config->n_simt_cores_per_cluster];
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++) {
    unsigned sid = m_config->cid_to_sid(i, m_cluster_id);
    m_core[i] = new trace_shader_core_ctx(m_gpu, this, sid, m_cluster_id,
                                          m_config, m_mem_config, m_stats);
    m_core_sim_order.push_back(i);
  }
}

unsigned trace_simt_core_cluster::get_not_completed() const {
  unsigned not_completed = 0;
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++)
    not_completed += m_core[i]->get_not_completed();
  return not_completed;
}

void trace_simt_core_cluster::icnt_cycle() {
  printf("icnt_cycle response buffer size=%lu\n", m_response_fifo.size());

  if (!m_response_fifo.empty()) {
    mem_fetch *mf = m_response_fifo.front();
    unsigned cid = m_config->sid_to_cid(mf->get_sid());
    if (mf->get_access_type() == INST_ACC_R) {
      // instruction fetch response
      if (!m_core[cid]->fetch_unit_response_buffer_full()) {
        m_response_fifo.pop_front();
        m_core[cid]->accept_fetch_response(mf);
        printf("accepted instr access fetch ");
        mf->print(stdout);
        printf("\n");

      } else {
        printf("instr access fetch ");
        mf->print(stdout);
        printf(" NOT YET ACCEPTED\n");
      }

    } else {
      // data response
      if (!m_core[cid]->ldst_unit_response_buffer_full()) {
        m_response_fifo.pop_front();
        m_memory_stats->memlatstat_read_done(mf);
        m_core[cid]->accept_ldst_unit_response(mf);
        printf("accepted ldst unit fetch ");
        mf->print(stdout);
        printf("\n");

      } else {
        printf("ldst unit fetch ");
        mf->print(stdout);
        printf(" NOT YET ACCEPTED\n");
      }
    }
  }
  if (m_response_fifo.size() < m_config->n_simt_ejection_buffer_size) {
    mem_fetch *mf = (mem_fetch *)::icnt_pop(m_cluster_id);
    if (!mf)
      return;

    printf(" \e[0;36m cluster::icnt_cycle() got fetch from interconn: ");
    mf->print(stdout);
    printf(" \e[0m \n");

    assert(mf->get_tpc() == m_cluster_id);
    assert(mf->get_type() == READ_REPLY || mf->get_type() == WRITE_ACK);
    // throw std::runtime_error("got the first fetch back");

    // The packet size varies depending on the type of request:
    // - For read request and atomic request, the packet contains the data
    // - For write-ack, the packet only has control metadata
    unsigned int packet_size =
        (mf->get_is_write()) ? mf->get_ctrl_size() : mf->size();
    m_stats->m_incoming_traffic_stats->record_traffic(mf, packet_size);
    mf->set_status(IN_CLUSTER_TO_SHADER_QUEUE,
                   m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
    // m_memory_stats->memlatstat_read_done(mf,m_shader_config->max_warps_per_shader);
    m_response_fifo.push_back(mf);
    m_stats->n_mem_to_simt[m_cluster_id] += mf->get_num_flits(false);
  } else {
    printf("skip: ejection buffer full (%lu/%u)", m_response_fifo.size(),
           m_config->n_simt_ejection_buffer_size);
  }
}

void trace_simt_core_cluster::core_cycle() {
  for (std::list<unsigned>::iterator it = m_core_sim_order.begin();
       it != m_core_sim_order.end(); ++it) {
    m_core[*it]->cycle();
  }

  if (m_config->simt_core_sim_order == 1) {
    m_core_sim_order.splice(m_core_sim_order.end(), m_core_sim_order,
                            m_core_sim_order.begin());
  }
}

unsigned trace_simt_core_cluster::get_n_active_sms() const {
  unsigned n = 0;
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++)
    n += m_core[i]->isactive();
  return n;
}

void trace_simt_core_cluster::cache_invalidate() {
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++)
    m_core[i]->cache_invalidate();
}

float trace_simt_core_cluster::get_current_occupancy(
    unsigned long long &active, unsigned long long &total) const {
  float aggregate = 0.f;
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++) {
    aggregate += m_core[i]->get_current_occupancy(active, total);
  }
  return aggregate / m_config->n_simt_cores_per_cluster;
}

void trace_simt_core_cluster::print_not_completed(FILE *fp) const {
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++) {
    unsigned not_completed = m_core[i]->get_not_completed();
    unsigned sid = m_config->cid_to_sid(i, m_cluster_id);
    fprintf(fp, "%u(%u) ", sid, not_completed);
  }
}

void trace_simt_core_cluster::get_cache_stats(cache_stats &cs) const {
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i) {
    m_core[i]->get_cache_stats(cs);
  }
}

void trace_simt_core_cluster::reinit() {
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++)
    m_core[i]->reinit(0, m_config->n_thread_per_shader, true);
}

void trace_simt_core_cluster::get_icnt_stats(long &n_simt_to_mem,
                                             long &n_mem_to_simt) const {
  long simt_to_mem = 0;
  long mem_to_simt = 0;
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i) {
    m_core[i]->get_icnt_power_stats(simt_to_mem, mem_to_simt);
  }
  n_simt_to_mem = simt_to_mem;
  n_mem_to_simt = mem_to_simt;
}

unsigned trace_simt_core_cluster::issue_block2core() {
  unsigned num_blocks_issued = 0;
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++) {
    unsigned core =
        (i + m_cta_issue_next_core + 1) % m_config->n_simt_cores_per_cluster;

    trace_kernel_info_t *kernel;
    // Jin: fetch kernel according to concurrent kernel setting
    if (m_config->gpgpu_concurrent_kernel_sm) { // concurrent kernel on sm
      // always select latest issued kernel
      trace_kernel_info_t *k = m_gpu->select_kernel();
      kernel = k;
    } else {
      // first select core kernel, if no more cta, get a new kernel
      // only when core completes
      kernel = m_core[core]->get_kernel();
      if (!m_gpu->kernel_more_cta_left(kernel)) {
        // wait till current kernel finishes
        if (m_core[core]->get_not_completed() == 0) {
          trace_kernel_info_t *k = m_gpu->select_kernel();
          if (k)
            m_core[core]->set_kernel(k);
          kernel = k;
        }
      }
    }

    if (m_gpu->kernel_more_cta_left(kernel) &&
        //            (m_core[core]->get_n_active_cta() <
        //            m_config->max_cta(*kernel)) ) {
        m_core[core]->can_issue_1block(*kernel)) {
      m_core[core]->issue_block2core(*kernel);
      num_blocks_issued++;
      m_cta_issue_next_core = core;
      break;
    }
  }
  return num_blocks_issued;
}

void trace_simt_core_cluster::get_L1I_sub_stats(
    struct cache_sub_stats &css) const {
  struct cache_sub_stats temp_css;
  struct cache_sub_stats total_css;
  temp_css.clear();
  total_css.clear();
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i) {
    m_core[i]->get_L1I_sub_stats(temp_css);
    total_css += temp_css;
  }
  css = total_css;
}

void trace_simt_core_cluster::get_L1D_sub_stats(
    struct cache_sub_stats &css) const {
  struct cache_sub_stats temp_css;
  struct cache_sub_stats total_css;
  temp_css.clear();
  total_css.clear();
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i) {
    m_core[i]->get_L1D_sub_stats(temp_css);
    total_css += temp_css;
  }
  css = total_css;
}

void trace_simt_core_cluster::get_L1C_sub_stats(
    struct cache_sub_stats &css) const {
  struct cache_sub_stats temp_css;
  struct cache_sub_stats total_css;
  temp_css.clear();
  total_css.clear();
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i) {
    m_core[i]->get_L1C_sub_stats(temp_css);
    total_css += temp_css;
  }
  css = total_css;
}

void trace_simt_core_cluster::get_L1T_sub_stats(
    struct cache_sub_stats &css) const {
  struct cache_sub_stats temp_css;
  struct cache_sub_stats total_css;
  temp_css.clear();
  total_css.clear();
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i) {
    m_core[i]->get_L1T_sub_stats(temp_css);
    total_css += temp_css;
  }
  css = total_css;
}

void trace_simt_core_cluster::print_cache_stats(FILE *fp,
                                                unsigned &dl1_accesses,
                                                unsigned &dl1_misses) const {
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i) {
    m_core[i]->print_cache_stats(fp, dl1_accesses, dl1_misses);
  }
}

bool trace_simt_core_cluster::icnt_injection_buffer_full(unsigned size,
                                                         bool write) {
  unsigned request_size = size;
  if (!write)
    request_size = READ_PACKET_SIZE;
  return !::icnt_has_buffer(m_cluster_id, request_size);
}

void trace_simt_core_cluster::icnt_inject_request_packet(class mem_fetch *mf) {

  // stats
  if (mf->get_is_write())
    m_stats->made_write_mfs++;
  else
    m_stats->made_read_mfs++;
  switch (mf->get_access_type()) {
  case CONST_ACC_R:
    m_stats->gpgpu_n_mem_const++;
    break;
  case TEXTURE_ACC_R:
    m_stats->gpgpu_n_mem_texture++;
    break;
  case GLOBAL_ACC_R:
    m_stats->gpgpu_n_mem_read_global++;
    break;
  // case GLOBAL_ACC_R: m_stats->gpgpu_n_mem_read_global++;
  // printf("read_global%d\n",m_stats->gpgpu_n_mem_read_global); break;
  case GLOBAL_ACC_W:
    m_stats->gpgpu_n_mem_write_global++;
    break;
  case LOCAL_ACC_R:
    m_stats->gpgpu_n_mem_read_local++;
    break;
  case LOCAL_ACC_W:
    m_stats->gpgpu_n_mem_write_local++;
    break;
  case INST_ACC_R:
    m_stats->gpgpu_n_mem_read_inst++;
    break;
  case L1_WRBK_ACC:
    m_stats->gpgpu_n_mem_write_global++;
    break;
  case L2_WRBK_ACC:
    m_stats->gpgpu_n_mem_l2_writeback++;
    break;
  case L1_WR_ALLOC_R:
    m_stats->gpgpu_n_mem_l1_write_allocate++;
    break;
  case L2_WR_ALLOC_R:
    m_stats->gpgpu_n_mem_l2_write_allocate++;
    break;
  default:
    assert(0);
  }

  // The packet size varies depending on the type of request:
  // - For write request and atomic request, the packet contains the data
  // - For read request (i.e. not write nor atomic), the packet only has control
  // metadata
  unsigned int packet_size = mf->size();
  if (!mf->get_is_write() && !mf->isatomic()) {
    packet_size = mf->get_ctrl_size();
  }
  m_stats->m_outgoing_traffic_stats->record_traffic(mf, packet_size);
  unsigned sub_partition_id = mf->get_sub_partition_id();
  unsigned destination = m_config->mem2device(sub_partition_id);

  printf("cluster %u icnt_inject_request_packet(", m_cluster_id);
  mf->print(stdout);
  printf(") sub partition id=%d dest mem node=%d\n", sub_partition_id,
         destination);
  printf("raw addr:\t\t");
  mf->get_tlx_addr().print_dec(stdout);
  printf("\n");

  addrdec_t fresh_raw_adrr;
  m_mem_config->m_address_mapping.addrdec_tlx(mf->get_addr(), &fresh_raw_adrr);
  printf("fresh raw addr:\t\t");
  fresh_raw_adrr.print_dec(stdout);
  printf("\n");

  mf->set_status(IN_ICNT_TO_MEM,
                 m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
  if (!mf->get_is_write() && !mf->isatomic())
    ::icnt_push(m_cluster_id, destination, (void *)mf, mf->get_ctrl_size());
  else
    ::icnt_push(m_cluster_id, destination, (void *)mf, mf->size());
}
