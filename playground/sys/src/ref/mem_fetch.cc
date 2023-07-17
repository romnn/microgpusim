#include "mem_fetch.hpp"

unsigned mem_fetch::sm_next_mf_request_uid = 1;

mem_fetch::mem_fetch(const mem_access_t &access, const warp_inst_t *inst,
                     unsigned ctrl_size, unsigned wid, unsigned sid,
                     unsigned tpc, const memory_config *config,
                     unsigned long long cycle, mem_fetch *m_original_mf,
                     mem_fetch *m_original_wr_mf)
    : m_access(access)

{
  m_request_uid = sm_next_mf_request_uid++;
  m_access = access;
  if (inst) {
    m_inst = *inst;
    assert(wid == m_inst.warp_id());
  }
  m_data_size = access.get_size();
  m_ctrl_size = ctrl_size;
  m_sid = sid;
  m_tpc = tpc;
  m_wid = wid;
  config->m_address_mapping.addrdec_tlx(access.get_addr(), &m_raw_addr);
  m_partition_addr =
      config->m_address_mapping.partition_address(access.get_addr());
  m_type = m_access.is_write() ? WRITE_REQUEST : READ_REQUEST;
  m_timestamp = cycle;
  m_timestamp2 = 0;
  m_status = MEM_FETCH_INITIALIZED;
  m_status_change = cycle;
  m_mem_config = config;
  icnt_flit_size = config->icnt_flit_size;
  original_mf = m_original_mf;
  original_wr_mf = m_original_wr_mf;
  if (m_original_mf) {
    m_raw_addr.chip = m_original_mf->get_tlx_addr().chip;
    m_raw_addr.sub_partition = m_original_mf->get_tlx_addr().sub_partition;
  }
}

mem_fetch::~mem_fetch() { m_status = MEM_FETCH_DELETED; }

void mem_fetch::set_status(enum mem_fetch_status status,
                           unsigned long long cycle) {
  m_status = status;
  m_status_change = cycle;
}

bool mem_fetch::isatomic() const {
  if (m_inst.empty()) return false;
  return m_inst.isatomic();
}

void mem_fetch::do_atomic() { m_inst.do_atomic(m_access.get_warp_mask()); }

bool mem_fetch::istexture() const {
  if (m_inst.empty()) return false;
  return m_inst.space.get_type() == tex_space;
}

bool mem_fetch::isconst() const {
  if (m_inst.empty()) return false;
  return (m_inst.space.get_type() == const_space) ||
         (m_inst.space.get_type() == param_space_kernel);
}

/// Returns number of flits traversing interconnect. simt_to_mem specifies the
/// direction
unsigned mem_fetch::get_num_flits(bool simt_to_mem) {
  unsigned sz = 0;
  // If atomic, write going to memory, or read coming back from memory, size =
  // ctrl + data. Else, only ctrl
  if (isatomic() || (simt_to_mem && get_is_write()) ||
      !(simt_to_mem || get_is_write()))
    sz = size();
  else
    sz = get_ctrl_size();

  return (sz / icnt_flit_size) + ((sz % icnt_flit_size) ? 1 : 0);
}

// void mem_fetch::print(FILE *fp, bool print_inst) const {
//   if (this == NULL) {
//     fprintf(fp, "NULL");
//     return;
//   }
//   fprintf(fp, "%s@%lu", get_access_type_str(), get_addr());
//
//   // if (this == NULL) {
//   //   fprintf(fp, " <NULL mem_fetch pointer>\n");
//   //   return;
//   // }
//   // fprintf(fp, "  mf: uid=%6u, sid%02u:w%02u, part=%u, ", m_request_uid,
//   // m_sid,
//   //         m_wid, m_raw_addr.chip);
//   // m_access.print(fp);
//   // if ((unsigned)m_status < NUM_MEM_REQ_STAT)
//   //   fprintf(fp, " status = %s (%llu), ", Status_str[m_status],
//   //   m_status_change);
//   // else
//   //   fprintf(fp, " status = %u??? (%llu), ", m_status, m_status_change);
//   // if (!m_inst.empty() && print_inst)
//   //   m_inst.print(fp);
//   // else
//   //   fprintf(fp, "\n");
// }

std::ostream &operator<<(std::ostream &os, const mem_fetch *fetch) {
  if (fetch == NULL) {
    os << "NULL";
  } else {
    if (fetch->is_reply()) {
      os << "Reply(";
    } else {
      os << "Req(";
    }
    os << fetch->m_access;
    // os << fetch->get_access_type_str() << "@";
    // new_addr_type addr = fetch->get_addr();
    // new_addr_type rel_addr = fetch->get_relative_addr();
    // if (addr == rel_addr) {
    //   os << addr;
    // } else {
    //   os << fetch->get_alloc_id() << "+" << rel_addr;
    // }
    os << ")";
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const mem_fetch &fetch) {
  os << static_cast<const mem_fetch *>(&fetch);
  return os;
}
