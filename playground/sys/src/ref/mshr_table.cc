#include "mshr_table.hpp"

#include <sstream>

#include "hal.hpp"

/// Checks if there is a pending request to the lower memory level already
bool mshr_table::probe(new_addr_type block_addr) const {
  table::const_iterator a = m_data.find(block_addr);
  return a != m_data.end();
}

/// Checks if there is space for tracking a new memory access
bool mshr_table::full(new_addr_type block_addr) const {
  table::const_iterator i = m_data.find(block_addr);
  if (i != m_data.end())
    return i->second.m_list.size() >= m_max_merged;
  else
    return m_data.size() >= m_num_entries;
}

/// Add or merge this access
void mshr_table::add(new_addr_type block_addr, mem_fetch *mf) {
  m_data[block_addr].m_list.push_back(mf);
  assert(m_data.size() <= m_num_entries);
  assert(m_data[block_addr].m_list.size() <= m_max_merged);
  // indicate that this MSHR entry contains an atomic operation
  if (mf->isatomic()) {
    m_data[block_addr].m_has_atomic = true;
  }
}

/// check is_read_after_write_pending
bool mshr_table::is_read_after_write_pending(new_addr_type block_addr) {
  std::list<mem_fetch *> my_list = m_data[block_addr].m_list;
  bool write_found = false;
  for (std::list<mem_fetch *>::iterator it = my_list.begin();
       it != my_list.end(); ++it) {
    if ((*it)->is_write())  // Pending Write Request
      write_found = true;
    else if (write_found)  // Pending Read Request and we found previous Write
      return true;
  }

  return false;
}

/// Accept a new cache fill response: mark entry ready for processing
void mshr_table::mark_ready(new_addr_type block_addr, bool &has_atomic) {
  logger->debug("mshr_table::mark_ready({}, {})", block_addr, has_atomic);
  assert(!busy());
  table::iterator a = m_data.find(block_addr);
  assert(a != m_data.end());
  m_current_response.push_back(block_addr);
  has_atomic = a->second.m_has_atomic;
  assert(m_current_response.size() <= m_data.size());
}

/// Returns next ready accesses
std::list<mem_fetch *> mshr_table::next_accesses() {
  if (m_current_response.empty()) {
    return std::list<mem_fetch *>();
  }
  new_addr_type block_addr = m_current_response.front();
  return m_data[block_addr].m_list;
}

/// Returns next ready access
mem_fetch *mshr_table::next_access() {
  assert(access_ready());
  new_addr_type block_addr = m_current_response.front();
  assert(!m_data[block_addr].m_list.empty());
  mem_fetch *result = m_data[block_addr].m_list.front();
  m_data[block_addr].m_list.pop_front();
  if (m_data[block_addr].m_list.empty()) {
    // release entry
    m_data.erase(block_addr);
    m_current_response.pop_front();
  }
  return result;
}

// void mshr_table::display(FILE *fp) const {
//   fprintf(fp, "MSHR contents\n");
//   for (table::const_iterator e = m_data.begin(); e != m_data.end(); ++e) {
//     unsigned block_addr = e->first;
//     fprintf(fp, "MSHR: tag=0x%06x, atomic=%d %zu entries : ", block_addr,
//             e->second.m_has_atomic, e->second.m_list.size());
//     if (!e->second.m_list.empty()) {
//       mem_fetch *mf = e->second.m_list.front();
//       fprintf(fp, "%p :", mf);
//       std::stringstream buffer;
//       buffer << mf;
//       fprintf(fp, "%s", buffer.str().c_str());
//
//       // (std::ostream &)fp << mf;
//       // mf->print(fp);
//     } else {
//       fprintf(fp, " no memory requests???\n");
//     }
//   }
// }
