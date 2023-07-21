#pragma once

#include <unordered_map>

#include "hal.hpp"
#include "mem_fetch.hpp"

class mshr_table {
 public:
  mshr_table(unsigned num_entries, unsigned max_merged,
             std::shared_ptr<spdlog::logger> logger)
      : logger(logger),
        m_num_entries(num_entries),
        m_max_merged(max_merged),
        m_data(2 * num_entries) {}

  /// Checks if there is a pending request to the lower memory level already
  bool probe(new_addr_type block_addr) const;
  /// Checks if there is space for tracking a new memory access
  bool full(new_addr_type block_addr) const;
  /// Add or merge this access
  void add(new_addr_type block_addr, mem_fetch *mf);
  /// Returns true if cannot accept new fill responses
  bool busy() const { return false; }
  /// Accept a new cache fill response: mark entry ready for processing
  void mark_ready(new_addr_type block_addr, bool &has_atomic);
  /// Returns true if ready accesses exist
  bool access_ready() const { return !m_current_response.empty(); }
  /// Returns next ready access
  mem_fetch *next_access();
  std::list<mem_fetch *> next_accesses();
  void display(FILE *fp) const;
  // Returns true if there is a pending read after write
  bool is_read_after_write_pending(new_addr_type block_addr);

  void check_mshr_parameters(unsigned num_entries, unsigned max_merged) {
    assert(m_num_entries == num_entries &&
           "Change of MSHR parameters between kernels is not allowed");
    assert(m_max_merged == max_merged &&
           "Change of MSHR parameters between kernels is not allowed");
  }

  std::shared_ptr<spdlog::logger> logger;

 private:
  // finite sized, fully associative table, with a finite maximum number of
  // merged requests
  const unsigned m_num_entries;
  const unsigned m_max_merged;

  struct mshr_entry {
    std::list<mem_fetch *> m_list;
    bool m_has_atomic;
    mshr_entry() : m_has_atomic(false) {}
  };

  typedef std::unordered_map<new_addr_type, mshr_entry> table;
  typedef std::unordered_map<new_addr_type, mshr_entry> line_table;

  table m_data;
  line_table pending_lines;

  // it may take several cycles to process the merged requests
  bool m_current_response_ready;
  std::list<new_addr_type> m_current_response;
};
