#pragma once

#include <unordered_map>
#include <zlib.h>

#include "../hal.hpp"

class gpgpu_context;

// thread control-flow locality logger
class thread_insn_span {
 public:
  thread_insn_span(unsigned long long cycle, gpgpu_context *ctx);
  thread_insn_span(const thread_insn_span &other, gpgpu_context *ctx);
  ~thread_insn_span();

  thread_insn_span &operator=(const thread_insn_span &other);
  thread_insn_span &operator+=(const thread_insn_span &other);
  void set_span(address_type pc);
  void reset(unsigned long long cycle);

  void print_span(FILE *fout) const;
  void print_histo(FILE *fout) const;
  void print_sparse_histo(FILE *fout) const;
  void print_sparse_histo(gzFile fout) const;

 private:
  gpgpu_context *gpgpu_ctx;

  typedef std::unordered_map<address_type, int> span_count_map;
  unsigned long long m_cycle;
  span_count_map m_insn_span_count;
};
