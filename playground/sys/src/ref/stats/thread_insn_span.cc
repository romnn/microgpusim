#include "thread_insn_span.hpp"

#include "../gpgpu_context.hpp"

thread_insn_span::thread_insn_span(unsigned long long cycle, gpgpu_context *ctx)
    : m_cycle(cycle), m_insn_span_count() {
  gpgpu_ctx = ctx;
}

thread_insn_span::~thread_insn_span() {}

thread_insn_span::thread_insn_span(const thread_insn_span &other,
                                   gpgpu_context *ctx)
    : m_cycle(other.m_cycle), m_insn_span_count(other.m_insn_span_count) {
  gpgpu_ctx = ctx;
}

thread_insn_span &thread_insn_span::operator=(const thread_insn_span &other) {
  printf("thread_insn_span& operator=\n");
  if (this != &other) {
    m_insn_span_count = other.m_insn_span_count;
    m_cycle = other.m_cycle;
  }
  return *this;
}

thread_insn_span &thread_insn_span::operator+=(const thread_insn_span &other) {
  span_count_map::const_iterator i_sc = other.m_insn_span_count.begin();
  for (; i_sc != other.m_insn_span_count.end(); ++i_sc) {
    m_insn_span_count[i_sc->first] += i_sc->second;
  }
  return *this;
}

void thread_insn_span::set_span(address_type pc) {
  if (((int)pc) >= 0) m_insn_span_count[pc] += 1;
}

void thread_insn_span::reset(unsigned long long cycle) {
  m_cycle = cycle;
  m_insn_span_count.clear();
}

void thread_insn_span::print_span(FILE *fout) const {
  fprintf(fout, "%d: ", (int)m_cycle);
  span_count_map::const_iterator i_sc = m_insn_span_count.begin();
  for (; i_sc != m_insn_span_count.end(); ++i_sc) {
    fprintf(fout, "%ld ", i_sc->first);
  }
  fprintf(fout, "\n");
}

void thread_insn_span::print_histo(FILE *fout) const {
  fprintf(fout, "%d:", (int)m_cycle);
  span_count_map::const_iterator i_sc = m_insn_span_count.begin();
  for (; i_sc != m_insn_span_count.end(); ++i_sc) {
    fprintf(fout, "%d ", i_sc->second);
  }
  fprintf(fout, "\n");
}

void thread_insn_span::print_sparse_histo(FILE *fout) const {
  int n_printed_entries = 0;
  span_count_map::const_iterator i_sc = m_insn_span_count.begin();
  for (; i_sc != m_insn_span_count.end(); ++i_sc) {
    // REMOVE: ptx
    // unsigned ptx_lineno = gpgpu_ctx->translate_pc_to_ptxlineno(i_sc->first);
    unsigned ptx_lineno = 0;
    fprintf(fout, "%u %d ", ptx_lineno, i_sc->second);
    n_printed_entries++;
  }
  if (n_printed_entries == 0) {
    fprintf(fout, "0 0 ");
  }
  fprintf(fout, "\n");
}

void thread_insn_span::print_sparse_histo(gzFile fout) const {
  int n_printed_entries = 0;
  span_count_map::const_iterator i_sc = m_insn_span_count.begin();
  for (; i_sc != m_insn_span_count.end(); ++i_sc) {
    // REMOVE: ptx
    // unsigned ptx_lineno = gpgpu_ctx->translate_pc_to_ptxlineno(i_sc->first);
    unsigned ptx_lineno = 0;
    gzprintf(fout, "%u %d ", ptx_lineno, i_sc->second);
    n_printed_entries++;
  }
  if (n_printed_entries == 0) {
    gzprintf(fout, "0 0 ");
  }
  gzprintf(fout, "\n");
}
