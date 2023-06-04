#include "thread_cflocality.hpp"

thread_CFlocality::thread_CFlocality(gpgpu_context *ctx, std::string name,
                                     unsigned long long snap_shot_interval,
                                     int nthreads, address_type start_pc,
                                     unsigned long long start_cycle)
    : snap_shot_trigger(snap_shot_interval), m_name(name), m_nthreads(nthreads),
      m_thread_pc(nthreads, start_pc), m_cycle(start_cycle),
      m_thd_span(start_cycle, ctx) {
  std::fill(
      m_thread_pc.begin(), m_thread_pc.end(),
      -1); // so that hw thread with no work assigned will not clobber results
}

thread_CFlocality::~thread_CFlocality() {}

void thread_CFlocality::update_thread_pc(int thread_id, address_type pc) {
  m_thread_pc[thread_id] = pc;
  m_thd_span.set_span(pc);
}

void thread_CFlocality::snap_shot(unsigned long long current_cycle) {
  m_thd_span_archive.push_back(m_thd_span);
  m_thd_span.reset(current_cycle);
  for (int i = 0; i < (int)m_thread_pc.size(); i++) {
    m_thd_span.set_span(m_thread_pc[i]);
  }
}

void thread_CFlocality::spill(FILE *fout, bool final) {
  std::list<thread_insn_span>::iterator lit = m_thd_span_archive.begin();
  for (; lit != m_thd_span_archive.end(); lit = m_thd_span_archive.erase(lit)) {
    fprintf(fout, "%s-", m_name.c_str());
    lit->print_histo(fout);
  }
  assert(m_thd_span_archive.empty());
  if (final) {
    fprintf(fout, "%s-", m_name.c_str());
    m_thd_span.print_histo(fout);
  }
}

void thread_CFlocality::print_visualizer(FILE *fout) {
  fprintf(fout, "%s: ", m_name.c_str());
  if (m_thd_span_archive.empty()) {
    // visualizer do no require snap_shots
    m_thd_span.print_sparse_histo(fout);

    // clean the thread span
    m_thd_span.reset(0);
    for (int i = 0; i < (int)m_thread_pc.size(); i++)
      m_thd_span.set_span(m_thread_pc[i]);
  } else {
    assert(0); // TODO: implement fall back so that visualizer can work with
               // snap shots
  }
}

void thread_CFlocality::print_visualizer(gzFile fout) {
  gzprintf(fout, "%s: ", m_name.c_str());
  if (m_thd_span_archive.empty()) {
    // visualizer do no require snap_shots
    m_thd_span.print_sparse_histo(fout);

    // clean the thread span
    m_thd_span.reset(0);
    for (int i = 0; i < (int)m_thread_pc.size(); i++) {
      m_thd_span.set_span(m_thread_pc[i]);
    }
  } else {
    assert(0); // TODO: implement fall back so that visualizer can work with
               // snap shots
  }
}

void thread_CFlocality::print_span(FILE *fout) const {
  std::list<thread_insn_span>::const_iterator lit = m_thd_span_archive.begin();
  for (; lit != m_thd_span_archive.end(); ++lit) {
    fprintf(fout, "%s-", m_name.c_str());
    lit->print_span(fout);
  }
  fprintf(fout, "%s-", m_name.c_str());
  m_thd_span.print_span(fout);
}

void thread_CFlocality::print_histo(FILE *fout) const {
  std::list<thread_insn_span>::const_iterator lit = m_thd_span_archive.begin();
  for (; lit != m_thd_span_archive.end(); ++lit) {
    fprintf(fout, "%s-", m_name.c_str());
    lit->print_histo(fout);
  }
  fprintf(fout, "%s-", m_name.c_str());
  m_thd_span.print_histo(fout);
}
