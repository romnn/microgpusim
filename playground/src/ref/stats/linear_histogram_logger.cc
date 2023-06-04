#include "linear_histogram_logger.hpp"

#include "tool.hpp"

linear_histogram_logger::linear_histogram_logger(
    int n_bins, unsigned long long snap_shot_interval, const char *name,
    bool reset_at_snap_shot, unsigned long long start_cycle)
    : snap_shot_trigger(snap_shot_interval), m_n_bins(n_bins),
      m_curr_lin_hist(m_n_bins, start_cycle), m_lin_hist_archive(),
      m_cycle(start_cycle), m_reset_at_snap_shot(reset_at_snap_shot),
      m_name(name), m_id(s_ids++) {}

linear_histogram_logger::linear_histogram_logger(
    const linear_histogram_logger &other)
    : snap_shot_trigger(other.get_interval()), m_n_bins(other.m_n_bins),
      m_curr_lin_hist(m_n_bins, other.m_cycle), m_lin_hist_archive(),
      m_cycle(other.m_cycle), m_reset_at_snap_shot(other.m_reset_at_snap_shot),
      m_name(other.m_name), m_id(s_ids++) {}

linear_histogram_logger::~linear_histogram_logger() {
  remove_snap_shot_trigger(this);
  remove_spill_log(this);
}

void linear_histogram_logger::snap_shot(unsigned long long current_cycle) {
  m_lin_hist_archive.push_back(m_curr_lin_hist);
  if (m_reset_at_snap_shot) {
    m_curr_lin_hist.reset(current_cycle);
  } else {
    m_curr_lin_hist.set_cycle(current_cycle);
  }
}

void linear_histogram_logger::spill(FILE *fout, bool final) {
  std::list<linear_histogram_snapshot>::iterator iter =
      m_lin_hist_archive.begin();
  for (; iter != m_lin_hist_archive.end();
       iter = m_lin_hist_archive.erase(iter)) {
    fprintf(fout, "%s%02d-", m_name.c_str(), (m_id >= 0) ? m_id : 0);
    iter->print(fout);
    fprintf(fout, "\n");
  }
  assert(m_lin_hist_archive.empty());
  if (final) {
    fprintf(fout, "%s%02d-", m_name.c_str(), (m_id >= 0) ? m_id : 0);
    m_curr_lin_hist.print(fout);
    fprintf(fout, "\n");
  }
}

void linear_histogram_logger::print(FILE *fout) const {
  std::list<linear_histogram_snapshot>::const_iterator iter =
      m_lin_hist_archive.begin();
  for (; iter != m_lin_hist_archive.end(); ++iter) {
    fprintf(fout, "%s%02d-", m_name.c_str(), m_id);
    iter->print(fout);
    fprintf(fout, "\n");
  }
  fprintf(fout, "%s%02d-", m_name.c_str(), m_id);
  m_curr_lin_hist.print(fout);
  fprintf(fout, "\n");
}

void linear_histogram_logger::print_visualizer(FILE *fout) {
  assert(m_lin_hist_archive.empty()); // don't support snapshot for now
  fprintf(fout, "%s", m_name.c_str());
  if (m_id >= 0) {
    fprintf(fout, "%02d: ", m_id);
  } else {
    fprintf(fout, ": ");
  }
  m_curr_lin_hist.print_visualizer(fout);
  fprintf(fout, "\n");
  if (m_reset_at_snap_shot) {
    m_curr_lin_hist.reset(0);
  }
}

void linear_histogram_logger::print_visualizer(gzFile fout) {
  assert(m_lin_hist_archive.empty()); // don't support snapshot for now
  gzprintf(fout, "%s", m_name.c_str());
  if (m_id >= 0) {
    gzprintf(fout, "%02d: ", m_id);
  } else {
    gzprintf(fout, ": ");
  }
  m_curr_lin_hist.print_visualizer(fout);
  gzprintf(fout, "\n");
  if (m_reset_at_snap_shot) {
    m_curr_lin_hist.reset(0);
  }
}
