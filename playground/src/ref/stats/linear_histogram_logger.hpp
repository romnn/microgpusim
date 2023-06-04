#pragma once

#include <assert.h>
#include <list>
#include <string>
#include <vector>
#include <zlib.h>

#include "snap_shot_trigger.hpp"
#include "spill_log_interface.hpp"

// generic linear histogram logger

class linear_histogram_snapshot {
public:
  linear_histogram_snapshot(int n_bins, unsigned long long cycle)
      : m_cycle(cycle), m_linear_histogram(n_bins, 0) {}

  linear_histogram_snapshot(const linear_histogram_snapshot &other)
      : m_cycle(other.m_cycle), m_linear_histogram(other.m_linear_histogram) {}

  ~linear_histogram_snapshot() {}

  void addsample(int pos) {
    assert((size_t)pos < m_linear_histogram.size());
    m_linear_histogram[pos] += 1;
  }

  void subsample(int pos) {
    assert((size_t)pos < m_linear_histogram.size());
    m_linear_histogram[pos] -= 1;
  }

  void reset(unsigned long long cycle) {
    m_cycle = cycle;
    m_linear_histogram.assign(m_linear_histogram.size(), 0);
  }

  void set_cycle(unsigned long long cycle) { m_cycle = cycle; }

  void print(FILE *fout) const {
    fprintf(fout, "%d = ", (int)m_cycle);
    for (unsigned int i = 0; i < m_linear_histogram.size(); i++) {
      fprintf(fout, "%d ", m_linear_histogram[i]);
    }
  }

  void print_visualizer(FILE *fout) const {
    for (unsigned int i = 0; i < m_linear_histogram.size(); i++) {
      fprintf(fout, "%d ", m_linear_histogram[i]);
    }
  }

  void print_visualizer(gzFile fout) const {
    for (unsigned int i = 0; i < m_linear_histogram.size(); i++) {
      gzprintf(fout, "%d ", m_linear_histogram[i]);
    }
  }

private:
  unsigned long long m_cycle;
  std::vector<int> m_linear_histogram;
};

class linear_histogram_logger : public snap_shot_trigger,
                                public spill_log_interface {
public:
  linear_histogram_logger(int n_bins, unsigned long long snap_shot_interval,
                          const char *name, bool reset_at_snap_shot = true,
                          unsigned long long start_cycle = 0);
  linear_histogram_logger(const linear_histogram_logger &other);

  ~linear_histogram_logger();

  void set_id(int id) { m_id = id; }
  void log(int pos) { m_curr_lin_hist.addsample(pos); }
  void unlog(int pos) { m_curr_lin_hist.subsample(pos); }
  void snap_shot(unsigned long long current_cycle);
  void spill(FILE *fout, bool final);

  void print(FILE *fout) const;
  void print_visualizer(FILE *fout);
  void print_visualizer(gzFile fout);

private:
  int m_n_bins;
  linear_histogram_snapshot m_curr_lin_hist;
  std::list<linear_histogram_snapshot> m_lin_hist_archive;
  unsigned long long m_cycle;
  bool m_reset_at_snap_shot;
  std::string m_name;
  int m_id;
  static int s_ids;
};
