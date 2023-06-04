#pragma once

#include <list>
#include <zlib.h>

#include "../hal.hpp"
#include "snap_shot_trigger.hpp"
#include "spill_log_interface.hpp"
#include "thread_insn_span.hpp"

class gpgpu_context;

class thread_CFlocality : public snap_shot_trigger, public spill_log_interface {
public:
  thread_CFlocality(gpgpu_context *ctx, std::string name,
                    unsigned long long snap_shot_interval, int nthreads,
                    address_type start_pc, unsigned long long start_cycle = 0);
  ~thread_CFlocality();

  void update_thread_pc(int thread_id, address_type pc);
  void snap_shot(unsigned long long current_cycle);
  void spill(FILE *fout, bool final);

  void print_visualizer(FILE *fout);
  void print_visualizer(gzFile fout);
  void print_span(FILE *fout) const;
  void print_histo(FILE *fout) const;

private:
  std::string m_name;

  int m_nthreads;
  std::vector<address_type> m_thread_pc;

  unsigned long long m_cycle;
  thread_insn_span m_thd_span;
  std::list<thread_insn_span> m_thd_span_archive;
};
