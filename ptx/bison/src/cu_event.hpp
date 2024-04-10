#pragma once

#include "time.h"

struct CUevent_st {
public:
  CUevent_st(bool blocking) {
    m_uid = ++m_next_event_uid;
    m_blocking = blocking;
    m_updates = 0;
    m_wallclock = 0;
    m_gpu_tot_sim_cycle = 0;
    m_issued = 0;
    m_done = false;
  }
  void update(double cycle, time_t clk) {
    m_updates++;
    m_wallclock = clk;
    m_gpu_tot_sim_cycle = cycle;
    m_done = true;
  }
  // void set_done() { assert(!m_done); m_done=true; }
  int get_uid() const { return m_uid; }
  unsigned num_updates() const { return m_updates; }
  bool done() const { return m_updates == m_issued; }
  time_t clock() const { return m_wallclock; }
  void issue() { m_issued++; }
  unsigned int num_issued() const { return m_issued; }

private:
  int m_uid;
  bool m_blocking;
  bool m_done;
  unsigned int m_updates;
  unsigned int m_issued;
  time_t m_wallclock;
  double m_gpu_tot_sim_cycle;

  static int m_next_event_uid;
};
