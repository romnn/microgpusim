#pragma once

class ptx_warp_info {
public:
  ptx_warp_info(); // add get_core or something, or threads?
  unsigned get_done_threads() const;
  void inc_done_threads();
  void reset_done_threads();

private:
  unsigned m_done_threads;
};
