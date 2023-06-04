#pragma once

#include <assert.h>
#include <cstdio>
#include <vector>

#include "../hal.hpp"
#include "histogram.hpp"

// per-insn active thread distribution (warp occ) logger
class insn_warp_occ_logger {
public:
  insn_warp_occ_logger(int simd_width)
      : m_simd_width(simd_width),
        m_insn_warp_occ(1, linear_histogram(1, "", m_simd_width)),
        m_id(s_ids++) {}

  insn_warp_occ_logger(const insn_warp_occ_logger &other)
      : m_simd_width(other.m_simd_width),
        m_insn_warp_occ(other.m_insn_warp_occ.size(),
                        linear_histogram(1, "", m_simd_width)),
        m_id(s_ids++) {}

  ~insn_warp_occ_logger() {}

  insn_warp_occ_logger &operator=(const insn_warp_occ_logger &p) {
    printf("insn_warp_occ_logger Operator= called: %02d \n", m_id);
    assert(0);
    return *this;
  }

  void set_id(int id) { m_id = id; }

  void log(address_type pc, int warp_occ) {
    if (pc >= m_insn_warp_occ.size())
      m_insn_warp_occ.resize(2 * pc, linear_histogram(1, "", m_simd_width));
    m_insn_warp_occ[pc].add2bin(warp_occ - 1);
  }

  void print(FILE *fout) const {
    for (unsigned i = 0; i < m_insn_warp_occ.size(); i++) {
      fprintf(fout, "InsnWarpOcc%02d-%d", m_id, i);
      m_insn_warp_occ[i].fprint(fout);
      fprintf(fout, "\n");
    }
  }

private:
  int m_simd_width;
  std::vector<linear_histogram> m_insn_warp_occ;
  int m_id;
  static int s_ids;
};
