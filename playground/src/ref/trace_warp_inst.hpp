#pragma once

#include <string>
#include <unordered_map>

#include "opcode_char.hpp"
#include "warp_instr.hpp"

class inst_trace_t;

class trace_warp_inst_t : public warp_inst_t {
public:
  trace_warp_inst_t() {
    // m_opcode = 0;
    should_do_atomic = false;
  }

  trace_warp_inst_t(const class core_config *config) : warp_inst_t(config) {
    // m_opcode = 0;
    should_do_atomic = false;
  }

  bool parse_from_trace_struct(
      const inst_trace_t &trace,
      const std::unordered_map<std::string, OpcodeChar> *OpcodeMap,
      const class trace_config *tconfig,
      const class kernel_trace_t *kernel_trace_info);
};

// void move_warp(trace_warp_inst_t *&dst, trace_warp_inst_t *&src);
// void move_warp(warp_inst_t *&dst, warp_inst_t *&src);

template <typename T> void move_warp(T *&dst, T *&src) {
  printf("\e[1;37m moving warp=%u \e[0m \n", src->warp_id());
  assert(dst->empty());
  T *temp = dst;
  dst = src;
  src = temp;
  src->clear();
}
