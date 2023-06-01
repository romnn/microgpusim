#pragma once

#include <unordered_map>
#include <vector>

#include "inst_trace.hpp"
#include "kernel_info.hpp"
#include "kernel_trace.hpp"
#include "opcode_char.hpp"
#include "trace_function_info.hpp"

class trace_parser;

class trace_kernel_info_t : public kernel_info_t {
public:
  trace_kernel_info_t(dim3 gridDim, dim3 blockDim,
                      trace_function_info *m_function_info,
                      trace_parser *parser, class trace_config *config,
                      kernel_trace_t *kernel_trace_info);

  void get_next_threadblock_traces(
      std::vector<std::vector<inst_trace_t> *> threadblock_traces);

  unsigned long get_cuda_stream_id() {
    return m_kernel_trace_info->cuda_stream_id;
  }

  kernel_trace_t *get_trace_info() { return m_kernel_trace_info; }

  bool was_launched() { return m_was_launched; }

  void set_launched() { m_was_launched = true; }

private:
  trace_config *m_tconfig;
  const std::unordered_map<std::string, OpcodeChar> *OpcodeMap;
  trace_parser *m_parser;
  kernel_trace_t *m_kernel_trace_info;
  bool m_was_launched;

  friend class trace_shd_warp_t;
};
