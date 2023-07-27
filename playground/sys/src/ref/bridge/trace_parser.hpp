#pragma once

#include "rust/cxx.h"
#include "trace_entry.hpp"
#include "../trace_parser.hpp"
#include "../kernel_trace.hpp"
#include "../trace_warp_inst.hpp"

// struct Threadblocks {
//   std::unique_ptr<std::vector<inst_trace_t>> inner;
//
//   // Threadblocks(Threadblocks &&rhs) = default;
//   // Threadblocks(const Threadblocks &) = delete;
//   // Threadblocks(Threadblocks &&) = default;
// };

class trace_parser_bridge {
 public:
  trace_parser_bridge(trace_parser parser) : parser(parser) {}

  std::unique_ptr<std::vector<trace_command>> parse_commandlist_file() const {
    return std::make_unique<std::vector<trace_command>>(
        parser.parse_commandlist_file());
  }

  // const trace_warp_inst_t *trace_shd_warp_t::parse_trace_instruction(
  //     const inst_trace_t &trace) const {
  //   trace_warp_inst_t *new_inst =
  //       new trace_warp_inst_t(get_shader()->get_config());
  //
  //   new_inst->parse_from_trace_struct(trace, m_kernel_info->OpcodeMap,
  //                                     m_kernel_info->m_tconfig,
  //                                     m_kernel_info->m_kernel_trace_info);
  //   return new_inst;
  // }

  std::unique_ptr<std::vector<TraceEntry>> get_next_threadblock_traces(
      kernel_trace_t *kernel) const;
  // , unsigned max_warps) const;

  // std::unique_ptr<std::vector<struct ThreadBlockTraces>>
  // get_next_threadblock_traces(kernel_trace_t *kernel, unsigned max_warps)
  // const;
  //
  // std::unique_ptr<std::vector<struct ThreadBlockInstructions>>
  // get_next_threadblock_trace_instructions(kernel_trace_t *kernel,
  //                                         unsigned max_warps) const;

  const trace_parser &inner() const { return parser; };
  trace_parser &inner_mut() { return parser; };

 private:
  trace_parser parser;
};

std::unique_ptr<trace_parser_bridge> new_trace_parser_bridge(
    rust::String kernellist_filepath);

std::unique_ptr<trace_warp_inst_t> new_trace_warp_inst();
