#include "trace_parser.hpp"

#include <fstream>

#include "playground-sys/src/bridge/trace_parser.rs.h"
#include "../trace_config.hpp"
#include "../opcode.hpp"
#include "../isa_def.hpp"
#include "../shader_core_config.hpp"

std::unique_ptr<trace_parser_bridge> new_trace_parser_bridge(
    rust::String kernellist_filepath) {
  trace_parser parser(kernellist_filepath.c_str());
  return std::make_unique<trace_parser_bridge>(parser);
}

std::unique_ptr<trace_warp_inst_t> new_trace_warp_inst() {
  return std::make_unique<trace_warp_inst_t>();
}

// void trace_parser_bridge::get_next_threadblock_traces(
//     kernel_trace_t &kernel,
//     const std::vector<ThreadBlockTraces> &threadblock_traces) const {
//   std::vector<std::vector<inst_trace_t> *> _threadblock_traces;
//   for (auto &threads : threadblock_traces) {
//     _threadblock_traces.push_back(threads.inner);
//   }
//   parser.get_next_threadblock_traces(_threadblock_traces,
//   kernel.trace_verion,
//                                      kernel.enable_lineinfo, kernel.ifs);
// }

// std::unique_ptr<std::vector<ThreadBlockTraces>>
// trace_parser_bridge::get_next_threadblock_traces(kernel_trace_t *kernel,
//                                                  unsigned max_warps) const {
//   std::vector<ThreadBlockTraces> out;
//   for (unsigned w = 0; w < max_warps; w++) {
//     out.push_back(
//         ThreadBlockTraces{std::make_unique<std::vector<inst_trace_t>>()});
//   }
//   std::vector<std::vector<inst_trace_t> *> _threadblock_traces;
//   for (unsigned w = 0; w < max_warps; w++) {
//     _threadblock_traces.push_back(out[w].inner.get());
//   }
//   parser.get_next_threadblock_traces(_threadblock_traces,
//   kernel->trace_verion,
//                                      kernel->enable_lineinfo, kernel->ifs);
//   return std::make_unique<std::vector<ThreadBlockTraces>>(std::move(out));
// }

std::unique_ptr<std::vector<TraceEntry>>
trace_parser_bridge::get_next_threadblock_traces(kernel_trace_t *kernel) const {
  std::vector<TraceEntry> out;

  trace_config *tconfig = new trace_config();
  core_config *config = new shader_core_config(NULL);
  config->warp_size = 32;
  config->mem_warp_parts = 1;

  const std::unordered_map<std::string, OpcodeChar> *OpcodeMap;
  // resolve the binary version
  unsigned binary_version = kernel->binary_verion;
  if (binary_version == AMPERE_RTX_BINART_VERSION ||
      binary_version == AMPERE_A100_BINART_VERSION)
    OpcodeMap = &Ampere_OpcodeMap;
  else if (binary_version == VOLTA_BINART_VERSION)
    OpcodeMap = &Volta_OpcodeMap;
  else if (binary_version == PASCAL_TITANX_BINART_VERSION ||
           binary_version == PASCAL_P100_BINART_VERSION)
    OpcodeMap = &Pascal_OpcodeMap;
  else if (binary_version == KEPLER_BINART_VERSION)
    OpcodeMap = &Kepler_OpcodeMap;
  else if (binary_version == TURING_BINART_VERSION)
    OpcodeMap = &Turing_OpcodeMap;
  else {
    assert(0 && "unsupported binary version");
  }

  while (!kernel->ifs->eof()) {
    unsigned block_id_x = (unsigned)-1;
    unsigned block_id_y = (unsigned)-1;
    unsigned block_id_z = (unsigned)-1;
    bool start_of_tb_stream_found = false;

    unsigned warp_id = 0;
    unsigned insts_num = 0;
    // unsigned inst_count = 0;

    while (!kernel->ifs->eof()) {
      std::string line;
      std::stringstream ss;
      std::string string1, string2;

      getline(*kernel->ifs, line);

      if (line.length() == 0) {
        continue;
      } else {
        ss.str(line);
        ss >> string1 >> string2;
        if (string1 == "#BEGIN_TB") {
          if (!start_of_tb_stream_found) {
            start_of_tb_stream_found = true;
          } else
            assert(0 &&
                   "Parsing error: thread block start before the previous one "
                   "finishes");
        } else if (string1 == "#END_TB") {
          assert(start_of_tb_stream_found);
          break;  // end of TB stream
        } else if (string1 == "thread" && string2 == "block") {
          assert(start_of_tb_stream_found);
          sscanf(line.c_str(), "thread block = %d,%d,%d", &block_id_x,
                 &block_id_y, &block_id_z);
          std::cout << line << std::endl;
        } else if (string1 == "warp") {
          // the start of new warp stream
          assert(start_of_tb_stream_found);
          sscanf(line.c_str(), "warp = %d", &warp_id);
        } else if (string1 == "insts") {
          assert(start_of_tb_stream_found);
          sscanf(line.c_str(), "insts = %d", &insts_num);
          // threadblock_traces[warp_id]->resize(
          //     insts_num);  // allocate all the space at once
          // inst_count = 0;
        } else {
          assert(start_of_tb_stream_found);
          // inst_trace_t trace_inst = inst_trace_t();
          // threadblock_traces[warp_id]
          //     ->at(inst_count)
          inst_trace_t trace_inst = inst_trace_t();
          trace_inst.parse_from_string(line, kernel->trace_verion,
                                       kernel->enable_lineinfo);

          trace_warp_inst_t warp_inst = trace_warp_inst_t(config);
          warp_inst.parse_from_trace_struct(trace_inst, OpcodeMap, tconfig,
                                            kernel);

          TraceEntry entry = {};
          entry.block_x = block_id_x;
          entry.block_y = block_id_y;
          entry.block_z = block_id_z;
          entry.line_num = kernel->enable_lineinfo ? trace_inst.line_num : 0;
          entry.pc = trace_inst.m_pc;
          entry.mask = trace_inst.mask;
          entry.raw_opcode = strcpy(new char[trace_inst.opcode.length() + 1],
                                    trace_inst.opcode.c_str());
          entry.opcode = (TraceInstrOpcode)warp_inst.m_opcode;
          entry.op = warp_inst.op;

          entry.reg_srcs_num = trace_inst.reg_srcs_num;
          entry.reg_dsts_num = trace_inst.reg_dsts_num;

          std::copy(std::begin(trace_inst.reg_dest),
                    std::end(trace_inst.reg_dest), std::begin(entry.reg_dest));
          std::copy(std::begin(trace_inst.reg_src),
                    std::end(trace_inst.reg_src), std::begin(entry.reg_src));

          out.push_back(entry);
          // inst_count++;
        }
      }
    }
  }

  return std::make_unique<std::vector<TraceEntry>>(std::move(out));
}

// std::unique_ptr<std::vector<ThreadBlockInstructions>>
// trace_parser_bridge::get_next_threadblock_trace_instructions(
//     kernel_trace_t *kernel, unsigned max_warps) const {
//   std::vector<ThreadBlockInstructions> out;
//   std::vector<std::vector<inst_trace_t> *> _threadblock_traces;
//   for (unsigned w = 0; w < max_warps; w++) {
//     _threadblock_traces.push_back(new std::vector<inst_trace_t>());
//   }
//   for (unsigned w = 0; w < max_warps; w++) {
//     out.push_back(
//         ThreadBlockInstructions{std::make_unique<std::vector<warp_inst_t>>()});
//     // std::make_unique<std::vector<trace_warp_inst_t>>()});
//   }
//
//   // std::vector<std::vector<inst_trace_t> *> _threadblock_traces;
//   // for (unsigned w = 0; w < max_warps; w++) {
//   //   _threadblock_traces.push_back(out[w]);
//   // }
//   parser.get_next_threadblock_traces(_threadblock_traces,
//   kernel->trace_verion,
//                                      kernel->enable_lineinfo, kernel->ifs);
//
//   const std::unordered_map<std::string, OpcodeChar> *OpcodeMap;
//
//   // resolve the binary version
//   unsigned binary_version = kernel->binary_verion;
//   if (binary_version == AMPERE_RTX_BINART_VERSION ||
//       binary_version == AMPERE_A100_BINART_VERSION)
//     OpcodeMap = &Ampere_OpcodeMap;
//   else if (binary_version == VOLTA_BINART_VERSION)
//     OpcodeMap = &Volta_OpcodeMap;
//   else if (binary_version == PASCAL_TITANX_BINART_VERSION ||
//            binary_version == PASCAL_P100_BINART_VERSION)
//     OpcodeMap = &Pascal_OpcodeMap;
//   else if (binary_version == KEPLER_BINART_VERSION)
//     OpcodeMap = &Kepler_OpcodeMap;
//   else if (binary_version == TURING_BINART_VERSION)
//     OpcodeMap = &Turing_OpcodeMap;
//   else {
//     assert(0 && "unsupported binary version");
//   }
//
//   trace_config *tconfig = new trace_config();
//   core_config *config = new shader_core_config(NULL);
//   config->warp_size = 32;
//   config->mem_warp_parts = 1;
//
//   // now parse
//   for (unsigned w = 0; w < max_warps; w++) {
//     for (inst_trace_t &trace : *_threadblock_traces[w]) {
//       // fmt::println("warp {} trace pc: {}", w, trace.m_pc);
//       trace_warp_inst_t new_inst =
//           trace_warp_inst_t(config);  // get_shader()->get_config());
//
//       new_inst.m_empty = false;
//       new_inst.parse_from_trace_struct(trace, OpcodeMap, tconfig, kernel);
//       new_inst.m_warp_id = w;
//       fmt::println("warp {} parsed trace pc: {}", w, new_inst.get_pc());
//       out[w].inner.get()->push_back(new_inst);
//     }
//   }
//
//   for (unsigned w = 0; w < max_warps; w++) {
//     fmt::println("warp {} inst: {}", w, (*out[w].inner.get())[0].warp_id());
//     for (warp_inst_t &inst : *out[w].inner.get()) {
//       // fmt::println("warp {} inst: {}", w, inst.warp_id());
//     }
//   }
//   fmt::println("done");
//
//   return
//   std::make_unique<std::vector<ThreadBlockInstructions>>(std::move(out));
// }
