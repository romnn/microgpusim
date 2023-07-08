#include <iomanip>

#include "trace_shd_warp.hpp"

#include "trace_instr_opcode.hpp"
#include "trace_kernel_info.hpp"
#include "trace_shader_core_ctx.hpp"
#include "trace_warp_inst.hpp"

const trace_warp_inst_t *trace_shd_warp_t::parse_trace_instruction(
    const inst_trace_t &trace) const {
  trace_warp_inst_t *new_inst =
      new trace_warp_inst_t(get_shader()->get_config());

  new_inst->parse_from_trace_struct(trace, m_kernel_info->OpcodeMap,
                                    m_kernel_info->m_tconfig,
                                    m_kernel_info->m_kernel_trace_info);
  return new_inst;
}

bool is_memory_instruction(const trace_warp_inst_t *inst) {
  if (inst->op == LOAD_OP || inst->op == STORE_OP || inst->opcode() == OP_LDC) {
    return true;
  }
  return false;
}

#ifdef BOX
const trace_warp_inst_t *trace_shd_warp_t::get_current_trace_inst() const {
  unsigned int temp_trace_pc = trace_pc;
  while (temp_trace_pc < warp_traces.size()) {
    const trace_warp_inst_t *new_inst =
        parse_trace_instruction(warp_traces[temp_trace_pc]);

    // trace_warp_inst_t *new_inst =
    //     new trace_warp_inst_t(get_shader()->get_config());
    //
    // const inst_trace_t &trace = warp_traces[temp_trace_pc];
    // new_inst->parse_from_trace_struct(trace, m_kernel_info->OpcodeMap,
    //                                   m_kernel_info->m_tconfig,
    //                                   m_kernel_info->m_kernel_trace_info);
    temp_trace_pc++;

    if (is_memory_instruction(new_inst) || new_inst->op == EXIT_OPS) {
      return new_inst;
    }
    // also consider constant loads, which are categorized as ALU_OP
    // if (new_inst->op == LOAD_OP || new_inst->op == STORE_OP ||
    //     new_inst->op == EXIT_OPS || new_inst->opcode() == OP_LDC) {
    //   return new_inst;
    // }
  }
  return NULL;
}
#endif

template <size_t N>
std::string mask_to_string(std::bitset<N> mask) {
  std::string out;
  for (int i = mask.size() - 1; i >= 0; i--)
    out.append(((mask[i]) ? "1" : "0"));
  return out;
}

void trace_shd_warp_t::print_trace_instructions(bool all) {
  unsigned temp_trace_pc = 0;
  while (temp_trace_pc < warp_traces.size()) {
    const inst_trace_t &trace = warp_traces[temp_trace_pc];
    const trace_warp_inst_t *new_inst = parse_trace_instruction(trace);

    // trace_warp_inst_t *new_inst =
    //     new trace_warp_inst_t(get_shader()->get_config());
    //
    // const inst_trace_t &trace = warp_traces[temp_trace_pc];
    // new_inst->parse_from_trace_struct(trace, m_kernel_info->OpcodeMap,
    //                                   m_kernel_info->m_tconfig,
    //                                   m_kernel_info->m_kernel_trace_info);

    // also consider constant loads, which are categorized as ALU_OP
    // if (all || new_inst->op == LOAD_OP || new_inst->op == STORE_OP ||
    //     new_inst->op == EXIT_OPS || new_inst->opcode() == OP_LDC) {
    if (all || is_memory_instruction(new_inst) || new_inst->op == EXIT_OPS) {
      assert(warp_traces[temp_trace_pc].m_pc == new_inst->pc);
      // std::cout << "====> instruction at trace pc " << std::left <<
      // std::setw(4)
      //           << temp_trace_pc << ":";
      // std::cout << "\t" << std::setfill(' ') << std::left << std::setw(10)
      //           << new_inst->opcode_str();
      // std::cout << "\t" << std::setfill(' ') << std::left << std::setw(15)
      //           << trace.opcode.c_str();
      // std::cout << "\t"
      //           << "active=" << mask_to_string(new_inst->get_active_mask());
      // std::cout << "\t"
      //           << "tpc=";
      // std::cout << std::setfill(' ') << std::left << std::setw(4)
      //           << warp_traces[temp_trace_pc].m_pc;
      // std::cout << "==";
      // std::cout << std::setfill(' ') << std::left << std::setw(4)
      //           << new_inst->pc;
      // std::cout << std::endl;
      printf(
          "====> instruction at trace pc %-4d:\t %-10s\t %-15s \t\tactive=%s "
          "\tpc = %-4lu\n",
          temp_trace_pc, new_inst->opcode_str(), trace.opcode.c_str(),
          mask_to_string(new_inst->get_active_mask()).c_str(), new_inst->pc);
    }
    temp_trace_pc++;
  }
}

const warp_inst_t *trace_shd_warp_t::get_next_trace_inst() {
#ifdef BOX
  while (trace_pc < warp_traces.size()) {
    const trace_warp_inst_t *new_inst =
        parse_trace_instruction(warp_traces[trace_pc]);

    // trace_warp_inst_t *new_inst =
    //     new trace_warp_inst_t(get_shader()->get_config());
    //
    // const inst_trace_t &trace = warp_traces[trace_pc];
    // new_inst->parse_from_trace_struct(trace, m_kernel_info->OpcodeMap,
    //                                   m_kernel_info->m_tconfig,
    //                                   m_kernel_info->m_kernel_trace_info);

    // printf("====> trace_shd_warp_t::get_next_trace_inst(): opcode = %s (%s, "
    //        "%d)\n",
    //        trace.opcode.c_str(), new_inst->opcode_str(), new_inst->opcode());

    // must be here otherwise we do not increment when finding an instruction
    trace_pc++;

    // also consider constant loads, which are categorized as ALU_OP
    if (is_memory_instruction(new_inst) || new_inst->op == EXIT_OPS) {
      // if (new_inst->op == LOAD_OP || new_inst->op == STORE_OP ||
      //     new_inst->op == EXIT_OPS || new_inst->opcode() == OP_LDC) {
      return new_inst;
    }
  }
  return NULL;
#else
  if (trace_pc < warp_traces.size()) {
    trace_warp_inst_t *new_inst =
        new trace_warp_inst_t(get_shader()->get_config());

    const inst_trace_t &trace = warp_traces[trace_pc];
    new_inst->parse_from_trace_struct(trace, m_kernel_info->OpcodeMap,
                                      m_kernel_info->m_tconfig,
                                      m_kernel_info->m_kernel_trace_info);

    // printf("====> trace_shd_warp_t::get_next_trace_inst(): opcode = %s (%s, "
    //        "%d)\n",
    //        trace.opcode.c_str(), new_inst->opcode_str(), new_inst->opcode());

    trace_pc++;
    return new_inst;
  } else
    return NULL;
#endif
}

unsigned long trace_shd_warp_t::instruction_count() const {
#ifdef BOX
  unsigned count = 0;
  for (const inst_trace_t &trace : warp_traces) {
    const trace_warp_inst_t *new_inst = parse_trace_instruction(trace);

    // trace_warp_inst_t *new_inst =
    //     new trace_warp_inst_t(get_shader()->get_config());
    //
    // new_inst->parse_from_trace_struct(trace, m_kernel_info->OpcodeMap,
    //                                   m_kernel_info->m_tconfig,
    //                                   m_kernel_info->m_kernel_trace_info);

    // note: we do not count EXIT when computing the instruction count
    if (is_memory_instruction(new_inst) || new_inst->op == EXIT_OPS) {
      // if (new_inst->op == LOAD_OP || new_inst->op == STORE_OP ||
      //     new_inst->opcode() == OP_LDC) {
      count++;
    }
  }
  return count;
#else
  return warp_traces.size();
#endif
}

void trace_shd_warp_t::clear() {
  trace_pc = 0;
  warp_traces.clear();
}

// functional_done
bool trace_shd_warp_t::trace_done() const {
  return trace_pc >= warp_traces.size();
}

address_type trace_shd_warp_t::get_start_trace_pc() {
  assert(warp_traces.size() > 0);
  return warp_traces[0].m_pc;
}

address_type trace_shd_warp_t::get_pc() const {
  assert(warp_traces.size() > 0);  // must at least contain an exit code
  assert(trace_pc < warp_traces.size());
#ifdef BOX
  return get_current_trace_inst()->pc;
#else
  return warp_traces[trace_pc].m_pc;
#endif
}

bool trace_shd_warp_t::waiting() {
  if (functional_done()) {
    // waiting to be initialized with a kernel
    return true;
  } else if (m_shader->warp_waiting_at_barrier(m_warp_id)) {
    // waiting for other warps in CTA to reach barrier
    return true;
  } else if (m_shader->warp_waiting_at_mem_barrier(m_warp_id)) {
    // waiting for memory barrier
    return true;
  } else if (m_n_atomic > 0) {
    // waiting for atomic operation to complete at memory:
    // this stall is not required for accurate timing model, but rather we
    // stall here since if a call/return instruction occurs in the meantime
    // the functional execution of the atomic when it hits DRAM can cause
    // the wrong register to be read.
    return true;
  }
  return false;
}

bool trace_shd_warp_t::functional_done() const {
  return get_n_completed() == m_warp_size;
}

bool trace_shd_warp_t::hardware_done() const {
  return functional_done() && stores_done() && !inst_in_pipeline();
}

std::unique_ptr<trace_shd_warp_t> new_trace_shd_warp(
    class trace_shader_core_ctx *shader, unsigned warp_size) {
  return std::make_unique<trace_shd_warp_t>(shader, warp_size);
}
