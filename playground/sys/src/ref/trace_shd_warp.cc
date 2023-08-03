#include "trace_shd_warp.hpp"

#include <iomanip>

#include "hal.hpp"
#include "io.hpp"
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
  // also consider constant loads, which are categorized as ALU_OP
  if (inst->op == LOAD_OP || inst->op == STORE_OP || inst->opcode() == OP_LDC) {
    return true;
  }
  return false;
}

const trace_warp_inst_t *trace_shd_warp_t::get_current_trace_inst() {
  if (m_shader->get_gpu()->gpgpu_ctx->accelsim_compat_mode) {
    assert(0 &&
           "get_current_trace_inst cannot be used in accelsim compat mode");
    // return warp_traces[trace_pc];
    // return NULL;
  }
  unsigned int temp_trace_pc = trace_pc;
  while (temp_trace_pc < warp_traces.size()) {
    const trace_warp_inst_t *parsed_inst =
        get_cached_trace_instruction(temp_trace_pc);
    // const trace_warp_inst_t *parsed_inst=
    //     parse_trace_instruction(warp_traces[temp_trace_pc]);

    temp_trace_pc++;

    if (is_memory_instruction(parsed_inst) || parsed_inst->op == EXIT_OPS) {
      return parsed_inst;
    }
  }
  return NULL;
}

const trace_warp_inst_t *trace_shd_warp_t::get_cached_trace_instruction(
    unsigned cache_pc) {
  if (m_shader->get_gpu()->gpgpu_ctx->accelsim_compat_mode) {
    assert(0 &&
           "get_cached_trace_instruction cannot be used in accelsim compat "
           "mode");
  }

  if (cache_pc < trace_pc) {
    assert(0 &&
           "cache_pc is smaller than trace pc, yet issued instructions are not "
           "longer valid");
  }
  if (cache_pc < parsed_warp_traces_cache.size()) {
    // cache hit
    assert(parsed_warp_traces_cache[cache_pc] != NULL);

    // this does not hold:
    // (the heap allocated instructions are at some point deleted with the
    // pointer dangling) assert(warp_traces[trace_pc].m_pc ==
    //        parsed_warp_traces_cache[trace_pc]->pc);
    return parsed_warp_traces_cache[cache_pc];
  }

  // cache miss
  // want trace pc=5
  // have cached size of 3 (0, 1, 2)
  // need length of 6
  // need to add 3 more entries
  // need to 3 to 5: 3, 4, 5
  unsigned temp_trace_pc = parsed_warp_traces_cache.size();
  while (temp_trace_pc < warp_traces.size() && temp_trace_pc <= cache_pc) {
    // pc = (address_type)trace.m_pc;
    const inst_trace_t &trace = warp_traces[temp_trace_pc];
    const trace_warp_inst_t *parsed_inst = parse_trace_instruction(trace);

    if (is_memory_instruction(parsed_inst) || parsed_inst->op == EXIT_OPS) {
      // count++;
    }

    parsed_warp_traces_cache.push_back(parsed_inst);
    assert(temp_trace_pc == parsed_warp_traces_cache.size() - 1);
    assert(warp_traces[temp_trace_pc].m_pc ==
           parsed_warp_traces_cache[temp_trace_pc]->pc);
    // printf("=> cached warp trace instruction[%u] pc=%u\n", temp_trace_pc,
    //        warp_traces[temp_trace_pc].m_pc);
    temp_trace_pc++;
  }

  assert(cache_pc < warp_traces.size());
  assert(parsed_warp_traces_cache.size() == cache_pc + 1);
  return parsed_warp_traces_cache[cache_pc];
}

void trace_shd_warp_t::print_trace_instructions(
    bool all, std::shared_ptr<spdlog::logger> &logger) {
  if (m_shader->get_gpu()->gpgpu_ctx->accelsim_compat_mode) {
    assert(0 &&
           "print_trace_instructions cannot be used in accelsim compat mode");
  }

  // the instructions before trace_pc might have been freed after issue already
  // unsigned temp_trace_pc = 0;
  logger->debug("====> instruction at trace pc < {:<4} already issued ...",
                trace_pc);
  unsigned temp_trace_pc = trace_pc;
  while (temp_trace_pc < warp_traces.size()) {
    const inst_trace_t &trace = warp_traces[temp_trace_pc];
    // const trace_warp_inst_t *new_inst = parse_trace_instruction(trace);
    const trace_warp_inst_t *parsed_inst =
        get_cached_trace_instruction(temp_trace_pc);

    if (all || is_memory_instruction(parsed_inst) ||
        parsed_inst->op == EXIT_OPS) {
      assert(warp_traces[temp_trace_pc].m_pc == parsed_inst->pc);

      std::vector<new_addr_type> addresses;
      if (trace.memadd_info != NULL) {
        for (unsigned i = 0; i < WARP_SIZE; ++i)
          addresses.push_back(trace.memadd_info->addrs[i]);
      }

      logger->debug(
          "====> instruction at trace pc {:>4}:\t {:<10}\t {:<15} "
          "\t\tactive={}\tpc = {:>4} = {:<4}",
          temp_trace_pc, parsed_inst->opcode_str(), trace.opcode,
          mask_to_string(parsed_inst->get_active_mask()),
          warp_traces[temp_trace_pc].m_pc, parsed_inst->pc);

      // unsigned tid = 0;
      // for (new_addr_type &address : addresses) {
      //   logger->trace("\t thread {:>2} accesses {}", tid, address);
      //   tid++;
      // }
    }
    temp_trace_pc++;
  }
}

const warp_inst_t *trace_shd_warp_t::get_next_trace_inst() {
  if (m_shader->get_gpu()->gpgpu_ctx->accelsim_compat_mode) {
    if (trace_pc < warp_traces.size()) {
      trace_warp_inst_t *new_inst =
          new trace_warp_inst_t(get_shader()->get_config());

      const inst_trace_t &trace = warp_traces[trace_pc];
      new_inst->parse_from_trace_struct(trace, m_kernel_info->OpcodeMap,
                                        m_kernel_info->m_tconfig,
                                        m_kernel_info->m_kernel_trace_info);
      trace_pc++;
      return new_inst;
    }
  } else {
    while (trace_pc < warp_traces.size()) {
      const trace_warp_inst_t *parsed_inst =
          get_cached_trace_instruction(trace_pc);

      // const trace_warp_inst_t *parsed_inst =
      //     parse_trace_instruction(warp_traces[trace_pc]);

      // must be here otherwise we do not increment when finding an instruction
      trace_pc++;

      if (is_memory_instruction(parsed_inst) || parsed_inst->op == EXIT_OPS) {
        return parsed_inst;
      }
    }
  }
  return NULL;
}

unsigned long trace_shd_warp_t::instruction_count() {
  if (m_shader->get_gpu()->gpgpu_ctx->accelsim_compat_mode) {
    return warp_traces.size();
  } else {
    // NOTE: we cannot assume cached instructions < trace_pc to still be valid,
    // since old warp instructions are being freed after issue.
    //
    // have two options:
    //  1. parse and delete in the loop (expensive)
    //  2. keep an instruction count and make sure its valid by filling the
    //  entire cache
    //
    // force filling up the entire cache
    // get_cached_trace_instruction(warp_traces.size() - 1);
    if (m_instruction_count == 0) {
      for (const inst_trace_t &trace : warp_traces) {
        // for (unsigned temp_pc = 0; temp_pc < warp_traces.size(); temp_pc++) {
        // const trace_warp_inst_t *parsed_inst =
        // get_cached_trace_instruction(temp_pc);
        const trace_warp_inst_t *parsed_inst = parse_trace_instruction(trace);

        if (is_memory_instruction(parsed_inst) || parsed_inst->op == EXIT_OPS) {
          m_instruction_count++;
        }
        delete parsed_inst;
      }
    }
    return m_instruction_count;
  }
}

void trace_shd_warp_t::clear() {
  trace_pc = 0;
  m_instruction_count = 0;
  warp_traces.clear();
  parsed_warp_traces_cache.clear();
}

// functional_done
bool trace_shd_warp_t::trace_done() const {
  return trace_pc >= warp_traces.size();
}

address_type trace_shd_warp_t::get_start_trace_pc() {
  assert(warp_traces.size() > 0);
  return warp_traces[0].m_pc;
}

address_type trace_shd_warp_t::get_pc() {
  assert(warp_traces.size() > 0);  // must at least contain an exit code
  assert(trace_pc < warp_traces.size());

  if (m_shader->get_gpu()->gpgpu_ctx->accelsim_compat_mode) {
    return warp_traces[trace_pc].m_pc;
  } else {
    return get_current_trace_inst()->pc;
  }
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
