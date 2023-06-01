#include "opndcoll_rfu.hpp"

#include <cmath>

#include "register_set.hpp"
#include "shader_core_ctx.hpp"

// modifiers
std::list<opndcoll_rfu_t::op_t> opndcoll_rfu_t::arbiter_t::allocate_reads() {
  std::list<op_t>
      result; // a list of registers that (a) are in different register banks,
              // (b) do not go to the same operand collector

  int input;
  int output;
  int _inputs = m_num_banks;
  int _outputs = m_num_collectors;
  int _square = (_inputs > _outputs) ? _inputs : _outputs;
  assert(_square > 0);
  int _pri = (int)m_last_cu;

  // Clear matching
  for (int i = 0; i < _inputs; ++i)
    _inmatch[i] = -1;
  for (int j = 0; j < _outputs; ++j)
    _outmatch[j] = -1;

  for (unsigned i = 0; i < m_num_banks; i++) {
    for (unsigned j = 0; j < m_num_collectors; j++) {
      assert(i < (unsigned)_inputs);
      assert(j < (unsigned)_outputs);
      _request[i][j] = 0;
    }
    if (!m_queue[i].empty()) {
      const op_t &op = m_queue[i].front();
      int oc_id = op.get_oc_id();
      assert(i < (unsigned)_inputs);
      assert(oc_id < _outputs);
      _request[i][oc_id] = 1;
    }
    if (m_allocated_bank[i].is_write()) {
      assert(i < (unsigned)_inputs);
      _inmatch[i] = 0; // write gets priority
    }
  }

  ///// wavefront allocator from booksim... --->

  // Loop through diagonals of request matrix
  // printf("####\n");

  for (int p = 0; p < _square; ++p) {
    output = (_pri + p) % _outputs;

    // Step through the current diagonal
    for (input = 0; input < _inputs; ++input) {
      assert(input < _inputs);
      assert(output < _outputs);
      if ((output < _outputs) && (_inmatch[input] == -1) &&
          //( _outmatch[output] == -1 ) &&   //allow OC to read multiple reg
          // banks at the same cycle
          (_request[input][output] /*.label != -1*/)) {
        // Grant!
        _inmatch[input] = output;
        _outmatch[output] = input;
        // printf("Register File: granting bank %d to OC %d, schedid %d, warpid
        // %d, Regid %d\n", input, output, (m_queue[input].front()).get_sid(),
        // (m_queue[input].front()).get_wid(),
        // (m_queue[input].front()).get_reg());
      }

      output = (output + 1) % _outputs;
    }
  }

  // Round-robin the priority diagonal
  _pri = (_pri + 1) % _outputs;

  /// <--- end code from booksim

  m_last_cu = _pri;
  for (unsigned i = 0; i < m_num_banks; i++) {
    if (_inmatch[i] != -1) {
      if (!m_allocated_bank[i].is_write()) {
        unsigned bank = (unsigned)i;
        op_t &op = m_queue[bank].front();
        result.push_back(op);
        m_queue[bank].pop_front();
      }
    }
  }

  return result;
}

void opndcoll_rfu_t::add_cu_set(unsigned set_id, unsigned num_cu,
                                unsigned num_dispatch) {
  m_cus[set_id].reserve(num_cu); // this is necessary to stop pointers in m_cu
                                 // from being invalid do to a resize;
  for (unsigned i = 0; i < num_cu; i++) {
    m_cus[set_id].push_back(collector_unit_t());
    m_cu.push_back(&m_cus[set_id].back());
  }
  // for now each collector set gets dedicated dispatch units.
  for (unsigned i = 0; i < num_dispatch; i++) {
    m_dispatch_units.push_back(dispatch_unit_t(&m_cus[set_id]));
  }
}

void opndcoll_rfu_t::add_port(port_vector_t &input, port_vector_t &output,
                              uint_vector_t cu_sets) {
  // m_num_ports++;
  // m_num_collectors += num_collector_units;
  // m_input.resize(m_num_ports);
  // m_output.resize(m_num_ports);
  // m_num_collector_units.resize(m_num_ports);
  // m_input[m_num_ports-1]=input_port;
  // m_output[m_num_ports-1]=output_port;
  // m_num_collector_units[m_num_ports-1]=num_collector_units;
  m_in_ports.push_back(input_port_t(input, output, cu_sets));
}

void opndcoll_rfu_t::init(unsigned num_banks, shader_core_ctx *shader) {
  m_shader = shader;
  m_arbiter.init(m_cu.size(), num_banks);
  // for( unsigned n=0; n<m_num_ports;n++ )
  //    m_dispatch_units[m_output[n]].init( m_num_collector_units[n] );
  m_num_banks = num_banks;
  m_bank_warp_shift = 0;
  m_warp_size = shader->get_config()->warp_size;
  m_bank_warp_shift =
      (unsigned)(int)(std::log(m_warp_size + 0.5) / std::log(2.0));
  assert((m_bank_warp_shift == 5) || (m_warp_size != 32));

  sub_core_model = shader->get_config()->sub_core_model;
  m_num_warp_scheds = shader->get_config()->gpgpu_num_sched_per_core;
  unsigned reg_id;
  if (sub_core_model) {
    assert(num_banks % shader->get_config()->gpgpu_num_sched_per_core == 0);
    assert(m_num_warp_scheds <= m_cu.size() &&
           m_cu.size() % m_num_warp_scheds == 0);
  }
  m_num_banks_per_sched =
      num_banks / shader->get_config()->gpgpu_num_sched_per_core;

  for (unsigned j = 0; j < m_cu.size(); j++) {
    if (sub_core_model) {
      unsigned cusPerSched = m_cu.size() / m_num_warp_scheds;
      reg_id = j / cusPerSched;
    }
    m_cu[j]->init(j, num_banks, m_bank_warp_shift, shader->get_config(), this,
                  sub_core_model, reg_id, m_num_banks_per_sched);
  }
  for (unsigned j = 0; j < m_dispatch_units.size(); j++) {
    m_dispatch_units[j].init(sub_core_model, m_num_warp_scheds);
  }
  m_initialized = true;
}

int register_bank(int regnum, int wid, unsigned num_banks,
                  unsigned bank_warp_shift, bool sub_core_model,
                  unsigned banks_per_sched, unsigned sched_id) {
  int bank = regnum;
  if (bank_warp_shift)
    bank += wid;
  if (sub_core_model) {
    unsigned bank_num = (bank % banks_per_sched) + (sched_id * banks_per_sched);
    assert(bank_num < num_banks);
    return bank_num;
  } else
    return bank % num_banks;
}

bool opndcoll_rfu_t::writeback(warp_inst_t &inst) {
  assert(!inst.empty());
  std::list<unsigned> regs = m_shader->get_regs_written(inst);
  for (unsigned op = 0; op < MAX_REG_OPERANDS; op++) {
    int reg_num = inst.arch_reg.dst[op]; // this math needs to match that used
                                         // in function_info::ptx_decode_inst
    if (reg_num >= 0) {                  // valid register
      unsigned bank = register_bank(reg_num, inst.warp_id(), m_num_banks,
                                    m_bank_warp_shift, sub_core_model,
                                    m_num_banks_per_sched, inst.get_schd_id());
      if (m_arbiter.bank_idle(bank)) {
        m_arbiter.allocate_bank_for_write(
            bank,
            op_t(&inst, reg_num, m_num_banks, m_bank_warp_shift, sub_core_model,
                 m_num_banks_per_sched, inst.get_schd_id()));
        inst.arch_reg.dst[op] = -1;
      } else {
        return false;
      }
    }
  }
  for (unsigned i = 0; i < (unsigned)regs.size(); i++) {
    if (m_shader->get_config()->gpgpu_clock_gated_reg_file) {
      unsigned active_count = 0;
      for (unsigned i = 0; i < m_shader->get_config()->warp_size;
           i = i + m_shader->get_config()->n_regfile_gating_group) {
        for (unsigned j = 0; j < m_shader->get_config()->n_regfile_gating_group;
             j++) {
          if (inst.get_active_mask().test(i + j)) {
            active_count += m_shader->get_config()->n_regfile_gating_group;
            break;
          }
        }
      }
      m_shader->incregfile_writes(active_count);
    } else {
      m_shader->incregfile_writes(
          m_shader->get_config()->warp_size); // inst.active_count());
    }
  }
  return true;
}

void opndcoll_rfu_t::dispatch_ready_cu() {
  for (unsigned p = 0; p < m_dispatch_units.size(); ++p) {
    dispatch_unit_t &du = m_dispatch_units[p];
    collector_unit_t *cu = du.find_ready();
    if (cu) {
      for (unsigned i = 0; i < (cu->get_num_operands() - cu->get_num_regs());
           i++) {
        if (m_shader->get_config()->gpgpu_clock_gated_reg_file) {
          unsigned active_count = 0;
          for (unsigned i = 0; i < m_shader->get_config()->warp_size;
               i = i + m_shader->get_config()->n_regfile_gating_group) {
            for (unsigned j = 0;
                 j < m_shader->get_config()->n_regfile_gating_group; j++) {
              if (cu->get_active_mask().test(i + j)) {
                active_count += m_shader->get_config()->n_regfile_gating_group;
                break;
              }
            }
          }
          m_shader->incnon_rf_operands(active_count);
        } else {
          m_shader->incnon_rf_operands(
              m_shader->get_config()->warp_size); // cu->get_active_count());
        }
      }
      cu->dispatch();
    }
  }
}

void opndcoll_rfu_t::allocate_cu(unsigned port_num) {
  input_port_t &inp = m_in_ports[port_num];
  for (unsigned i = 0; i < inp.m_in.size(); i++) {
    if ((*inp.m_in[i]).has_ready()) {
      // find a free cu
      for (unsigned j = 0; j < inp.m_cu_sets.size(); j++) {
        std::vector<collector_unit_t> &cu_set = m_cus[inp.m_cu_sets[j]];
        bool allocated = false;
        unsigned cuLowerBound = 0;
        unsigned cuUpperBound = cu_set.size();
        unsigned schd_id;
        if (sub_core_model) {
          // Sub core model only allocates on the subset of CUs assigned to the
          // scheduler that issued
          unsigned reg_id = (*inp.m_in[i]).get_ready_reg_id();
          schd_id = (*inp.m_in[i]).get_schd_id(reg_id);
          assert(cu_set.size() % m_num_warp_scheds == 0 &&
                 cu_set.size() >= m_num_warp_scheds);
          unsigned cusPerSched = cu_set.size() / m_num_warp_scheds;
          cuLowerBound = schd_id * cusPerSched;
          cuUpperBound = cuLowerBound + cusPerSched;
          assert(0 <= cuLowerBound && cuUpperBound <= cu_set.size());
        }
        for (unsigned k = cuLowerBound; k < cuUpperBound; k++) {
          if (cu_set[k].is_free()) {
            collector_unit_t *cu = &cu_set[k];
            allocated = cu->allocate(inp.m_in[i], inp.m_out[i]);
            m_arbiter.add_read_requests(cu);
            break;
          }
        }
        if (allocated)
          break; // cu has been allocated, no need to search more.
      }
      // break;  // can only service a single input, if it failed it will fail
      // for
      // others.
    }
  }
}

void opndcoll_rfu_t::allocate_reads() {
  // process read requests that do not have conflicts
  std::list<op_t> allocated = m_arbiter.allocate_reads();
  std::map<unsigned, op_t> read_ops;
  for (std::list<op_t>::iterator r = allocated.begin(); r != allocated.end();
       r++) {
    const op_t &rr = *r;
    unsigned reg = rr.get_reg();
    unsigned wid = rr.get_wid();
    unsigned bank =
        register_bank(reg, wid, m_num_banks, m_bank_warp_shift, sub_core_model,
                      m_num_banks_per_sched, rr.get_sid());
    m_arbiter.allocate_for_read(bank, rr);
    read_ops[bank] = rr;
  }
  std::map<unsigned, op_t>::iterator r;
  for (r = read_ops.begin(); r != read_ops.end(); ++r) {
    op_t &op = r->second;
    unsigned cu = op.get_oc_id();
    unsigned operand = op.get_operand();
    m_cu[cu]->collect_operand(operand);
    if (m_shader->get_config()->gpgpu_clock_gated_reg_file) {
      unsigned active_count = 0;
      for (unsigned i = 0; i < m_shader->get_config()->warp_size;
           i = i + m_shader->get_config()->n_regfile_gating_group) {
        for (unsigned j = 0; j < m_shader->get_config()->n_regfile_gating_group;
             j++) {
          if (op.get_active_mask().test(i + j)) {
            active_count += m_shader->get_config()->n_regfile_gating_group;
            break;
          }
        }
      }
      m_shader->incregfile_reads(active_count);
    } else {
      m_shader->incregfile_reads(
          m_shader->get_config()->warp_size); // op.get_active_count());
    }
  }
}

bool opndcoll_rfu_t::collector_unit_t::ready() const {
  return (!m_free) && m_not_ready.none() &&
         (*m_output_register).has_free(m_sub_core_model, m_reg_id);
}

void opndcoll_rfu_t::collector_unit_t::dump(
    FILE *fp, const shader_core_ctx *shader) const {
  if (m_free) {
    fprintf(fp, "    <free>\n");
  } else {
    m_warp->print(fp);
    for (unsigned i = 0; i < MAX_REG_OPERANDS * 2; i++) {
      if (m_not_ready.test(i)) {
        std::string r = m_src_op[i].get_reg_string();
        fprintf(fp, "    '%s' not ready\n", r.c_str());
      }
    }
  }
}

void opndcoll_rfu_t::collector_unit_t::init(
    unsigned n, unsigned num_banks, unsigned log2_warp_size,
    const core_config *config, opndcoll_rfu_t *rfu, bool sub_core_model,
    unsigned reg_id, unsigned banks_per_sched) {
  m_rfu = rfu;
  m_cuid = n;
  m_num_banks = num_banks;
  assert(m_warp == NULL);
  m_warp = new warp_inst_t(config);
  m_bank_warp_shift = log2_warp_size;
  m_sub_core_model = sub_core_model;
  m_reg_id = reg_id;
  m_num_banks_per_sched = banks_per_sched;
}

bool opndcoll_rfu_t::collector_unit_t::allocate(register_set *pipeline_reg_set,
                                                register_set *output_reg_set) {
  assert(m_free);
  assert(m_not_ready.none());
  m_free = false;
  m_output_register = output_reg_set;
  warp_inst_t **pipeline_reg = pipeline_reg_set->get_ready();
  if ((pipeline_reg) and !((*pipeline_reg)->empty())) {
    m_warp_id = (*pipeline_reg)->warp_id();
    std::vector<int> prev_regs; // remove duplicate regs within same instr
    for (unsigned op = 0; op < MAX_REG_OPERANDS; op++) {
      int reg_num =
          (*pipeline_reg)
              ->arch_reg.src[op]; // this math needs to match that used in
                                  // function_info::ptx_decode_inst
      bool new_reg = true;
      for (auto r : prev_regs) {
        if (r == reg_num)
          new_reg = false;
      }
      if (reg_num >= 0 && new_reg) { // valid register
        prev_regs.push_back(reg_num);
        m_src_op[op] = op_t(this, op, reg_num, m_num_banks, m_bank_warp_shift,
                            m_sub_core_model, m_num_banks_per_sched,
                            (*pipeline_reg)->get_schd_id());
        m_not_ready.set(op);
      } else
        m_src_op[op] = op_t();
    }
    // move_warp(m_warp,*pipeline_reg);
    pipeline_reg_set->move_out_to(m_warp);
    return true;
  }
  return false;
}

void opndcoll_rfu_t::collector_unit_t::dispatch() {
  assert(m_not_ready.none());
  m_output_register->move_in(m_sub_core_model, m_reg_id, m_warp);
  m_free = true;
  m_output_register = NULL;
  for (unsigned i = 0; i < MAX_REG_OPERANDS * 2; i++)
    m_src_op[i].reset();
}
