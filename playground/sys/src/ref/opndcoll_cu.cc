#include "opndcoll_rfu.hpp"

#include <sstream>
#include "io.hpp"
#include "register_set.hpp"

collector_unit_t::collector_unit_t(unsigned set_id) {
  m_set_id = set_id;
  m_free = true;
  m_warp = NULL;
  m_output_register = NULL;
  m_src_op = new op_t[MAX_REG_OPERANDS * 2];
  m_not_ready.reset();
  m_warp_id = -1;
  m_reg_id = 0;
  m_num_banks = 0;
  m_bank_warp_shift = 0;
}

bool collector_unit_t::ready() const {
  if (m_free) {
    // to make the print not segfault
    return false;
  }
  printf("is ready?: active = %s (ready=%d), has free = %d",
         mask_to_string(m_not_ready).c_str(), m_not_ready.none(),
         (*m_output_register).has_free(m_sub_core_model, m_reg_id));
  std::cout << " output register = " << (*m_output_register) << std::endl;

  return (!m_free) && m_not_ready.none() &&
         (*m_output_register).has_free(m_sub_core_model, m_reg_id);
}

// void opndcoll_rfu_t::collector_unit_t::dump(
//     FILE *fp, const trace_shader_core_ctx *shader) const {
//   if (m_free) {
//     fprintf(fp, "    <free>\n");
//   } else {
//     m_warp->print(fp);
//     for (unsigned i = 0; i < MAX_REG_OPERANDS * 2; i++) {
//       if (m_not_ready.test(i)) {
//         std::string r = m_src_op[i].get_reg_string();
//         fprintf(fp, "    '%s' not ready\n", r.c_str());
//       }
//     }
//   }
// }

void collector_unit_t::init(unsigned n, unsigned num_banks,
                            unsigned log2_warp_size, const core_config *config,
                            opndcoll_rfu_t *rfu, bool sub_core_model,
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

bool collector_unit_t::allocate(register_set *pipeline_reg_set,
                                register_set *output_reg_set) {
  printf("operand collector::allocate(%s)\n",
         operand_collector_unit_kind_str[m_set_id]);
  assert(m_free);
  assert(m_not_ready.none());
  m_free = false;
  m_output_register = output_reg_set;
  warp_inst_t **pipeline_reg = pipeline_reg_set->get_ready();
  if ((pipeline_reg) and !((*pipeline_reg)->empty())) {
    const int *arch_reg_src = ((*pipeline_reg)->arch_reg).src;  // int[32]
    std::vector<int> arch_reg_src_vec(arch_reg_src, arch_reg_src + 32);
    printf("operand collector::allocate(%s)",
           operand_collector_unit_kind_str[m_set_id]);
    std::cout << " => src arch reg = " << arch_reg_src_vec << std::endl;
    m_warp_id = (*pipeline_reg)->warp_id();
    std::vector<int> prev_regs;  // remove duplicate regs within same instr
    for (unsigned op = 0; op < MAX_REG_OPERANDS; op++) {
      int reg_num =
          (*pipeline_reg)
              ->arch_reg.src[op];  // this math needs to match that used in
                                   // function_info::ptx_decode_inst
      bool new_reg = true;
      for (auto r : prev_regs) {
        if (r == reg_num) new_reg = false;
      }
      if (reg_num >= 0 && new_reg) {  // valid register
        prev_regs.push_back(reg_num);
        m_src_op[op] = op_t(this, op, reg_num, m_num_banks, m_bank_warp_shift,
                            m_sub_core_model, m_num_banks_per_sched,
                            (*pipeline_reg)->get_schd_id());
        // assert(0 && "setting op as not ready");
        m_not_ready.set(op);
      } else
        m_src_op[op] = op_t();
    }
    std::cout << "operand collector::allocate() => active = "
              << mask_to_string(m_not_ready) << std::endl;

    // move_warp(m_warp,*pipeline_reg);
    std::stringstream msg;
    msg << "operand collector: move input register "
        << pipeline_reg_set->get_name() << " to warp instruction " << m_warp;
    pipeline_reg_set->move_out_to(m_warp, msg.str());
    return true;
  }
  return false;
}

void collector_unit_t::dispatch() {
  assert(m_not_ready.none());
  std::stringstream msg;
  msg << "operand collector: move warp instr " << m_warp
      << " to output register (reg id=" << m_reg_id << ")";

  m_output_register->move_in(m_sub_core_model, m_reg_id, m_warp, msg.str());

  m_free = true;
  m_output_register = NULL;
  for (unsigned i = 0; i < MAX_REG_OPERANDS * 2; i++) m_src_op[i].reset();
}
