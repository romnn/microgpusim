#include "operand_info.hpp"

#include "gpgpu_context.hpp"
#include "symbol.hpp"

unsigned operand_info::get_uid() {
  unsigned result = (gpgpu_ctx->operand_info_sm_next_uid)++;
  return result;
}

operand_info::operand_info(const symbol *addr, gpgpu_context *ctx) {
  init(ctx);
  m_is_non_arch_reg = false;
  m_addr_space = undefined_space;
  m_operand_lohi = 0;
  m_double_operand_type = 0;
  m_operand_neg = false;
  m_const_mem_offset = 0;
  m_uid = get_uid();
  m_valid = true;
  if (addr->is_label()) {
    m_type = label_t;
  } else if (addr->is_shared()) {
    m_type = symbolic_t;
  } else if (addr->is_const()) {
    m_type = symbolic_t;
  } else if (addr->is_global()) {
    m_type = symbolic_t;
  } else if (addr->is_local()) {
    m_type = symbolic_t;
  } else if (addr->is_param_local()) {
    m_type = symbolic_t;
  } else if (addr->is_param_kernel()) {
    m_type = symbolic_t;
  } else if (addr->is_tex()) {
    m_type = symbolic_t;
  } else if (addr->is_func_addr()) {
    m_type = symbolic_t;
  } else if (!addr->is_reg()) {
    m_type = symbolic_t;
  } else {
    m_type = reg_t;
  }

  m_is_non_arch_reg = addr->is_non_arch_reg();
  m_value.m_symbolic = addr;
  m_addr_offset = 0;
  m_vector = false;
  m_neg_pred = false;
  m_is_return_var = false;
  m_immediate_address = false;
}

const std::string &operand_info::name() const {
  assert(m_type == symbolic_t || m_type == reg_t || m_type == address_t ||
         m_type == memory_t || m_type == label_t);
  return m_value.m_symbolic->name();
}

const std::string &operand_info::vec_name1() const {
  assert(m_type == vector_t);
  return m_value.m_vector_symbolic[0]->name();
}

const std::string &operand_info::vec_name2() const {
  assert(m_type == vector_t);
  return m_value.m_vector_symbolic[1]->name();
}

const std::string &operand_info::vec_name3() const {
  assert(m_type == vector_t);
  return m_value.m_vector_symbolic[2]->name();
}

const std::string &operand_info::vec_name4() const {
  assert(m_type == vector_t);
  return m_value.m_vector_symbolic[3]->name();
}

bool operand_info::is_reg() const {
  if (m_type == reg_t) {
    return true;
  }
  if (m_type != symbolic_t) {
    return false;
  }
  return m_value.m_symbolic->type()->get_key().is_reg();
}

bool operand_info::is_param_local() const {
  if (m_type != symbolic_t)
    return false;
  return m_value.m_symbolic->type()->get_key().is_param_local();
}

bool operand_info::is_param_kernel() const {
  if (m_type != symbolic_t)
    return false;
  return m_value.m_symbolic->type()->get_key().is_param_kernel();
}

int operand_info::reg_num() const { return m_value.m_symbolic->reg_num(); }

int operand_info::reg1_num() const {
  return m_value.m_vector_symbolic[0]->reg_num();
}

int operand_info::reg2_num() const {
  return m_value.m_vector_symbolic[1]->reg_num();
}

int operand_info::reg3_num() const {
  return m_value.m_vector_symbolic[2] ? m_value.m_vector_symbolic[2]->reg_num()
                                      : 0;
}

int operand_info::reg4_num() const {
  return m_value.m_vector_symbolic[3] ? m_value.m_vector_symbolic[3]->reg_num()
                                      : 0;
}

int operand_info::reg5_num() const {
  return m_value.m_vector_symbolic[4] ? m_value.m_vector_symbolic[4]->reg_num()
                                      : 0;
}

int operand_info::reg6_num() const {
  return m_value.m_vector_symbolic[5] ? m_value.m_vector_symbolic[5]->reg_num()
                                      : 0;
}

int operand_info::reg7_num() const {
  return m_value.m_vector_symbolic[6] ? m_value.m_vector_symbolic[6]->reg_num()
                                      : 0;
}

int operand_info::reg8_num() const {
  return m_value.m_vector_symbolic[7] ? m_value.m_vector_symbolic[7]->reg_num()
                                      : 0;
}

int operand_info::arch_reg_num() const {
  return m_value.m_symbolic->arch_reg_num();
}

int operand_info::arch_reg_num(unsigned n) const {
  return (m_value.m_vector_symbolic[n])
             ? m_value.m_vector_symbolic[n]->arch_reg_num()
             : -1;
}

bool operand_info::is_shared() const {
  if (!(m_type == symbolic_t || m_type == address_t || m_type == memory_t)) {
    return false;
  }
  return m_value.m_symbolic->is_shared();
}

bool operand_info::is_sstarr() const { return m_value.m_symbolic->is_sstarr(); }

bool operand_info::is_const() const { return m_value.m_symbolic->is_const(); }

bool operand_info::is_global() const { return m_value.m_symbolic->is_global(); }

bool operand_info::is_local() const { return m_value.m_symbolic->is_local(); }

bool operand_info::is_tex() const { return m_value.m_symbolic->is_tex(); }

bool operand_info::is_return_var() const { return m_is_return_var; }

bool operand_info::is_function_address() const {
  if (m_type != symbolic_t) {
    return false;
  }
  return m_value.m_symbolic->is_func_addr();
}
