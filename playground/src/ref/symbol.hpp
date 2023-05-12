#pragma once

#include <list>

// #include "gpgpu_context.hpp"
#include "hal.hpp"
#include "gpgpu_context.hpp"
#include "type_info.hpp"
// #include "operand_info.hpp"
// #include "function_info.hpp"

class operand_info;
class function_info;
class gpgpu_context;

class symbol {
 public:
  symbol(const char *name, const type_info *type, const char *location,
         unsigned size, gpgpu_context *ctx) {
    gpgpu_ctx = ctx;
    m_uid = get_uid();
    m_name = name;
    m_decl_location = location;
    m_type = type;
    m_size = size;
    m_address_valid = false;
    m_is_label = false;
    m_is_shared = false;
    m_is_const = false;
    m_is_global = false;
    m_is_local = false;
    m_is_param_local = false;
    m_is_param_kernel = false;
    m_is_tex = false;
    m_is_func_addr = false;
    m_reg_num_valid = false;
    m_function = NULL;
    m_reg_num = (unsigned)-1;
    m_arch_reg_num = (unsigned)-1;
    m_address = (unsigned)-1;
    m_initializer.clear();
    if (type) m_is_shared = type->get_key().is_shared();
    if (type) m_is_const = type->get_key().is_const();
    if (type) m_is_global = type->get_key().is_global();
    if (type) m_is_local = type->get_key().is_local();
    if (type) m_is_param_local = type->get_key().is_param_local();
    if (type) m_is_param_kernel = type->get_key().is_param_kernel();
    if (type) m_is_tex = type->get_key().is_tex();
    if (type) m_is_func_addr = type->get_key().is_func_addr();
  }
  unsigned get_size_in_bytes() const { return m_size; }
  const std::string &name() const { return m_name; }
  const std::string &decl_location() const { return m_decl_location; }
  const type_info *type() const { return m_type; }
  addr_t get_address() const {
    assert(m_is_label ||
           !m_type->get_key().is_reg());  // todo : other assertions
    assert(m_address_valid);
    return m_address;
  }
  function_info *get_pc() const { return m_function; }
  void set_regno(unsigned regno, unsigned arch_regno) {
    m_reg_num_valid = true;
    m_reg_num = regno;
    m_arch_reg_num = arch_regno;
  }

  void set_address(addr_t addr) {
    m_address_valid = true;
    m_address = addr;
  }
  void set_label_address(addr_t addr) {
    m_address_valid = true;
    m_address = addr;
    m_is_label = true;
  }
  void set_function(function_info *func) {
    m_function = func;
    m_is_func_addr = true;
  }

  bool is_label() const { return m_is_label; }
  bool is_shared() const { return m_is_shared; }
  bool is_sstarr() const { return m_is_sstarr; }
  bool is_const() const { return m_is_const; }
  bool is_global() const { return m_is_global; }
  bool is_local() const { return m_is_local; }
  bool is_param_local() const { return m_is_param_local; }
  bool is_param_kernel() const { return m_is_param_kernel; }
  bool is_tex() const { return m_is_tex; }
  bool is_func_addr() const { return m_is_func_addr; }
  bool is_reg() const {
    if (m_type == NULL) {
      return false;
    }
    return m_type->get_key().is_reg();
  }
  bool is_non_arch_reg() const {
    if (m_type == NULL) {
      return false;
    }
    return m_type->get_key().is_non_arch_reg();
  }

  void add_initializer(const std::list<operand_info> &init);
  bool has_initializer() const { return m_initializer.size() > 0; }
  std::list<operand_info> get_initializer() const { return m_initializer; }
  unsigned reg_num() const {
    assert(m_reg_num_valid);
    return m_reg_num;
  }
  unsigned arch_reg_num() const {
    assert(m_reg_num_valid);
    return m_arch_reg_num;
  }
  void print_info(FILE *fp) const;
  unsigned uid() const { return m_uid; }

 private:
  gpgpu_context *gpgpu_ctx;
  unsigned get_uid();
  unsigned m_uid;
  const type_info *m_type;
  unsigned m_size;  // in bytes
  std::string m_name;
  std::string m_decl_location;

  unsigned m_address;
  function_info *m_function;  // used for function symbols

  bool m_address_valid;
  bool m_is_label;
  bool m_is_shared;
  bool m_is_sstarr;
  bool m_is_const;
  bool m_is_global;
  bool m_is_local;
  bool m_is_param_local;
  bool m_is_param_kernel;
  bool m_is_tex;
  bool m_is_func_addr;
  unsigned m_reg_num;
  unsigned m_arch_reg_num;
  bool m_reg_num_valid;

  std::list<operand_info> m_initializer;
};
