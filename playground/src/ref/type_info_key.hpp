#pragma once

#include <assert.h>
#include <stddef.h>

#include "memory_space.hpp"

class type_info_key {
public:
  type_info_key() {
    m_is_non_arch_reg = false;
    m_init = false;
  }
  type_info_key(memory_space_t space_spec, int scalar_type_spec,
                int vector_spec, int alignment_spec, int extern_spec,
                int array_dim) {
    m_is_non_arch_reg = false;
    m_init = true;
    m_space_spec = space_spec;
    m_scalar_type_spec = scalar_type_spec;
    m_vector_spec = vector_spec;
    m_alignment_spec = alignment_spec;
    m_extern_spec = extern_spec;
    m_array_dim = array_dim;
    m_is_function = 0;
  }
  void set_is_func() {
    assert(!m_init);
    m_init = true;
    m_space_spec = undefined_space;
    m_scalar_type_spec = 0;
    m_vector_spec = 0;
    m_alignment_spec = 0;
    m_extern_spec = 0;
    m_array_dim = 0;
    m_is_function = 1;
  }

  void set_array_dim(int array_dim) { m_array_dim = array_dim; }
  int get_array_dim() const {
    assert(m_init);
    return m_array_dim;
  }
  void set_is_non_arch_reg() { m_is_non_arch_reg = true; }

  bool is_non_arch_reg() const { return m_is_non_arch_reg; }
  bool is_reg() const { return m_space_spec == reg_space; }
  bool is_param_kernel() const { return m_space_spec == param_space_kernel; }
  bool is_param_local() const { return m_space_spec == param_space_local; }
  bool is_param_unclassified() const {
    return m_space_spec == param_space_unclassified;
  }
  bool is_global() const { return m_space_spec == global_space; }
  bool is_local() const { return m_space_spec == local_space; }
  bool is_shared() const { return m_space_spec == shared_space; }
  bool is_const() const { return m_space_spec.get_type() == const_space; }
  bool is_tex() const { return m_space_spec == tex_space; }
  bool is_func_addr() const { return m_is_function ? true : false; }
  int scalar_type() const { return m_scalar_type_spec; }
  int get_alignment_spec() const { return m_alignment_spec; }
  unsigned type_decode(size_t &size, int &t) const;
  static unsigned type_decode(int type, size_t &size, int &t);
  memory_space_t get_memory_space() const { return m_space_spec; }

private:
  bool m_init;
  memory_space_t m_space_spec;
  int m_scalar_type_spec;
  int m_vector_spec;
  int m_alignment_spec;
  int m_extern_spec;
  int m_array_dim;
  int m_is_function;
  bool m_is_non_arch_reg;

  friend struct type_info_key_compare;
};

struct type_info_key_compare {
  bool operator()(const type_info_key &a, const type_info_key &b) const {
    assert(a.m_init && b.m_init);
    if (a.m_space_spec < b.m_space_spec)
      return true;
    if (a.m_scalar_type_spec < b.m_scalar_type_spec)
      return true;
    if (a.m_vector_spec < b.m_vector_spec)
      return true;
    if (a.m_alignment_spec < b.m_alignment_spec)
      return true;
    if (a.m_extern_spec < b.m_extern_spec)
      return true;
    if (a.m_array_dim < b.m_array_dim)
      return true;
    if (a.m_is_function < b.m_is_function)
      return true;

    return false;
  }
};
