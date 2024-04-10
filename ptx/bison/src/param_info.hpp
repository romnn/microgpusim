#pragma once

#include <string>

#include "memory_space.hpp"

struct param_t {
  const void *pdata;
  int type;
  size_t size;
  size_t offset;
};

class param_info {
public:
  param_info() {
    m_valid = false;
    m_value_set = false;
    m_size = 0;
    m_is_ptr = false;
  }
  param_info(std::string name, int type, size_t size, bool is_ptr,
             memory_space_t ptr_space) {
    m_valid = true;
    m_value_set = false;
    m_name = name;
    m_type = type;
    m_size = size;
    m_is_ptr = is_ptr;
    m_ptr_space = ptr_space;
  }
  void add_data(param_t v) {
    assert((!m_value_set) ||
           (m_value.size == v.size)); // if this fails concurrent kernel
                                      // launches might execute incorrectly
    m_value_set = true;
    m_value = v;
  }
  void add_offset(unsigned offset) { m_offset = offset; }
  unsigned get_offset() {
    assert(m_valid);
    return m_offset;
  }
  std::string get_name() const {
    assert(m_valid);
    return m_name;
  }
  int get_type() const {
    assert(m_valid);
    return m_type;
  }
  param_t get_value() const {
    assert(m_value_set);
    return m_value;
  }
  size_t get_size() const {
    assert(m_valid);
    return m_size;
  }
  bool is_ptr_shared() const {
    assert(m_valid);
    return (m_is_ptr and m_ptr_space == shared_space);
  }

private:
  bool m_valid;
  std::string m_name;
  int m_type;
  size_t m_size;
  bool m_value_set;
  param_t m_value;
  unsigned m_offset;
  bool m_is_ptr;
  memory_space_t m_ptr_space;
};
