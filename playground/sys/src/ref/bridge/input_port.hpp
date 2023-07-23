#pragma once

#include "../opndcoll_rfu.hpp"
#include "register_set.hpp"

class input_port_bridge {
 public:
  input_port_bridge(const input_port_t *ptr) : ptr(ptr) {}

  const input_port_t *inner() const { return ptr; }

  std::unique_ptr<std::vector<register_set_ptr>> get_in_ports() const {
    std::vector<register_set_ptr> out;
    std::vector<register_set *>::const_iterator iter;
    for (iter = (ptr->m_in).begin(); iter != (ptr->m_in).end(); iter++) {
      out.push_back(register_set_ptr{*iter});
    }
    return std::make_unique<std::vector<register_set_ptr>>(out);
  }

  std::unique_ptr<std::vector<register_set_ptr>> get_out_ports() const {
    std::vector<register_set_ptr> out;
    std::vector<register_set *>::const_iterator iter;
    for (iter = (ptr->m_out).begin(); iter != (ptr->m_out).end(); iter++) {
      out.push_back(register_set_ptr{*iter});
    }
    return std::make_unique<std::vector<register_set_ptr>>(out);
  }

  const uint_vector_t &get_cu_sets() const { return ptr->m_cu_sets; }

 private:
  const class input_port_t *ptr;
};

std::shared_ptr<input_port_bridge> new_input_port_bridge(
    const input_port_t *ptr);
