#pragma once

#include "../opndcoll_rfu.hpp"

struct collector_unit_set {
  unsigned collector_set;
  const collector_unit_t &unit;
  const collector_unit_t &get_unit() const { return unit; }
  unsigned get_set() const { return collector_set; }
};

class operand_collector_bridge {
 public:
  operand_collector_bridge(const opndcoll_rfu_t *ptr) : ptr(ptr) {}

  const opndcoll_rfu_t *inner() const { return ptr; }

  const arbiter_t &get_arbiter() const { return ptr->m_arbiter; }

  const std::vector<input_port_t> &get_input_ports() const {
    return ptr->m_in_ports;
  }

  const std::vector<dispatch_unit_t> &get_dispatch_units() const {
    return ptr->m_dispatch_units;
  }

  std::unique_ptr<std::vector<collector_unit_set>> get_collector_units() const {
    // collector set, collector sets
    std::vector<collector_unit_set> out;
    std::map<unsigned, std::vector<collector_unit_t>>::const_iterator iter;
    for (iter = (ptr->m_cus).begin(); iter != (ptr->m_cus).end(); iter++) {
      const std::vector<collector_unit_t> &cus = iter->second;
      std::vector<collector_unit_t>::const_iterator cu;
      for (cu = cus.begin(); cu != cus.end(); cu++) {
        out.push_back(collector_unit_set{iter->first, *cu});
      }
    }
    return std::make_unique<std::vector<collector_unit_set>>(out);
  }

 private:
  const class opndcoll_rfu_t *ptr;
};
