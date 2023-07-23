#pragma once

#include "../register_set.hpp"
#include "warp_inst.hpp"

struct register_set_ptr {
  const register_set *ptr;
  const register_set *get() const { return ptr; }
};

class register_set_bridge {
 public:
  register_set_bridge(const register_set *ptr) : ptr(ptr) {}

  const register_set *inner() const { return ptr; };

  std::unique_ptr<std::vector<warp_inst_ptr>> get_registers() const {
    std::vector<warp_inst_ptr> out;
    std::vector<warp_inst_t *>::const_iterator iter;
    for (iter = (ptr->regs).begin(); iter != (ptr->regs).end(); iter++) {
      out.push_back(warp_inst_ptr{*iter});
    }
    return std::make_unique<std::vector<warp_inst_ptr>>(out);
  }

 private:
  const register_set *ptr;
};

std::shared_ptr<register_set_bridge> new_register_set_bridge(
    const register_set *ptr);
