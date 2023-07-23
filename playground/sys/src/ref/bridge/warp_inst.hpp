#pragma once

#include "../warp_instr.hpp"

struct warp_inst_ptr {
  const warp_inst_t *ptr;
  const warp_inst_t *get() const { return ptr; }
};

class warp_inst_bridge {
 public:
  warp_inst_bridge(const warp_inst_t *ptr) : ptr(ptr){};

  const warp_inst_t *inner() const { return ptr; }

 private:
  const warp_inst_t *ptr;
};

std::shared_ptr<warp_inst_bridge> new_warp_inst_bridge(const warp_inst_t *ptr);
