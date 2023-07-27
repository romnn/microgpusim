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

  // std::unique_ptr<std::vector<new_addr_type>> get_addresses(
  //     unsigned warp_id) const {
  //   assert(ptr->m_per_scalar_thread_valid);
  //   const new_addr_type *memreqaddr =
  //       ptr->m_per_scalar_thread[warp_id].memreqaddr;
  //   std::vector<new_addr_type> v(memreqaddr,
  //                                memreqaddr +
  //                                MAX_ACCESSES_PER_INSN_PER_THREAD);
  //   return std::make_unique<std::vector<new_addr_type>>(v);
  // }

 private:
  const warp_inst_t *ptr;
};

std::shared_ptr<warp_inst_bridge> new_warp_inst_bridge(const warp_inst_t *ptr);
