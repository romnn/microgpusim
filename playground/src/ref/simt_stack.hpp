#pragma once

#include <deque>

#include "hal.hpp"

class simt_stack {
 public:
  simt_stack(unsigned wid, unsigned warpSize, class trace_gpgpu_sim *gpu);

  void reset();
  void launch(address_type start_pc, const simt_mask_t &active_mask);
  void update(simt_mask_t &thread_done, addr_vector_t &next_pc,
              address_type recvg_pc, op_type next_inst_op,
              unsigned next_inst_size, address_type next_inst_pc);

  const simt_mask_t &get_active_mask() const;
  void get_pdom_stack_top_info(unsigned *pc, unsigned *rpc) const;
  unsigned get_rp() const;
  void print(FILE *fp) const;
  void resume(char *fname);
  void print_checkpoint(FILE *fout) const;

 protected:
  unsigned m_warp_id;
  unsigned m_warp_size;

  enum stack_entry_type { STACK_ENTRY_TYPE_NORMAL = 0, STACK_ENTRY_TYPE_CALL };

  struct simt_stack_entry {
    address_type m_pc;
    unsigned int m_calldepth;
    simt_mask_t m_active_mask;
    address_type m_recvg_pc;
    unsigned long long m_branch_div_cycle;
    stack_entry_type m_type;
    simt_stack_entry()
        : m_pc(-1),
          m_calldepth(0),
          m_active_mask(),
          m_recvg_pc(-1),
          m_branch_div_cycle(0),
          m_type(STACK_ENTRY_TYPE_NORMAL){};
  };

  std::deque<simt_stack_entry> m_stack;

  class trace_gpgpu_sim *m_gpu;
};
