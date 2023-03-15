#include <cstdarg>
#include <stdint.h>
#include <stdio.h>

#include "utils/channel.hpp"
#include "utils/utils.h"

// contains definition of the mem_access_t structure
#include "common.h"

// Instrumentation function that we want to inject.
// Please note the use of extern "C" __device__ __noinline__
// to prevent "dead"-code elimination by the compiler.
extern "C" __device__ __noinline__ void
instrument_inst(int pred, int instr_opcode_id, uint32_t instr_offset,
                uint32_t instr_idx, int instr_predicate_num,
                bool instr_predicate_is_neg, bool instr_predicate_is_uniform,
                uint32_t instr_mem_space, bool instr_is_load,
                bool instr_is_store, bool instr_is_extended, uint64_t addr,
                uint64_t grid_launch_id, uint64_t pchannel_dev) {

  /* if thread is predicated off, return */
  if (!pred) {
    return;
  }

  int active_mask = __ballot_sync(__activemask(), 1);
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;

  mem_access_t ma;

  /* collect memory address information from other threads */
  for (int i = 0; i < 32; i++) {
    ma.addrs[i] = __shfl_sync(active_mask, addr, i);
  }

  int4 cta = get_ctaid();
  ma.grid_launch_id = grid_launch_id;
  ma.cta_id_x = cta.x;
  ma.cta_id_y = cta.y;
  ma.cta_id_z = cta.z;
  ma.warp_id = get_warpid();
  ma.instr_opcode_id = instr_opcode_id;
  ma.instr_offset = instr_offset;
  ma.instr_idx = instr_idx;
  ma.instr_predicate_num = instr_predicate_num;
  ma.instr_predicate_is_neg = instr_predicate_is_neg;
  ma.instr_predicate_is_uniform = instr_predicate_is_uniform;
  ma.instr_mem_space = instr_mem_space;
  ma.instr_is_load = instr_is_load;
  ma.instr_is_store = instr_is_store;
  ma.instr_is_extended = instr_is_extended;

  /* first active lane pushes information on the channel */
  if (first_laneid == laneid) {
    ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
    channel_dev->push(&ma, sizeof(mem_access_t));
  }
}
