#include <cassert>
#include <cstdarg>
#include <stdint.h>
#include <stdio.h>

#include "utils/channel.hpp"
#include "utils/utils.h"

// contains definition of the mem_access_t structure
#include "common.h"

#define GET_BIT(value, pos) (((1 << (pos)) & (value)) >> (pos))

// Instrumentation function that we want to inject.
//
// Note the use of `extern "C" __device__ __noinline__`
// to prevent "dead"-code elimination by the compiler.
//
// Note: CUDA functions only allowf or at most 11 arguments of <= 256 bytes
// each. For more or larger arguments fall back to passing device allocated
// pointers.
extern "C" __device__ __noinline__ void
instrument_inst(uint32_t pred, uint32_t instr_data_width,
                uint32_t instr_opcode_id, uint32_t instr_offset,
                uint32_t instr_idx, uint32_t line_num, uint32_t instr_mem_space,
                uint32_t instr_predicate_num, uint32_t instr_flags,
                uint64_t ptr_reg_info, uint64_t addr, uint64_t ptr_channel_dev,
                uint64_t kernel_id) {

  // if thread is predicated off, do NOT return!
  // otherwise we end up with different number of instructions for each thread !
  // if (!pred) {
  //   return;
  // }

  const int active_mask = __ballot_sync(__activemask(), 1);
  const int predicate_mask = __ballot_sync(__activemask(), pred);
  // A predefined, read-only special register that returns the thread’s lane
  // within the warp. The lane identifier ranges from zero to WARP_SZ-1.
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;

  mem_access_t ma;

  bool instr_is_mem = (bool)GET_BIT(instr_flags, 0);
  bool instr_is_load = (bool)GET_BIT(instr_flags, 1);
  bool instr_is_store = (bool)GET_BIT(instr_flags, 2);
  bool instr_is_extended = (bool)GET_BIT(instr_flags, 3);
  bool instr_predicate_is_neg = (bool)GET_BIT(instr_flags, 4);
  bool instr_predicate_is_uniform = (bool)GET_BIT(instr_flags, 5);

  // collect memory address information from other threads
  for (int i = 0; i < 32; i++) {
    if (instr_is_mem) {
      ma.addrs[i] = __shfl_sync(active_mask, addr, i);
    } else {
      ma.addrs[i] = 0;
    }
  }

  ma.kernel_id = kernel_id;

  int4 block = get_ctaid();
  ma.block_id_x = block.x;
  ma.block_id_y = block.y;
  ma.block_id_z = block.z;
  assert(blockIdx.x == block.x);
  assert(blockIdx.y == block.y);
  assert(blockIdx.z == block.z);

  ma.sm_id = get_smid();
  // int unique_thread_id = threadIdx.z * (blockDim.y * blockDim.x) +
  //                        threadIdx.y * blockDim.x + threadIdx.x;
  int unique_thread_id = (threadIdx.z * (blockDim.x * blockDim.y)) +
                         (threadIdx.y * blockDim.x) + threadIdx.x;

  // int unique_thread_id = (threadIdx.x * (blockDim.y * blockDim.z)) +
  //                        (threadIdx.y * blockDim.z) + threadIdx.z;
  // unique_thread_id += threadIdx.z;
  // int thread_id = (threadIdx.x * threadIdx.x(blockDim.x *
  // blockDim.y)) +

  // int l_thread_id = (threadIdx.z * (blockDim.x * blockDim.y))
  // +
  //                   (threadIdx.y * blockDim.x) + threadIdx.x;

  ma.thread_id = unique_thread_id;
  ma.thread_id_x = threadIdx.x;
  ma.thread_id_y = threadIdx.y;
  ma.thread_id_z = threadIdx.z;

  ma.global_warp_id = get_global_warp_id();
  ma.warp_id_in_block = unique_thread_id / warpSize;
  ma.warp_id_in_sm = get_warpid();
  ma.warp_size = warpSize;
  ma.line_num = line_num;
  ma.instr_data_width = instr_data_width;
  ma.instr_opcode_id = instr_opcode_id;
  ma.instr_offset = instr_offset;
  ma.instr_idx = instr_idx;
  ma.instr_predicate_num = instr_predicate_num;
  ma.instr_mem_space = instr_mem_space;
  ma.instr_is_mem = instr_is_mem;
  ma.instr_is_load = instr_is_load;
  ma.instr_is_store = instr_is_store;
  ma.instr_is_extended = instr_is_extended;
  ma.instr_predicate_is_neg = instr_predicate_is_neg;
  ma.instr_predicate_is_uniform = instr_predicate_is_uniform;

  ma.active_mask = active_mask;
  ma.predicate_mask = predicate_mask;

  // register info
  reg_info_t *reg_info = (reg_info_t *)ptr_reg_info;
  for (int r = 0; r < MAX_DST; r++) {
    ma.dest_regs[r] = reg_info->dest_regs[r];
  }
  ma.num_dest_regs = reg_info->num_dest_regs;

  for (int r = 0; r < MAX_SRC; r++) {
    ma.src_regs[r] = reg_info->src_regs[r];
  }
  ma.num_src_regs = reg_info->num_src_regs;

  // first active lane pushes information on the channel
  if (first_laneid == laneid) {
    ChannelDev *channel_dev = (ChannelDev *)ptr_channel_dev;
    channel_dev->push(&ma, sizeof(mem_access_t));
  }
}
