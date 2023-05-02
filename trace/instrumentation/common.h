#include <stdint.h>

extern "C" void flush_channel(void *channel_dev);

typedef struct {
  uint64_t kernel_id;
  // use int here, so we identify end of the kernel with cta_idx_id = -1
  int block_id_x;
  int block_id_y;
  int block_id_z;
  int warp_id;
  // opcode_id is purely internal
  uint32_t instr_opcode_id;
  uint32_t instr_offset;
  uint32_t instr_idx;
  int instr_predicate_num;
  bool instr_predicate_is_neg;
  bool instr_predicate_is_uniform;
  uint32_t instr_mem_space;
  bool instr_is_load;
  bool instr_is_store;
  bool instr_is_extended;
  // addr per thread of a warp?
  uint64_t addrs[32];
} mem_access_t;
