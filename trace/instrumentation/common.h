#include <stdint.h>

#define MAX_DST 1
#define MAX_SRC 5

typedef struct {
  uint32_t dest_regs[MAX_DST];
  uint32_t num_dest_regs;
  uint32_t src_regs[MAX_SRC];
  uint32_t num_src_regs;
} reg_info_t;

typedef struct {
  // we identify end of the kernel with block_idx_id = -1
  int block_id_x;
  int block_id_y;
  int block_id_z;
  uint32_t sm_id;
  int warp_id_in_block;
  int warp_id_in_sm;
  uint32_t line_num;
  uint32_t instr_data_width;
  // opcode_id is purely internal
  uint32_t instr_opcode_id;
  uint32_t instr_offset;
  uint32_t instr_idx;
  int instr_predicate_num;
  uint32_t instr_mem_space;

  // boolean instr flags
  bool instr_is_mem;
  bool instr_is_load;
  bool instr_is_store;
  bool instr_is_extended;
  bool instr_predicate_is_neg;
  bool instr_predicate_is_uniform;

  uint32_t warp_size;
  uint32_t active_mask;
  uint32_t predicate_mask;
  uint64_t addrs[32];

  // register info
  uint32_t dest_regs[MAX_DST];
  uint32_t num_dest_regs;
  uint32_t src_regs[MAX_SRC];
  uint32_t num_src_regs;
} mem_access_t;

extern "C" void flush_channel(void *channel_dev);

extern "C" reg_info_t *allocate_reg_info(reg_info_t host_info);

extern "C" void cuda_free(void *dev_ptr);
