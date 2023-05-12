#pragma once

#include <bitset>
#include <vector>

typedef unsigned long long new_addr_type;
typedef unsigned long long cudaTextureObject_t;
typedef unsigned long long address_type;
typedef unsigned long long addr_t;

const unsigned MAX_ACCESSES_PER_INSN_PER_THREAD = 8;

const unsigned MAX_MEMORY_ACCESS_SIZE = 128;
typedef std::bitset<MAX_MEMORY_ACCESS_SIZE> mem_access_byte_mask_t;

const unsigned SECTOR_CHUNCK_SIZE = 4; // four sectors
const unsigned SECTOR_SIZE = 32;       // sector is 32 bytes width
typedef std::bitset<SECTOR_CHUNCK_SIZE> mem_access_sector_mask_t;

// bounded stack that implements simt reconvergence using pdom mechanism from
// MICRO'07 paper
const unsigned MAX_WARP_SIZE = 32;
typedef std::bitset<MAX_WARP_SIZE> active_mask_t;
#define MAX_WARP_SIZE_SIMT_STACK MAX_WARP_SIZE
typedef std::bitset<MAX_WARP_SIZE_SIMT_STACK> simt_mask_t;
typedef std::vector<address_type> addr_vector_t;

#define MAX_DEFAULT_CACHE_SIZE_MULTIBLIER 4

// the maximum number of destination, source, or address uarch operands in a
// instruction
#define MAX_REG_OPERANDS 32

// the following are operations the timing model can see
#define SPECIALIZED_UNIT_NUM 8
#define SPEC_UNIT_START_ID 100

enum uarch_op_t {
  NO_OP = -1,
  ALU_OP = 1,
  SFU_OP,
  TENSOR_CORE_OP,
  DP_OP,
  SP_OP,
  INTP_OP,
  ALU_SFU_OP,
  LOAD_OP,
  TENSOR_CORE_LOAD_OP,
  TENSOR_CORE_STORE_OP,
  STORE_OP,
  BRANCH_OP,
  BARRIER_OP,
  MEMORY_BARRIER_OP,
  CALL_OPS,
  RET_OPS,
  EXIT_OPS,
  SPECIALIZED_UNIT_1_OP = SPEC_UNIT_START_ID,
  SPECIALIZED_UNIT_2_OP,
  SPECIALIZED_UNIT_3_OP,
  SPECIALIZED_UNIT_4_OP,
  SPECIALIZED_UNIT_5_OP,
  SPECIALIZED_UNIT_6_OP,
  SPECIALIZED_UNIT_7_OP,
  SPECIALIZED_UNIT_8_OP
};
typedef enum uarch_op_t op_type;

enum uarch_bar_t { NOT_BAR = -1, SYNC = 1, ARRIVE, RED };
typedef enum uarch_bar_t barrier_type;

enum uarch_red_t { NOT_RED = -1, POPC_RED = 1, AND_RED, OR_RED };
typedef enum uarch_red_t reduction_type;

enum uarch_operand_type_t { UN_OP = -1, INT_OP, FP_OP };
typedef enum uarch_operand_type_t types_of_operands;

enum special_operations_t {
  OTHER_OP,
  INT__OP,
  INT_MUL24_OP,
  INT_MUL32_OP,
  INT_MUL_OP,
  INT_DIV_OP,
  FP_MUL_OP,
  FP_DIV_OP,
  FP__OP,
  FP_SQRT_OP,
  FP_LG_OP,
  FP_SIN_OP,
  FP_EXP_OP,
  DP_MUL_OP,
  DP_DIV_OP,
  DP___OP,
  TENSOR__OP,
  TEX__OP
};

typedef enum special_operations_t
    special_ops; // Required to identify for the power model
enum operation_pipeline_t {
  UNKOWN_OP,
  SP__OP,
  DP__OP,
  INTP__OP,
  SFU__OP,
  TENSOR_CORE__OP,
  MEM__OP,
  SPECIALIZED__OP,
};
typedef enum operation_pipeline_t operation_pipeline;
enum mem_operation_t { NOT_TEX, TEX };
typedef enum mem_operation_t mem_operation;

enum _memory_op_t { no_memory_op = 0, memory_load, memory_store };
