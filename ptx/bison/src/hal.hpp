#pragma once

const unsigned MAX_ACCESSES_PER_INSN_PER_THREAD = 8;

// the maximum number of destination, source, or address uarch
// operands in a instruction
#define MAX_REG_OPERANDS 32

// the following are operations the timing model can see
#define SPECIALIZED_UNIT_NUM 8
#define SPEC_UNIT_START_ID 100

#define MAX_INST_SIZE 8 /*bytes*/

// Set a hard limit of 32 CTAs per shader [cuda only has 8]
#define MAX_CTA_PER_SHADER 32
#define MAX_BARRIERS_PER_CTA 16

// After expanding the vector input and output operands
#define MAX_INPUT_VALUES 24
#define MAX_OUTPUT_VALUES 8

// Let's just upgrade to C++11 so we can use constexpr here...
// start allocating from this address (lower values used for allocating globals
// in .ptx file)
const unsigned long long GLOBAL_HEAP_START = 0xC0000000;
// Volta max shmem size is 96kB
const unsigned long long SHARED_MEM_SIZE_MAX = 96 * (1 << 10);
// Volta max local mem is 16kB
const unsigned long long LOCAL_MEM_SIZE_MAX = 1 << 14;
// Volta Titan V has 80 SMs
const unsigned MAX_STREAMING_MULTIPROCESSORS = 80;
// Max 2048 threads / SM
const unsigned MAX_THREAD_PER_SM = 1 << 11;
// MAX 64 warps / SM
const unsigned MAX_WARP_PER_SM = 1 << 6;
const unsigned long long TOTAL_LOCAL_MEM_PER_SM =
    MAX_THREAD_PER_SM * LOCAL_MEM_SIZE_MAX;
const unsigned long long TOTAL_SHARED_MEM =
    MAX_STREAMING_MULTIPROCESSORS * SHARED_MEM_SIZE_MAX;
const unsigned long long TOTAL_LOCAL_MEM =
    MAX_STREAMING_MULTIPROCESSORS * MAX_THREAD_PER_SM * LOCAL_MEM_SIZE_MAX;
const unsigned long long SHARED_GENERIC_START =
    GLOBAL_HEAP_START - TOTAL_SHARED_MEM;
const unsigned long long LOCAL_GENERIC_START =
    SHARED_GENERIC_START - TOTAL_LOCAL_MEM;
const unsigned long long STATIC_ALLOC_LIMIT =
    GLOBAL_HEAP_START - (TOTAL_LOCAL_MEM + TOTAL_SHARED_MEM);

enum divergence_support_t { POST_DOMINATOR = 1, NUM_SIMD_MODEL };
