#pragma once

#include "addrdec.hpp"
#include "hal.hpp"
#include <stdio.h>

#include "cache_config.hpp"
#include "dim3.hpp"
#include "instr.hpp"
#include "mem_fetch.hpp"
#include "memory_config.hpp"
#include "tag_array.hpp"
#include "warp_instr.hpp"

// typedef struct CacheConfig {
//   char ct;
//   unsigned m_nset;
//   unsigned m_line_sz;
//   unsigned m_assoc;
//   //
//   char rp;
//   char wp;
//   char ap;
//   char wap;
//   char sif;
//   //
//   char mshr_type;
//   unsigned m_mshr_entries;
//   unsigned m_mshr_max_merge;
//   unsigned m_miss_queue_size;
//   unsigned m_result_fifo_entries;
//   unsigned m_data_port_width;
// } cache_config;
//
// extern "C" void parse_cache_config(char *config, cache_config *dest);

// typedef unsigned long long new_addr_type;
// typedef unsigned long long address_type;
// typedef unsigned long long addr_t;

// extern "C" new_addr_type addrdec_packbits(new_addr_type mask, new_addr_type
// val,
//                                           unsigned char high,
//                                           unsigned char low);
//
// extern "C" unsigned int LOGB2_32(unsigned int v) {
//   unsigned int shift;
//   unsigned int r;
//
//   r = 0;
//
//   shift = ((v & 0xFFFF0000) != 0) << 4;
//   v >>= shift;
//   r |= shift;
//   shift = ((v & 0xFF00) != 0) << 3;
//   v >>= shift;
//   r |= shift;
//   shift = ((v & 0xF0) != 0) << 2;
//   v >>= shift;
//   r |= shift;
//   shift = ((v & 0xC) != 0) << 1;
//   v >>= shift;
//   r |= shift;
//   shift = ((v & 0x2) != 0) << 0;
//   v >>= shift;
//   r |= shift;
//
//   return r;
// }
//
// extern "C" unsigned next_powerOf2(unsigned n) {
//   // decrement n (to handle the case when n itself
//   // is a power of 2)
//   n = n - 1;
//
//   // do till only one bit is left
//   while (n & n - 1)
//     n = n & (n - 1); // unset rightmost bit
//
//   // n is now a power of two (less than n)
//
//   // return next power of 2
//   return n << 1;
// }
//
// // compute x to the y
// extern "C" long int powli(long int x, long int y) {
//   long int r = 1;
//   int i;
//   for (i = 0; i < y; ++i) {
//     r *= x;
//   }
//   return r;
// }
