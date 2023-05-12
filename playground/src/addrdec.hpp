#pragma once

#include "hal.hpp"
#include <unordered_map>

extern std::unordered_map<new_addr_type, unsigned> address_random_interleaving;

long int powli(long int x, long int y);
unsigned int LOGB2_32(unsigned int v);
unsigned next_powerOf2(unsigned n);

new_addr_type addrdec_packbits(new_addr_type mask, new_addr_type val,
                               unsigned char high, unsigned char low);
void addrdec_getmasklimit(new_addr_type mask, unsigned char *high,
                          unsigned char *low);

enum partition_index_function {
  CONSECUTIVE = 0,
  BITWISE_PERMUTATION,
  IPOLY,
  PAE,
  RANDOM,
  CUSTOM
};

struct addrdec_t {
  void print(FILE *fp) const;

  unsigned chip;
  unsigned bk;
  unsigned row;
  unsigned col;
  unsigned burst;

  unsigned sub_partition;
};

class linear_to_raw_address_translation {
public:
  linear_to_raw_address_translation();
  // void addrdec_setoption(option_parser_t opp);
  void init(unsigned int n_channel, unsigned int n_sub_partition_in_channel);

  // accessors
  void addrdec_tlx(new_addr_type addr, addrdec_t *tlx) const;
  new_addr_type partition_address(new_addr_type addr) const;

private:
  void addrdec_parseoption(const char *option);
  void sweep_test() const; // sanity check to ensure no overlapping

  enum { CHIP = 0, BK = 1, ROW = 2, COL = 3, BURST = 4, N_ADDRDEC };

  const char *addrdec_option;
  int gpgpu_mem_address_mask;
  partition_index_function memory_partition_indexing;
  bool run_test;

  int ADDR_CHIP_S;
  unsigned char addrdec_mklow[N_ADDRDEC];
  unsigned char addrdec_mkhigh[N_ADDRDEC];
  new_addr_type addrdec_mask[N_ADDRDEC];
  new_addr_type sub_partition_id_mask;

  unsigned int gap;
  unsigned m_n_channel;
  int m_n_sub_partition_in_channel;
  int m_n_sub_partition_total;
  unsigned log2channel;
  unsigned log2sub_partition;
  unsigned nextPowerOf2_m_n_channel;
};
