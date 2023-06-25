#include "evicted_block_info.hpp"

#include <iostream>

std::ostream &operator<<(std::ostream &os, const evicted_block_info &info) {
  os << "EvictedBlock(";
  os << "block_addr=" << info.m_block_addr;
  os << " ";
  os << "modified size=" << info.m_modified_size;
  os << ")";
  return os;
}
