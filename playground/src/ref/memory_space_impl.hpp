#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <unordered_map>

#include "hal.hpp"
#include "mem_storage.hpp"
#include "memory_space.hpp"

class ptx_instruction;

template <unsigned BSIZE> class memory_space_impl : public memory_space {
public:
  memory_space_impl(std::string name, unsigned hash_size);

  virtual void write(mem_addr_t addr, size_t length, const void *data,
                     ptx_thread_info *thd, const ptx_instruction *pI);
  virtual void write_only(mem_addr_t index, mem_addr_t offset, size_t length,
                          const void *data);
  virtual void read(mem_addr_t addr, size_t length, void *data) const;
  virtual void print(const char *format, FILE *fout) const;

  virtual void set_watch(addr_t addr, unsigned watchpoint);

private:
  void read_single_block(mem_addr_t blk_idx, mem_addr_t addr, size_t length,
                         void *data) const;
  std::string m_name;
  unsigned m_log2_block_size;

  typedef std::unordered_map<mem_addr_t, mem_storage<BSIZE>> map_t;
  map_t m_data;
  std::map<unsigned, mem_addr_t> m_watchpoints;
};
