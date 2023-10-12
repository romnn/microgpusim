#pragma once

#include <bitset>

#include "gpgpu_context.hpp"
#include "hal.hpp"
#include "mem_access_type.hpp"

class mem_access_t {
 public:
  mem_access_t(gpgpu_context *ctx) { init(ctx); }
  mem_access_t(mem_access_type type, new_addr_type address, unsigned size,
               bool wr, gpgpu_context *ctx) {
    init(ctx);
    m_type = type;
    // if (m_type == GLOBAL_ACC_R)
    //   throw std::runtime_error("global acc r");
    m_addr = address;
    m_req_size = size;
    m_write = wr;
  }
  mem_access_t(mem_access_type type, new_addr_type address, unsigned size,
               bool wr, const active_mask_t &active_mask,
               const mem_access_byte_mask_t &byte_mask,
               const mem_access_sector_mask_t &sector_mask, gpgpu_context *ctx)
      : m_warp_mask(active_mask),
        m_byte_mask(byte_mask),
        m_sector_mask(sector_mask) {
    init(ctx);
    m_type = type;
    // if (m_type == GLOBAL_ACC_R)
    //   throw std::runtime_error("global acc r");
    m_addr = address;
    m_req_size = size;
    m_write = wr;
  }

  void set_addr(new_addr_type addr) { m_addr = addr; }
  new_addr_type get_addr() const { return m_addr; }

  new_addr_type get_byte() const {
    for (new_addr_type byte = 0; byte < MAX_MEMORY_ACCESS_SIZE; byte++) {
      if (m_byte_mask[byte]) {
        return byte;
      }
    }
    // this should be unreachable
    return 0;
  }

  new_addr_type get_byte_addr() const {
    return get_addr() + (get_byte() % SECTOR_SIZE);
  }

  new_addr_type get_relative_byte_addr() const {
    return get_relative_addr() + (get_byte() % SECTOR_SIZE);
  }

  new_addr_type get_relative_addr() const {
    assert(m_addr >= m_allocation_start_addr);
    return m_addr - m_allocation_start_addr;
  }

  new_addr_type get_alloc_start_addr() const { return m_allocation_start_addr; }
  void set_alloc_start_addr(new_addr_type addr) {
    m_allocation_start_addr = addr;
  }
  unsigned get_alloc_id() const { return m_allocation_id; }
  void set_alloc_id(unsigned id) { m_allocation_id = id; }

  unsigned get_size() const { return m_req_size; }
  const active_mask_t &get_warp_mask() const { return m_warp_mask; }
  bool is_write() const { return m_write; }
  enum mem_access_type get_type() const { return m_type; }
  const char *get_type_str() const { return get_mem_access_type_str(m_type); }
  mem_access_byte_mask_t get_byte_mask() const { return m_byte_mask; }
  mem_access_sector_mask_t get_sector_mask() const { return m_sector_mask; }

  // void print(FILE *fp) const {
  //   fprintf(fp, "addr=0x%ld, %s, size=%u, ", m_addr,
  //           m_write ? "store" : "load ", m_req_size);
  //   switch (m_type) {
  //     case GLOBAL_ACC_R:
  //       fprintf(fp, "GLOBAL_R");
  //       break;
  //     case LOCAL_ACC_R:
  //       fprintf(fp, "LOCAL_R ");
  //       break;
  //     case CONST_ACC_R:
  //       fprintf(fp, "CONST   ");
  //       break;
  //     case TEXTURE_ACC_R:
  //       fprintf(fp, "TEXTURE ");
  //       break;
  //     case GLOBAL_ACC_W:
  //       fprintf(fp, "GLOBAL_W");
  //       break;
  //     case LOCAL_ACC_W:
  //       fprintf(fp, "LOCAL_W ");
  //       break;
  //     case L2_WRBK_ACC:
  //       fprintf(fp, "L2_WRBK ");
  //       break;
  //     case INST_ACC_R:
  //       fprintf(fp, "INST    ");
  //       break;
  //     case L1_WRBK_ACC:
  //       fprintf(fp, "L1_WRBK ");
  //       break;
  //     default:
  //       fprintf(fp, "unknown ");
  //       break;
  //   }
  // }

  // gpgpu_context *gpgpu_ctx;

 private:
  void init(gpgpu_context *ctx);

  unsigned m_uid;
  new_addr_type m_addr;                   // request address
  new_addr_type m_allocation_start_addr;  // allocation start address
  unsigned m_allocation_id;               // allocation id
  bool m_write;
  unsigned m_req_size;  // bytes
  mem_access_type m_type;
  active_mask_t m_warp_mask;
  mem_access_byte_mask_t m_byte_mask;
  mem_access_sector_mask_t m_sector_mask;
};

std::ostream &operator<<(std::ostream &os, const mem_access_t &access);

#include "fmt/core.h"

template <>
struct fmt::formatter<mem_access_t> {
  constexpr auto parse(format_parse_context &ctx)
      -> format_parse_context::iterator {
    return ctx.end();
  }

  auto format(const mem_access_t &access, format_context &ctx) const
      -> format_context::iterator {
    fmt::format_to(ctx.out(), "{}", access.get_type_str());
    new_addr_type addr = access.get_addr();
    new_addr_type rel_addr = access.get_relative_addr();
    if (addr == rel_addr) {
      return fmt::format_to(ctx.out(), "@{}", addr);
    } else {
      return fmt::format_to(ctx.out(), "@{}+{}", access.get_alloc_id(),
                            rel_addr);
    }
  }
};
