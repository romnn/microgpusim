#include "memory_space.hpp"

#include "gpgpu_context.hpp"
#include "gpgpu_sim.hpp"
#include "gpgpusim_ctx.hpp"
#include "ptx_thread_info.hpp"

template <unsigned BSIZE>
memory_space_impl<BSIZE>::memory_space_impl(std::string name,
                                            unsigned hash_size) {
  m_name = name;
  // MEM_MAP_RESIZE(hash_size);

  m_log2_block_size = -1;
  for (unsigned n = 0, mask = 1; mask != 0; mask <<= 1, n++) {
    if (BSIZE & mask) {
      assert(m_log2_block_size == (unsigned)-1);
      m_log2_block_size = n;
    }
  }
  assert(m_log2_block_size != (unsigned)-1);
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::write_only(mem_addr_t offset, mem_addr_t index,
                                          size_t length, const void *data) {
  m_data[index].write(offset, length, (const unsigned char *)data);
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::write(mem_addr_t addr, size_t length,
                                     const void *data,
                                     class ptx_thread_info *thd,
                                     const ptx_instruction *pI) {
  mem_addr_t index = addr >> m_log2_block_size;

  if ((addr + length) <= (index + 1) * BSIZE) {
    // fast route for intra-block access
    unsigned offset = addr & (BSIZE - 1);
    unsigned nbytes = length;
    m_data[index].write(offset, nbytes, (const unsigned char *)data);
  } else {
    // slow route for inter-block access
    unsigned nbytes_remain = length;
    unsigned src_offset = 0;
    mem_addr_t current_addr = addr;

    while (nbytes_remain > 0) {
      unsigned offset = current_addr & (BSIZE - 1);
      mem_addr_t page = current_addr >> m_log2_block_size;
      mem_addr_t access_limit = offset + nbytes_remain;
      if (access_limit > BSIZE) {
        access_limit = BSIZE;
      }

      size_t tx_bytes = access_limit - offset;
      m_data[page].write(offset, tx_bytes,
                         &((const unsigned char *)data)[src_offset]);

      // advance pointers
      src_offset += tx_bytes;
      current_addr += tx_bytes;
      nbytes_remain -= tx_bytes;
    }
    assert(nbytes_remain == 0);
  }
  if (!m_watchpoints.empty()) {
    std::map<unsigned, mem_addr_t>::iterator i;
    for (i = m_watchpoints.begin(); i != m_watchpoints.end(); i++) {
      mem_addr_t wa = i->second;
      if (((addr <= wa) && ((addr + length) > wa)) ||
          ((addr > wa) && (addr < (wa + 4))))
        thd->get_gpu()->gpgpu_ctx->the_gpgpusim->g_the_gpu->hit_watchpoint(
            i->first, thd, pI);
    }
  }
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::read_single_block(mem_addr_t blk_idx,
                                                 mem_addr_t addr, size_t length,
                                                 void *data) const {
  if ((addr + length) > (blk_idx + 1) * BSIZE) {
    printf("GPGPU-Sim PTX: ERROR * access to memory \'%s\' is unaligned : "
           "addr=0x%llx, length=%zu\n",
           m_name.c_str(), addr, length);
    printf("GPGPU-Sim PTX: (addr+length)=0x%llx > 0x%llx=(index+1)*BSIZE, "
           "index=0x%llx, BSIZE=0x%x\n",
           (addr + length), (blk_idx + 1) * BSIZE, blk_idx, BSIZE);
    throw 1;
  }
  typename map_t::const_iterator i = m_data.find(blk_idx);
  if (i == m_data.end()) {
    for (size_t n = 0; n < length; n++)
      ((unsigned char *)data)[n] = (unsigned char)0;
    // printf("GPGPU-Sim PTX:  WARNING reading %zu bytes from unititialized
    // memory at address 0x%x in space %s\n", length, addr, m_name.c_str() );
  } else {
    unsigned offset = addr & (BSIZE - 1);
    unsigned nbytes = length;
    i->second.read(offset, nbytes, (unsigned char *)data);
  }
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::read(mem_addr_t addr, size_t length,
                                    void *data) const {
  mem_addr_t index = addr >> m_log2_block_size;
  if ((addr + length) <= (index + 1) * BSIZE) {
    // fast route for intra-block access
    read_single_block(index, addr, length, data);
  } else {
    // slow route for inter-block access
    unsigned nbytes_remain = length;
    unsigned dst_offset = 0;
    mem_addr_t current_addr = addr;

    while (nbytes_remain > 0) {
      unsigned offset = current_addr & (BSIZE - 1);
      mem_addr_t page = current_addr >> m_log2_block_size;
      mem_addr_t access_limit = offset + nbytes_remain;
      if (access_limit > BSIZE) {
        access_limit = BSIZE;
      }

      size_t tx_bytes = access_limit - offset;
      read_single_block(page, current_addr, tx_bytes,
                        &((unsigned char *)data)[dst_offset]);

      // advance pointers
      dst_offset += tx_bytes;
      current_addr += tx_bytes;
      nbytes_remain -= tx_bytes;
    }
    assert(nbytes_remain == 0);
  }
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::print(const char *format, FILE *fout) const {
  typename map_t::const_iterator i_page;

  for (i_page = m_data.begin(); i_page != m_data.end(); ++i_page) {
    fprintf(fout, "%s %08llx:", m_name.c_str(), i_page->first);
    i_page->second.print(format, fout);
  }
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::set_watch(addr_t addr, unsigned watchpoint) {
  m_watchpoints[watchpoint] = addr;
}

template class memory_space_impl<32>;
template class memory_space_impl<64>;
template class memory_space_impl<8192>;
template class memory_space_impl<16 * 1024>;
