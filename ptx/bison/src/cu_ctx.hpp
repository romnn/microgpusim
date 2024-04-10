#pragma once

#include <cassert>
#include <cstddef>

#include "function_info.hpp"
#include "symbol.hpp"
#include "symbol_table.hpp"

class _cuda_device_id;

struct CUctx_st {
  CUctx_st(_cuda_device_id *gpu) {
    m_gpu = gpu;
    m_binary_info.cmem = 0;
    m_binary_info.gmem = 0;
    no_of_ptx = 0;
  }

  _cuda_device_id *get_device() { return m_gpu; }

  void add_binary(symbol_table *symtab, unsigned fat_cubin_handle) {
    m_code[fat_cubin_handle] = symtab;
    m_last_fat_cubin_handle = fat_cubin_handle;
  }

  void add_ptxinfo(const char *deviceFun,
                   const struct gpgpu_ptx_sim_info &info) {
    symbol *s = m_code[m_last_fat_cubin_handle]->lookup(deviceFun);
    assert(s != NULL);
    function_info *f = s->get_pc();
    assert(f != NULL);
    f->set_kernel_info(info);
  }

  void add_ptxinfo(const struct gpgpu_ptx_sim_info &info) {
    m_binary_info = info;
  }

  void register_function(unsigned fat_cubin_handle, const char *hostFun,
                         const char *deviceFun) {
    if (m_code.find(fat_cubin_handle) != m_code.end()) {
      symbol *s = m_code[fat_cubin_handle]->lookup(deviceFun);
      if (s != NULL) {
        function_info *f = s->get_pc();
        assert(f != NULL);
        m_kernel_lookup[hostFun] = f;
      } else {
        printf("Warning: cannot find deviceFun %s\n", deviceFun);
        m_kernel_lookup[hostFun] = NULL;
      }
      //		assert( s != NULL );
      //		function_info *f = s->get_pc();
      //		assert( f != NULL );
      //		m_kernel_lookup[hostFun] = f;
    } else {
      m_kernel_lookup[hostFun] = NULL;
    }
  }

  void register_hostFun_function(const char *hostFun, function_info *f) {
    m_kernel_lookup[hostFun] = f;
  }

  function_info *get_kernel(const char *hostFun) {
    std::map<const void *, function_info *>::iterator i =
        m_kernel_lookup.find(hostFun);
    assert(i != m_kernel_lookup.end());
    return i->second;
  }

  int no_of_ptx;

private:
  _cuda_device_id *m_gpu; // selected gpu
  std::map<unsigned, symbol_table *>
      m_code; // fat binary handle => global symbol table
  unsigned m_last_fat_cubin_handle;
  std::map<const void *, function_info *>
      m_kernel_lookup; // unique id (CUDA app function address) => kernel entry
                       // point
  struct gpgpu_ptx_sim_info m_binary_info;
};
