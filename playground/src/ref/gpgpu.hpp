#pragma once

#include <map>
#include <set>

#include "gpgpu_context.hpp"
#include "gpgpu_functional_sim_config.hpp"

class gpgpu_t {
public:
  gpgpu_t(const gpgpu_functional_sim_config &config, gpgpu_context *ctx);
  // backward pointer
  class gpgpu_context *gpgpu_ctx;
  int checkpoint_option;
  int checkpoint_kernel;
  int checkpoint_CTA;
  unsigned resume_option;
  unsigned resume_kernel;
  unsigned resume_CTA;
  unsigned checkpoint_CTA_t;
  int checkpoint_insn_Y;

  // Move some cycle core stats here instead of being global
  unsigned long long gpu_sim_cycle;
  unsigned long long gpu_tot_sim_cycle;

  void *gpu_malloc(size_t size);
  void *gpu_mallocarray(size_t count);
  void gpu_memset(size_t dst_start_addr, int c, size_t count);
  void memcpy_to_gpu(size_t dst_start_addr, const void *src, size_t count);
  void memcpy_from_gpu(void *dst, size_t src_start_addr, size_t count);
  void memcpy_gpu_to_gpu(size_t dst, size_t src, size_t count);

  class memory_space *get_global_memory() {
    return m_global_mem;
  }
  class memory_space *get_tex_memory() {
    return m_tex_mem;
  }
  class memory_space *get_surf_memory() {
    return m_surf_mem;
  }

  void gpgpu_ptx_sim_bindTextureToArray(const struct textureReference *texref,
                                        const struct cudaArray *array);
  void gpgpu_ptx_sim_bindNameToTexture(const char *name,
                                       const struct textureReference *texref,
                                       int dim, int readmode, int ext);
  void gpgpu_ptx_sim_unbindTexture(const struct textureReference *texref);
  const char *
  gpgpu_ptx_sim_findNamefromTexture(const struct textureReference *texref);

  const struct textureReference *get_texref(const std::string &texname) const {
    std::map<std::string,
             std::set<const struct textureReference *>>::const_iterator t =
        m_NameToTextureRef.find(texname);
    assert(t != m_NameToTextureRef.end());
    return *(t->second.begin());
  }

  const struct cudaArray *get_texarray(const std::string &texname) const {
    std::map<std::string, const struct cudaArray *>::const_iterator t =
        m_NameToCudaArray.find(texname);
    assert(t != m_NameToCudaArray.end());
    return t->second;
  }

  const struct textureInfo *get_texinfo(const std::string &texname) const {
    std::map<std::string, const struct textureInfo *>::const_iterator t =
        m_NameToTextureInfo.find(texname);
    assert(t != m_NameToTextureInfo.end());
    return t->second;
  }

  const struct textureReferenceAttr *
  get_texattr(const std::string &texname) const {
    std::map<std::string, const struct textureReferenceAttr *>::const_iterator
        t = m_NameToAttribute.find(texname);
    assert(t != m_NameToAttribute.end());
    return t->second;
  }

  const gpgpu_functional_sim_config &get_config() const {
    return m_function_model_config;
  }
  FILE *get_ptx_inst_debug_file() { return ptx_inst_debug_file; }
  // FILE *get_mem_debug_file() { return mem_debug_file; }

  //  These maps return the current texture mappings for the GPU at any given
  //  time.
  std::map<std::string, const struct cudaArray *> getNameArrayMapping() {
    return m_NameToCudaArray;
  }
  std::map<std::string, const struct textureInfo *> getNameInfoMapping() {
    return m_NameToTextureInfo;
  }

  virtual ~gpgpu_t() {}

protected:
  const gpgpu_functional_sim_config &m_function_model_config;
  FILE *ptx_inst_debug_file;
  // FILE *mem_debug_file;

  class memory_space *m_global_mem;
  class memory_space *m_tex_mem;
  class memory_space *m_surf_mem;

  unsigned long long m_dev_malloc;
  //  These maps contain the current texture mappings for the GPU at any given
  //  time.
  std::map<std::string, std::set<const struct textureReference *>>
      m_NameToTextureRef;
  std::map<const struct textureReference *, std::string> m_TextureRefToName;
  std::map<std::string, const struct cudaArray *> m_NameToCudaArray;
  std::map<std::string, const struct textureInfo *> m_NameToTextureInfo;
  std::map<std::string, const struct textureReferenceAttr *> m_NameToAttribute;
};
