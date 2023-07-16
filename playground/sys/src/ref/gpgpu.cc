#include "gpgpu.hpp"

// #include "gpgpu_functional_sim_config.hpp"
#include "memory_space_impl.hpp"

gpgpu_t::gpgpu_t(const gpgpu_functional_sim_config &config, gpgpu_context *ctx)
    : m_function_model_config(config) {
  gpgpu_ctx = ctx;
  m_global_mem = new memory_space_impl<8192>("global", 64 * 1024);

  m_tex_mem = new memory_space_impl<8192>("tex", 64 * 1024);
  m_surf_mem = new memory_space_impl<8192>("surf", 64 * 1024);

  m_dev_malloc = GLOBAL_HEAP_START;
  checkpoint_option = m_function_model_config.get_checkpoint_option();
  checkpoint_kernel = m_function_model_config.get_checkpoint_kernel();
  checkpoint_CTA = m_function_model_config.get_checkpoint_CTA();
  resume_option = m_function_model_config.get_resume_option();
  resume_kernel = m_function_model_config.get_resume_kernel();
  resume_CTA = m_function_model_config.get_resume_CTA();
  checkpoint_CTA_t = m_function_model_config.get_checkpoint_CTA_t();
  checkpoint_insn_Y = m_function_model_config.get_checkpoint_insn_Y();

  // initialize texture mappings to empty
  m_NameToTextureInfo.clear();
  m_NameToCudaArray.clear();
  m_TextureRefToName.clear();
  m_NameToAttribute.clear();

  if (m_function_model_config.get_ptx_inst_debug_to_file() != 0)
    ptx_inst_debug_file =
        fopen(m_function_model_config.get_ptx_inst_debug_file(), "w");

  // mem_debug_file = fopen(m_function_model_config.get_mem_debug_file(), "w");

  gpu_sim_cycle = 0;
  gpu_tot_sim_cycle = 0;
}

void gpgpu_t::gpgpu_ptx_sim_bindNameToTexture(
    const char *name, const struct textureReference *texref, int dim,
    int readmode, int ext) {
  std::string texname(name);
  if (m_NameToTextureRef.find(texname) == m_NameToTextureRef.end()) {
    m_NameToTextureRef[texname] = std::set<const struct textureReference *>();
  } else {
    const struct textureReference *tr = *m_NameToTextureRef[texname].begin();
    assert(tr != NULL);
    // asserts that all texrefs in set have same fields
    assert(tr->normalized == texref->normalized &&
           tr->filterMode == texref->filterMode &&
           tr->addressMode[0] == texref->addressMode[0] &&
           tr->addressMode[1] == texref->addressMode[1] &&
           tr->addressMode[2] == texref->addressMode[2] &&
           tr->channelDesc.x == texref->channelDesc.x &&
           tr->channelDesc.y == texref->channelDesc.y &&
           tr->channelDesc.z == texref->channelDesc.z &&
           tr->channelDesc.w == texref->channelDesc.w &&
           tr->channelDesc.f == texref->channelDesc.f);
  }
  m_NameToTextureRef[texname].insert(texref);
  m_TextureRefToName[texref] = texname;
  const textureReferenceAttr *texAttr = new textureReferenceAttr(
      texref, dim, (enum cudaTextureReadMode)readmode, ext);
  m_NameToAttribute[texname] = texAttr;
}

const char *gpgpu_t::gpgpu_ptx_sim_findNamefromTexture(
    const struct textureReference *texref) {
  std::map<const struct textureReference *, std::string>::const_iterator t =
      m_TextureRefToName.find(texref);
  assert(t != m_TextureRefToName.end());
  return t->second.c_str();
}

void gpgpu_t::gpgpu_ptx_sim_bindTextureToArray(
    const struct textureReference *texref, const struct cudaArray *array) {
  std::string texname = gpgpu_ptx_sim_findNamefromTexture(texref);

  std::map<std::string, const struct cudaArray *>::const_iterator t =
      m_NameToCudaArray.find(texname);
  // check that there's nothing there first
  if (t != m_NameToCudaArray.end()) {
    printf(
        "GPGPU-Sim PTX:   Warning: binding to texref associated with %s, which "
        "was previously bound.\nImplicitly unbinding texref associated to %s "
        "first\n",
        texname.c_str(), texname.c_str());
  }
  m_NameToCudaArray[texname] = array;
  unsigned int texel_size_bits =
      array->desc.w + array->desc.x + array->desc.y + array->desc.z;
  unsigned int texel_size = texel_size_bits / 8;
  unsigned int Tx, Ty;
  int r;

  printf("GPGPU-Sim PTX:   texel size = %d\n", texel_size);
  printf("GPGPU-Sim PTX:   texture cache linesize = %d\n",
         m_function_model_config.get_texcache_linesize());
  // first determine base Tx size for given linesize
  switch (m_function_model_config.get_texcache_linesize()) {
    case 16:
      Tx = 4;
      break;
    case 32:
      Tx = 8;
      break;
    case 64:
      Tx = 8;
      break;
    case 128:
      Tx = 16;
      break;
    case 256:
      Tx = 16;
      break;
    default:
      printf(
          "GPGPU-Sim PTX:   Line size of %d bytes currently not supported.\n",
          m_function_model_config.get_texcache_linesize());
      assert(0);
      break;
  }
  r = texel_size >> 2;
  // modify base Tx size to take into account size of each texel in bytes
  while (r != 0) {
    Tx = Tx >> 1;
    r = r >> 2;
  }
  // by now, got the correct Tx size, calculate correct Ty size
  Ty = m_function_model_config.get_texcache_linesize() / (Tx * texel_size);

  printf(
      "GPGPU-Sim PTX:   Tx = %d; Ty = %d, Tx_numbits = %d, Ty_numbits = %d\n",
      Tx, Ty, intLOGB2(Tx), intLOGB2(Ty));
  printf("GPGPU-Sim PTX:   Texel size = %d bytes; texel_size_numbits = %d\n",
         texel_size, intLOGB2(texel_size));
  printf(
      "GPGPU-Sim PTX:   Binding texture to array starting at devPtr32 = 0x%x\n",
      array->devPtr32);
  printf("GPGPU-Sim PTX:   Texel size = %d bytes\n", texel_size);
  struct textureInfo *texInfo =
      (struct textureInfo *)malloc(sizeof(struct textureInfo));
  texInfo->Tx = Tx;
  texInfo->Ty = Ty;
  texInfo->Tx_numbits = intLOGB2(Tx);
  texInfo->Ty_numbits = intLOGB2(Ty);
  texInfo->texel_size = texel_size;
  texInfo->texel_size_numbits = intLOGB2(texel_size);
  m_NameToTextureInfo[texname] = texInfo;
}

void gpgpu_t::gpgpu_ptx_sim_unbindTexture(
    const struct textureReference *texref) {
  // assumes bind-use-unbind-bind-use-unbind pattern
  std::string texname = gpgpu_ptx_sim_findNamefromTexture(texref);
  m_NameToCudaArray.erase(texname);
  m_NameToTextureInfo.erase(texname);
}
