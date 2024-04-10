#include "ptxinfo_data.hpp"

#include "cu_ctx.hpp"
#include "gpgpu_context.hpp"
#include "gpgpusim_ctx.hpp"

// gpgpu_context *GPGPU_Context() {
//   static gpgpu_context *gpgpu_ctx = NULL;
//   if (gpgpu_ctx == NULL) {
//     gpgpu_ctx = new gpgpu_context();
//   }
//   return gpgpu_ctx;
// }

CUctx_st *GPGPUSim_Context(gpgpu_context *ctx) {
  // static CUctx_st *the_context = NULL;
  CUctx_st *the_context = ctx->the_gpgpusim->the_context;
  if (the_context == NULL) {
    _cuda_device_id *the_gpu = ctx->GPGPUSim_Init();
    ctx->the_gpgpusim->the_context = new CUctx_st(the_gpu);
    the_context = ctx->the_gpgpusim->the_context;
  }
  return the_context;
}

static char *g_ptxinfo_kname = NULL;
static struct gpgpu_ptx_sim_info g_ptxinfo;
static std::map<unsigned, const char *> g_duplicate;
static const char *g_last_dup_type;

const char *get_ptxinfo_kname() { return g_ptxinfo_kname; }

void print_ptxinfo() {
  if (!get_ptxinfo_kname()) {
    printf("GPGPU-Sim PTX: Binary info : gmem=%u, cmem=%u\n", g_ptxinfo.gmem,
           g_ptxinfo.cmem);
  }
  if (get_ptxinfo_kname()) {
    printf(
        "GPGPU-Sim PTX: Kernel \'%s\' : regs=%u, lmem=%u, smem=%u, cmem=%u\n",
        get_ptxinfo_kname(), g_ptxinfo.regs, g_ptxinfo.lmem, g_ptxinfo.smem,
        g_ptxinfo.cmem);
  }
}

struct gpgpu_ptx_sim_info get_ptxinfo() { return g_ptxinfo; }

void clear_ptxinfo() {
  free(g_ptxinfo_kname);
  g_ptxinfo_kname = NULL;
  g_ptxinfo.regs = 0;
  g_ptxinfo.lmem = 0;
  g_ptxinfo.smem = 0;
  g_ptxinfo.cmem = 0;
  g_ptxinfo.gmem = 0;
  g_ptxinfo.ptx_version = 0;
  g_ptxinfo.sm_target = 0;
}

void ptxinfo_data::ptxinfo_addinfo() {
  CUctx_st *context = GPGPUSim_Context(gpgpu_ctx);
  if (!get_ptxinfo_kname()) {
    /* This info is not per kernel (since CUDA 5.0 some info (e.g. gmem, and
     * cmem) is added at the beginning for the whole binary ) */
    print_ptxinfo();
    context->add_ptxinfo(get_ptxinfo());
    clear_ptxinfo();
    return;
  }
  if (!strcmp("__cuda_dummy_entry__", get_ptxinfo_kname())) {
    // this string produced by ptxas for empty ptx files (e.g., bandwidth test)
    clear_ptxinfo();
    return;
  }
  print_ptxinfo();
  context->add_ptxinfo(get_ptxinfo_kname(), get_ptxinfo());
  clear_ptxinfo();
}
