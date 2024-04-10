#include "lib.hpp"

#include "gpgpu_context.hpp"
#include "gpgpu_sim.hpp"
#include "gpgpu_sim_config.hpp"
#include "symbol_table.hpp"
#include <cstdio>

int load_ptx_from_filename(const char *file_name) {
  gpgpu_context ctx = gpgpu_context();
  gpgpu_sim_config config = gpgpu_sim_config(&ctx);
  // config.m_shader_config.warp_size = 32;
  // config.m_shader_config.n_simt_clusters = 28;
  // config.m_shader_config.n_simt_cores_per_cluster = 1;
  // config.m_shader_config.gpgpu_shmem_size = 1;

  // config.m_shader_config.m_L1I_config.init("test",
  //                                          FuncCache::FuncCachePreferL1);
  // unsigned n_thread_per_shader;
  // unsigned warp_size;
  // unsigned max_cta_per_core;
  // unsigned n_simt_cores_per_cluster;
  // unsigned n_simt_clusters;
  // unsigned gpgpu_shader_registers;

  // config.init();
  // config.m_shader_config.m_L1I_config.init(char *config, FuncCache status);
  // void init(char *config, FuncCache status) {

  printf("config: num_shader=%d\n", config.num_shader());

  gpgpu_sim sim = gpgpu_sim(config, &ctx);

  printf("parsing %s ...\n", file_name);
  symbol_table *table =
      sim.gpgpu_ctx->gpgpu_ptx_sim_load_ptx_from_filename(file_name);
  return 0;
}
