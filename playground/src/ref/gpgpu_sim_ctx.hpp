#pragma once

#include <cstddef>
#include <pthread.h>
#include <semaphore.h>
#include <time.h>

class gpgpu_context;

class GPGPUsim_ctx {
public:
  GPGPUsim_ctx(gpgpu_context *ctx) {
    g_sim_active = false;
    g_sim_done = true;
    break_limit = false;
    g_sim_lock = PTHREAD_MUTEX_INITIALIZER;

    g_the_gpu_config = NULL;
    g_the_gpu = NULL;
    g_stream_manager = NULL;
    the_cude_device = NULL;
    the_context = NULL;
    gpgpu_ctx = ctx;
  }

  // struct gpgpu_ptx_sim_arg *grid_params;

  sem_t g_sim_signal_start;
  sem_t g_sim_signal_finish;
  sem_t g_sim_signal_exit;
  time_t g_simulation_starttime;
  pthread_t g_simulation_thread;

  class gpgpu_sim_config *g_the_gpu_config;
  class trace_gpgpu_sim *g_the_gpu;
  class stream_manager *g_stream_manager;

  struct _cuda_device_id *the_cude_device;
  struct CUctx_st *the_context;
  gpgpu_context *gpgpu_ctx;

  pthread_mutex_t g_sim_lock;
  bool g_sim_active;
  bool g_sim_done;
  bool break_limit;
};
