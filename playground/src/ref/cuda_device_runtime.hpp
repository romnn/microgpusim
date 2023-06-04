#pragma once

#include <list>
#include <map>

class cuda_device_runtime {
public:
  // cuda_device_runtime(gpgpu_context *ctx) {
  cuda_device_runtime() {
    // g_total_param_size = 0;
    g_max_total_param_size = 0;
    // gpgpu_ctx = ctx;
  }
  // unsigned long long g_total_param_size;
  // std::map<void *, device_launch_config_t> g_cuda_device_launch_param_map;
  // std::list<device_launch_operation_t> g_cuda_device_launch_op;
  unsigned g_kernel_launch_latency;
  unsigned g_TB_launch_latency;
  unsigned long long g_max_total_param_size;
  bool g_cdp_enabled;
  //
  //   // backward pointer
  //   class gpgpu_context *gpgpu_ctx;
  // #if (CUDART_VERSION >= 5000)
  // #pragma once
  //   void gpgpusim_cuda_launchDeviceV2(const ptx_instruction *pI,
  //                                     ptx_thread_info *thread,
  //                                     const function_info *target_func);
  //   void gpgpusim_cuda_streamCreateWithFlags(const ptx_instruction *pI,
  //                                            ptx_thread_info *thread,
  //                                            const function_info
  //                                            *target_func);
  //   void gpgpusim_cuda_getParameterBufferV2(const ptx_instruction *pI,
  //                                           ptx_thread_info *thread,
  //                                           const function_info
  //                                           *target_func);
  //   void launch_all_device_kernels();
  //   void launch_one_device_kernel();
  // #endif
};
