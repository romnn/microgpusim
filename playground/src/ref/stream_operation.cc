#include "stream_operation.hpp"

#include "cu_stream.hpp"
#include "gpgpu_sim.hpp"

bool stream_operation::do_operation(gpgpu_sim *gpu) {
  if (is_noop())
    return true;

  assert(!m_done && m_stream);
  if (g_debug_execution >= 3)
    printf("GPGPU-Sim API: stream %u performing ", m_stream->get_uid());
  switch (m_type) {
  case stream_memcpy_host_to_device:
    if (g_debug_execution >= 3)
      printf("memcpy host-to-device\n");
    gpu->memcpy_to_gpu(m_device_address_dst, m_host_address_src, m_cnt);
    m_stream->record_next_done();
    break;
  case stream_memcpy_device_to_host:
    if (g_debug_execution >= 3)
      printf("memcpy device-to-host\n");
    gpu->memcpy_from_gpu(m_host_address_dst, m_device_address_src, m_cnt);
    m_stream->record_next_done();
    break;
  case stream_memcpy_device_to_device:
    if (g_debug_execution >= 3)
      printf("memcpy device-to-device\n");
    gpu->memcpy_gpu_to_gpu(m_device_address_dst, m_device_address_src, m_cnt);
    m_stream->record_next_done();
    break;
  case stream_memcpy_to_symbol:
    if (g_debug_execution >= 3)
      printf("memcpy to symbol\n");
    gpu->gpgpu_ctx->func_sim->gpgpu_ptx_sim_memcpy_symbol(
        m_symbol, m_host_address_src, m_cnt, m_offset, 1, gpu);
    m_stream->record_next_done();
    break;
  case stream_memcpy_from_symbol:
    if (g_debug_execution >= 3)
      printf("memcpy from symbol\n");
    gpu->gpgpu_ctx->func_sim->gpgpu_ptx_sim_memcpy_symbol(
        m_symbol, m_host_address_dst, m_cnt, m_offset, 0, gpu);
    m_stream->record_next_done();
    break;
  case stream_kernel_launch:
    if (m_sim_mode) { // Functional Sim
      if (g_debug_execution >= 3) {
        printf("kernel %d: \'%s\' transfer to GPU hardware scheduler\n",
               m_kernel->get_uid(), m_kernel->name().c_str());
        m_kernel->print_parent_info();
      }
      gpu->set_cache_config(m_kernel->name());
      gpu->functional_launch(m_kernel);
    } else { // Performance Sim
      if (gpu->can_start_kernel() && m_kernel->m_launch_latency == 0) {
        if (g_debug_execution >= 3) {
          printf("kernel %d: \'%s\' transfer to GPU hardware scheduler\n",
                 m_kernel->get_uid(), m_kernel->name().c_str());
          m_kernel->print_parent_info();
        }
        gpu->set_cache_config(m_kernel->name());
        gpu->launch(m_kernel);
      } else {
        if (m_kernel->m_launch_latency)
          m_kernel->m_launch_latency--;
        if (g_debug_execution >= 3)
          printf("kernel %d: \'%s\', latency %u not ready to transfer to GPU "
                 "hardware scheduler\n",
                 m_kernel->get_uid(), m_kernel->name().c_str(),
                 m_kernel->m_launch_latency);
        return false;
      }
    }
    break;
  case stream_event: {
    printf("event update\n");
    time_t wallclock = time((time_t *)NULL);
    m_event->update(gpu->gpu_tot_sim_cycle, wallclock);
    m_stream->record_next_done();
  } break;
  case stream_wait_event:
    // only allows next op to go if event is done
    // otherwise stays in the stream queue
    printf("stream wait event processing...\n");
    if (m_event->num_updates() >= m_cnt) {
      printf("stream wait event done\n");
      m_stream->record_next_done();
    } else {
      return false;
    }
    break;
  default:
    abort();
  }
  m_done = true;
  fflush(stdout);
  return true;
}

void stream_operation::print(FILE *fp) const {
  fprintf(fp, " stream operation ");
  switch (m_type) {
  case stream_event:
    fprintf(fp, "event");
    break;
  case stream_kernel_launch:
    fprintf(fp, "kernel");
    break;
  case stream_memcpy_device_to_device:
    fprintf(fp, "memcpy device-to-device");
    break;
  case stream_memcpy_device_to_host:
    fprintf(fp, "memcpy device-to-host");
    break;
  case stream_memcpy_host_to_device:
    fprintf(fp, "memcpy host-to-device");
    break;
  case stream_memcpy_to_symbol:
    fprintf(fp, "memcpy to symbol");
    break;
  case stream_memcpy_from_symbol:
    fprintf(fp, "memcpy from symbol");
    break;
  case stream_no_op:
    fprintf(fp, "no-op");
    break;
  }
}
