#include "trace_kernel_info.hpp"

#include "gpgpu_context.hpp"
#include "isa_def.hpp"
#include "memory_space_impl.hpp"
#include "trace_parser.hpp"

trace_kernel_info_t::trace_kernel_info_t(dim3 gridDim, dim3 blockDim,
                                         trace_function_info *m_function_info,
                                         trace_parser *parser,
                                         class trace_config *config,
                                         kernel_trace_t *kernel_trace_info) {
  // : kernel_info_t(gridDim, blockDim, m_function_info) {
  m_kernel_entry = m_function_info;
  m_grid_dim = gridDim;
  m_block_dim = blockDim;
  m_next_cta.x = 0;
  m_next_cta.y = 0;
  m_next_cta.z = 0;
  m_next_tid = m_next_cta;
  m_num_cores_running = 0;
  m_uid = (m_function_info->gpgpu_ctx->kernel_info_m_next_uid)++;
  // TODO: required?
  // m_param_mem = new memory_space_impl<8192>("param", 64 * 1024);

  // Jin: parent and child kernel management for CDP
  // m_parent_kernel = NULL;

  // Jin: launch latency management
  m_launch_latency =
      m_function_info->gpgpu_ctx->device_runtime->g_kernel_launch_latency;

  m_kernel_TB_latency =
      m_function_info->gpgpu_ctx->device_runtime->g_kernel_launch_latency +
      num_blocks() *
          m_function_info->gpgpu_ctx->device_runtime->g_TB_launch_latency;

  cache_config_set = false;

  m_parser = parser;
  m_tconfig = config;
  m_kernel_trace_info = kernel_trace_info;
  m_was_launched = false;

  // resolve the binary version
  if (kernel_trace_info->binary_verion == AMPERE_RTX_BINART_VERSION ||
      kernel_trace_info->binary_verion == AMPERE_A100_BINART_VERSION)
    OpcodeMap = &Ampere_OpcodeMap;
  else if (kernel_trace_info->binary_verion == VOLTA_BINART_VERSION)
    OpcodeMap = &Volta_OpcodeMap;
  else if (kernel_trace_info->binary_verion == PASCAL_TITANX_BINART_VERSION ||
           kernel_trace_info->binary_verion == PASCAL_P100_BINART_VERSION)
    OpcodeMap = &Pascal_OpcodeMap;
  else if (kernel_trace_info->binary_verion == KEPLER_BINART_VERSION)
    OpcodeMap = &Kepler_OpcodeMap;
  else if (kernel_trace_info->binary_verion == TURING_BINART_VERSION)
    OpcodeMap = &Turing_OpcodeMap;
  else {
    printf("unsupported binary version: %d\n",
           kernel_trace_info->binary_verion);
    fflush(stdout);
    exit(0);
  }
}

void trace_kernel_info_t::get_next_threadblock_traces(
    std::vector<std::vector<inst_trace_t> *> threadblock_traces) {
  m_parser->get_next_threadblock_traces(
      threadblock_traces, m_kernel_trace_info->trace_verion,
      m_kernel_trace_info->enable_lineinfo, m_kernel_trace_info->ifs);
}

std::string trace_kernel_info_t::name() const {
  return m_kernel_entry->get_name();
}

bool trace_kernel_info_t::is_finished() {
  if (done() && children_all_finished())
    return true;
  else
    return false;
}

bool trace_kernel_info_t::children_all_finished() {
  if (!m_child_kernels.empty())
    return false;

  return true;
}

void trace_kernel_info_t::notify_parent_finished() {
  throw std::runtime_error("notify_parent_finished needs device runtime");
  // if (m_parent_kernel) {
  //   m_kernel_entry->gpgpu_ctx->device_runtime->g_total_param_size -=
  //       ((m_kernel_entry->get_args_aligned_size() + 255) / 256 * 256);
  //   m_parent_kernel->remove_child(this);
  //   m_kernel_entry->gpgpu_ctx->the_gpgpusim->g_stream_manager
  //       ->register_finished_kernel(m_parent_kernel->get_uid());
  // }
}

void trace_kernel_info_t::print_parent_info() {
  if (m_parent_kernel) {
    printf("Parent %d: \'%s\', Block (%d, %d, %d), Thread (%d, %d, %d)\n",
           m_parent_kernel->get_uid(), m_parent_kernel->name().c_str(),
           m_parent_ctaid.x, m_parent_ctaid.y, m_parent_ctaid.z, m_parent_tid.x,
           m_parent_tid.y, m_parent_tid.z);
  }
}
