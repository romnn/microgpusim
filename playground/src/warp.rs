// const trace_warp_inst_t *trace_shd_warp_t::parse_trace_instruction(
//     const inst_trace_t &trace) const {
//   trace_warp_inst_t *new_inst =
//       new trace_warp_inst_t(get_shader()->get_config());
//
//   new_inst->parse_from_trace_struct(trace, m_kernel_info->OpcodeMap,
//                                     m_kernel_info->m_tconfig,
//                                     m_kernel_info->m_kernel_trace_info);
//   return new_inst;
// }
//

// void trace_kernel_info_t::get_next_threadblock_traces(
//     std::vector<std::vector<inst_trace_t> *> threadblock_traces) {
//   m_parser->get_next_threadblock_traces(
//       threadblock_traces, m_kernel_trace_info->trace_verion,
//       m_kernel_trace_info->enable_lineinfo, m_kernel_trace_info->ifs);
// }
