#pragma once

#include "ref/addrdec.hpp"                   // done
#include "ref/barrier_set.hpp"               // done
#include "ref/cache_access_logger_types.hpp" // done
#include "ref/cache_config.hpp"              // done
#include "ref/cache_sub_stats.hpp"           // done
#include "ref/stats_wrapper.hpp"           // done
#include "ref/command_type.hpp"              // done
#include "ref/cu_event.hpp"                  // done
#include "ref/cu_stream.hpp"                 // done
#include "ref/cuda_sim.hpp"                  // done
#include "ref/cache.hpp"                // done
#include "ref/data_cache.hpp"                // done
#include "ref/dim3.hpp"                      // done
#include "ref/dram.hpp"                      // done
#include "ref/exec_unit_type.hpp"            // done
#include "ref/fifo.hpp"                      // done
#include "ref/frfcfs_scheduler.hpp" // done
#include "ref/function_info.hpp"               // done
#include "ref/gpgpu_context.hpp"               // done
#include "ref/gpgpu_functional_sim_config.hpp" // done
#include "ref/gpgpu_sim.hpp"
#include "ref/gpgpu_sim_config.hpp" // done
#include "ref/gpgpu_sim_ctx.hpp"
#include "ref/hal.hpp" // done
#include "ref/icnt_wrapper.hpp"
#include "ref/ifetch_buffer.hpp"         // done
#include "ref/inst_memadd_info.hpp"      // done
#include "ref/inst_trace.hpp"            // done
#include "ref/instr.hpp"                 // done
#include "ref/kernel_trace.hpp"          // done?
#include "ref/l1_cache.hpp"              // done
#include "ref/l2_cache.hpp"              // done
#include "ref/l2_cache_config.hpp"       // done
#include "ref/ldst_unit.hpp"             // done
#include "ref/local_interconnect.hpp"    // done
#include "ref/lrr_scheduler.hpp"         // done
#include "ref/main.hpp"                  // done
#include "ref/mem_fetch.hpp"             // done
#include "ref/mem_fetch_allocator.hpp"   // done
#include "ref/mem_fetch_interface.hpp"   // done
#include "ref/mem_stage_access_type.hpp" // done
#include "ref/mem_stage_stall_type.hpp"  // done
#include "ref/memory_config.hpp"         // done
#include "ref/memory_partition_unit.hpp"
#include "ref/opcode_char.hpp"         // done
#include "ref/operand_info.hpp"        // done
#include "ref/operand_type.hpp"        // done
#include "ref/opndcoll_rfu.hpp"        // done
#include "ref/option_parser.hpp"       // done
#include "ref/pipelined_simd_unit.hpp" // done
#include "ref/memory_sub_partition.hpp" // done
#include "ref/partition_mf_allocator.hpp" // done
#include "ref/l2_interface.hpp"
#include "ref/ptx_cta_info.hpp"
#include "ref/ptx_instruction.hpp"
#include "ref/ptx_reg.hpp"
#include "ref/ptx_thread_info.hpp"
#include "ref/read_only_cache.hpp"    // done
#include "ref/rec_pts.hpp"            // done
#include "ref/register_set.hpp"       // done
#include "ref/scheduler_unit.hpp"     // done
#include "ref/scoreboard.hpp"         // done
#include "ref/shader_core_config.hpp" // done
#include "ref/shader_core_ctx.hpp"
#include "ref/shader_core_mem_fetch_allocator.hpp" // done
#include "ref/shader_trace.hpp"                    //done
#include "ref/shd_warp.hpp"                        // done
#include "ref/simd_function_unit.hpp"              // done
#include "ref/simt_core_cluster.hpp"               // done
#include "ref/stream_manager.hpp"                  // done
#include "ref/stream_operation.hpp"                // done
#include "ref/symbol_table.hpp"                    // done
#include "ref/tag_array.hpp"                       // done
#include "ref/tex_cache.hpp"                       // done
#include "ref/thread_ctx.hpp"                      // done
#include "ref/trace.hpp"                           // done
#include "ref/trace_command.hpp"                   // done
#include "ref/trace_config.hpp"                    // done
#include "ref/trace_function_info.hpp"             // done?
#include "ref/trace_gpgpu_sim.hpp"
#include "ref/trace_kernel_info.hpp"       // done?
#include "ref/trace_parser.hpp"            // done
#include "ref/trace_shader_core_ctx.hpp"   // done
#include "ref/trace_shd_warp.hpp"          // done
#include "ref/trace_simt_core_cluster.hpp" // done
#include "ref/warp_instr.hpp"              // done
#include "ref/warp_set.hpp"                // done

#include "tests/parse_cache_config.hpp"
