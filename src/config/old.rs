use super::{MHz, KB};
use crate::{cache, core::PipelineStage, mshr};
use std::collections::HashMap;
use std::sync::Arc;

impl super::GPU {
    pub fn old() -> Self {
        Self {
            log_after_cycle: None,
            parallelization: super::Parallelization::Serial,
            memory_only: false,
            accelsim_compat: false,
            simulate_clock_domains: false,
            simulation_threads: None,
            deadlock_check: false,
            // l2_prefetch_percent: None, // for TitanX
            l2_prefetch_percent: Some(90.0), // for TitanX
            // l2_prefetch_percent: 25.0, // for GTX 1080
            memory_controller_unit: std::sync::OnceLock::new(),
            occupancy_sm_number: 60,
            max_threads_per_core: 2048,
            warp_size: 32,
            clock_frequencies: super::ClockFrequenciesBuilder {
                core_freq_hz: 1417 * MHz,
                interconn_freq_hz: 1417 * MHz,
                l2_freq_hz: 1417 * MHz,
                dram_freq_hz: 2500 * MHz,
                // core_freq_hz: 1607 * MHz, // GTX1080
                // interconn_freq_hz: 1607 * MHz, // GTX1080
                // l2_freq_hz: 1607 * MHz, // GTX1080
                // dram_freq_hz: 1251 * MHz, // GTX1080
            }
            .build(),
            // N:16:128:24,L:R:m:N:L,F:128:4,128:2
            // {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>:<rf>}
            tex_cache_l1: Some(Arc::new(super::Cache {
                // accelsim_compat,
                kind: super::CacheKind::Normal,
                num_sets: 4, // 16,
                line_size: 128,
                associativity: 48, // 24,
                replacement_policy: cache::config::ReplacementPolicy::LRU,
                write_policy: cache::config::WritePolicy::READ_ONLY,
                allocate_policy: cache::config::AllocatePolicy::ON_MISS,
                write_allocate_policy: cache::config::WriteAllocatePolicy::NO_WRITE_ALLOCATE,
                // set_index_function: CacheSetIndexFunc::LINEAR_SET_FUNCTION,
                // set_index_function: Box::<cache::set_index::linear::SetIndex>::default(),
                mshr_kind: mshr::Kind::TEX_FIFO,
                mshr_entries: 128,
                mshr_max_merge: 4,
                miss_queue_size: 128,
                // result_fifo_entries: Some(2),
                l1_cache_write_ratio_percent: 0,
                data_port_width: None,
            })),
            // N:128:64:2,L:R:f:N:L,A:2:64,4
            // {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>}
            const_cache_l1: Some(Arc::new(super::Cache {
                // accelsim_compat,
                kind: super::CacheKind::Normal,
                num_sets: 128,
                line_size: 64,
                associativity: 2,
                replacement_policy: cache::config::ReplacementPolicy::LRU,
                write_policy: cache::config::WritePolicy::READ_ONLY,
                allocate_policy: cache::config::AllocatePolicy::ON_FILL,
                write_allocate_policy: cache::config::WriteAllocatePolicy::NO_WRITE_ALLOCATE,
                // set_index_function: CacheSetIndexFunc::LINEAR_SET_FUNCTION,
                // set_index_function: Box::<cache::set_index::linear::SetIndex>::default(),
                mshr_kind: mshr::Kind::ASSOC,
                mshr_entries: 2,
                mshr_max_merge: 64,
                miss_queue_size: 4,
                // result_fifo_entries: None,
                l1_cache_write_ratio_percent: 0,
                data_port_width: None,
            })),
            // N:8:128:4,L:R:f:N:L,A:2:48,4
            // total of 4KB (should be 8KB right?)
            // 32KB L1.5 (see volta dissect paper)
            // 2KB of L2 (see volta dissect paper)
            // {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>}
            inst_cache_l1: Some(Arc::new(super::Cache {
                // accelsim_compat,
                kind: super::CacheKind::Normal,
                num_sets: 8,
                line_size: 256,
                associativity: 4,
                replacement_policy: cache::config::ReplacementPolicy::LRU,
                write_policy: cache::config::WritePolicy::READ_ONLY,
                allocate_policy: cache::config::AllocatePolicy::ON_FILL,
                write_allocate_policy: cache::config::WriteAllocatePolicy::NO_WRITE_ALLOCATE,
                // set_index_function: CacheSetIndexFunc::LINEAR_SET_FUNCTION,
                // set_index_function: Box::<cache::set_index::linear::SetIndex>::default(),
                mshr_kind: mshr::Kind::ASSOC,
                mshr_entries: 2,
                mshr_max_merge: 48,
                miss_queue_size: 4,
                // result_fifo_entries: None,
                l1_cache_write_ratio_percent: 0,
                data_port_width: None,
            })),
            // N:64:128:6,L:L:m:N:H,A:128:8,8
            // {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq> | none}
            // l1 hit latency = 135.70804974
            // l2 hit latency = 274.3884858
            // l2 miss latency = 474.04434122
            data_cache_l1: Some(Arc::new(super::L1DCache {
                l1_latency: 1,
                l1_hit_latency: 81,
                // l1_banks_hashing_function: CacheSetIndexFunc::LINEAR_SET_FUNCTION,
                // l1_banks_hashing_function: Box::<cache::set_index::linear::SetIndex>::default(),
                l1_banks_byte_interleaving: 32,
                l1_banks: 1,
                inner: Arc::new(super::Cache {
                    // accelsim_compat,
                    kind: super::CacheKind::Sector,
                    // kind: CacheKind::Normal,
                    num_sets: 4, // 64,
                    line_size: 128,
                    associativity: 48, // 6,
                    replacement_policy: cache::config::ReplacementPolicy::LRU,
                    write_policy: cache::config::WritePolicy::LOCAL_WRITE_BACK_GLOBAL_WRITE_THROUGH,
                    allocate_policy: cache::config::AllocatePolicy::ON_MISS,
                    write_allocate_policy: cache::config::WriteAllocatePolicy::NO_WRITE_ALLOCATE,
                    // set_index_function: CacheSetIndexFunc::FERMI_HASH_SET_FUNCTION,
                    // set_index_function: Box::<cache::set_index::fermi::SetIndex>::default(),
                    mshr_kind: mshr::Kind::ASSOC,
                    // mshr_kind: mshr::Kind::SECTOR_ASSOC,
                    mshr_entries: 128,
                    mshr_max_merge: 8,
                    miss_queue_size: 4,
                    // result_fifo_entries: None,
                    l1_cache_write_ratio_percent: 0,
                    // l1_cache_write_ratio_percent: 50,
                    data_port_width: None,
                }),
            })),
            // N:64:128:16,L:B:m:W:L,A:1024:1024,4:0,32
            // {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>}
            data_cache_l2: Some(Arc::new(super::L2DCache {
                inner: Arc::new(super::Cache {
                    // accelsim_compat,
                    kind: super::CacheKind::Sector,
                    // kind: CacheKind::Normal,
                    // num_sets: 64,
                    // num_memory_controllers=8 * num_sub_partitions_per_memory_controller=2 = 16
                    // 16 * 64 * 128 * 16 = 2097152 = 2MiB
                    // 64 * 128 * 16 / 128 = 64 * 16 = 1024 lines per slice
                    // num_sets: 32,
                    // num_sets: 128,
                    // line_size: 128,
                    // associativity: 16,
                    // associativity: 16,
                    num_sets: 64,
                    line_size: 128,
                    associativity: 16,
                    replacement_policy: cache::config::ReplacementPolicy::LRU,
                    write_policy: cache::config::WritePolicy::WRITE_BACK,
                    allocate_policy: cache::config::AllocatePolicy::ON_MISS,
                    write_allocate_policy: cache::config::WriteAllocatePolicy::WRITE_ALLOCATE,
                    // set_index_function: CacheSetIndexFunc::LINEAR_SET_FUNCTION,
                    // set_index_function: Box::<cache::set_index::linear::SetIndex>::default(),
                    mshr_kind: mshr::Kind::ASSOC,
                    mshr_entries: 1024,
                    mshr_max_merge: 1024,
                    miss_queue_size: 4,
                    // result_fifo_entries: None, // 0 is none?
                    l1_cache_write_ratio_percent: 0,
                    data_port_width: Some(32),
                }),
            })),
            shared_memory_latency: 24, // 3 for GTX1080
            // TODO: make this better, or just parse accelsim configs
            max_sp_latency: 13,
            max_int_latency: 4,
            max_dp_latency: 19,
            max_sfu_latency: 8.max(330),
            global_mem_skip_l1_data_cache: false,
            perfect_mem: false,
            shader_registers: 65536,
            registers_per_block: 8192,
            ignore_resources_limitation: false,
            max_concurrent_blocks_per_core: 32,
            kernel_launch_latency: 0, // 5000,
            block_launch_latency: 0,
            max_barriers_per_block: 16,
            num_simt_clusters: 28, // 20 for GTX1080
            num_cores_per_simt_cluster: 1,
            num_cluster_ejection_buffer_size: 32, // 8 for GTX1080
            num_ldst_response_buffer_size: 2,
            shared_memory_per_block: 48 * KB as usize,
            shared_memory_size: 96 * KB as u32,
            shared_memory_option: false,
            unified_l1_data_cache_size: false,
            adaptive_cache_config: false,
            shared_memory_sizes: vec![],
            shared_memory_size_pref_l1: 16 * KB as usize,
            shared_memory_size_pref_shared: 16 * KB as usize,
            shared_memory_num_banks: 32,
            shared_memory_limited_broadcast: false,
            shared_memory_warp_parts: 1,
            mem_unit_ports: 1,
            warp_distro_shader_core: -1,
            warp_issue_shader_core: 0,
            local_mem_map: true,
            num_reg_banks: 16, // 32 for GTX1080
            reg_bank_use_warp_id: false,
            sub_core_model: true,                        // false for GTX 1080 ?
            enable_specialized_operand_collector: false, // true for GTX 1080 ?
            // specialized collectors (deprecated GTX 1080 config)
            operand_collector_num_units_sp: 20, // 4,
            operand_collector_num_units_dp: 0,
            operand_collector_num_units_sfu: 4,
            operand_collector_num_units_int: 0,
            operand_collector_num_units_tensor_core: 4,
            operand_collector_num_units_mem: 8,   // 2,
            operand_collector_num_in_ports_sp: 4, // 1,
            operand_collector_num_in_ports_dp: 0,
            operand_collector_num_in_ports_sfu: 1,
            operand_collector_num_in_ports_int: 0,
            operand_collector_num_in_ports_tensor_core: 1,
            operand_collector_num_in_ports_mem: 1,
            operand_collector_num_out_ports_sp: 4, // 1,
            operand_collector_num_out_ports_dp: 0,
            operand_collector_num_out_ports_sfu: 1,
            operand_collector_num_out_ports_int: 0,
            operand_collector_num_out_ports_tensor_core: 1,
            operand_collector_num_out_ports_mem: 1,
            // generic collectors
            operand_collector_num_units_gen: 8,
            operand_collector_num_in_ports_gen: 8,
            operand_collector_num_out_ports_gen: 8,
            coalescing_arch: super::Architecture::Pascal,
            num_schedulers_per_core: 4,
            max_instruction_issue_per_warp: 2,
            dual_issue_only_to_different_exec_units: true,
            simt_core_sim_order: super::SchedulingOrder::RoundRobin,
            pipeline_widths: HashMap::from_iter([
                (PipelineStage::ID_OC_SP, 4),
                (PipelineStage::ID_OC_DP, 0),
                (PipelineStage::ID_OC_INT, 0),
                (PipelineStage::ID_OC_SFU, 4), // 1 GTX1080
                (PipelineStage::ID_OC_MEM, 4), // 1 GTX1080
                (PipelineStage::OC_EX_SP, 4),
                (PipelineStage::OC_EX_DP, 0),
                (PipelineStage::OC_EX_INT, 0),
                (PipelineStage::OC_EX_SFU, 4), // 1 GTX1080
                (PipelineStage::OC_EX_MEM, 4), // 1 GTX1080
                (PipelineStage::EX_WB, 8),     // 6 GTX1080
                // don't have tensor cores
                (PipelineStage::ID_OC_TENSOR_CORE, 0),
                (PipelineStage::OC_EX_TENSOR_CORE, 0),
            ]),
            num_sp_units: 4,
            num_dp_units: 0,
            num_int_units: 0,
            num_sfu_units: 4, // 1 GTX1080 ?
            num_tensor_core_avail: 0,
            num_tensor_core_units: 0,
            scheduler: super::CoreSchedulerKind::GTO,
            concurrent_kernel_sm: false,
            perfect_inst_const_cache: false, // true
            inst_fetch_throughput: 1,
            reg_file_port_throughput: 2, // 1 for GTX1080
            fill_l2_on_memcopy: true,
            // simple_dram_model: false,
            dram_scheduler: super::DRAMSchedulerKind::FrFcfs,
            dram_partition_queue_interconn_to_l2: 8,
            dram_partition_queue_l2_to_dram: 8,
            dram_partition_queue_dram_to_l2: 8,
            dram_partition_queue_l2_to_interconn: 8,
            ideal_l2: false,
            data_cache_l2_texture_only: false,
            num_memory_controllers: 12, // 8 for GTX1080
            num_sub_partitions_per_memory_controller: 2,
            num_dram_chips_per_memory_controller: 1,
            dram_frfcfs_sched_queue_size: 64,
            dram_return_queue_size: 64, // 116 for GTX 1080?
            dram_buswidth: 4,
            dram_burst_length: 8,
            dram_data_command_freq_ratio: 4,
            // "nbk=16:CCD=2:RRD=6:RCD=12:RAS=28:RP=12:RC=40:
            // CL=12:WL=4:CDLR=5:WR=12:nbkgrp=1:CCDL=0:RTPL=0"
            dram_timing_options: super::TimingOptions { num_banks: 16 },
            // this is the l2 latency 216 L2 latency
            // l2_rop_latency: 1,
            // dram_latency: 1,
            l2_rop_latency: 210, // was 120
            dram_latency: 190,   // was 100
            dram_dual_bus_interface: false,
            dram_bank_indexing_policy: super::DRAMBankIndexPolicy::Normal,
            dram_bank_group_indexing_policy: super::DRAMBankGroupIndexPolicy::LowerBits,
            dram_seperate_write_queue_enable: false,
            dram_frfcfs_write_queue_size: 32, // 32:28:16
            dram_elimnate_rw_turnaround: false,
            memory_addr_mapping: Some(
                "dramid@8;00000000.00000000.00000000.00000000.0000RRRR.RRRRRRRR.RBBBCCCC.BCCSSSSS"
                    .to_string(),
            ),
            memory_address_mask: super::MemoryAddressingMask::New, // 1
            memory_partition_indexing: super::MemoryPartitionIndexingScheme::Consecutive,
            // memory_partition_indexing: MemoryPartitionIndexingScheme::BitwiseXor,
            // memory_partition_indexing: MemoryPartitionIndexingScheme::IPoly,
            compute_capability_major: 6,
            compute_capability_minor: 1,
            flush_l1_cache: false,
            flush_l2_cache: false,
            max_concurrent_kernels: 32,
            // from gpgpusim.trace.config
            // trace_opcode_latency_initiation_int: (2, 2), // default 4, 1
            // trace_opcode_latency_initiation_sp: (2, 1),  // default 4, 1
            // trace_opcode_latency_initiation_dp: (64, 64), // default 4, 1
            // trace_opcode_latency_initiation_sfu: (21, 8), // default 4, 1
            // trace_opcode_latency_initiation_tensor: (32, 32), // default 4, 1
            //
            /// On Pascal, most integer and single-precision instructions have a latency of 6 cycles.
            /// Double-precision instructions have latency 8 cycles
            /// More complex instructions, some of which run on the SFU,
            /// require 14 cycles.
            /// On Maxwell and Pascal, instructions IMAD and IMUL have
            /// a long latency because they are emulated
            ///
            /// source: https://arxiv.org/pdf/1804.06826.pdf
            // format: fill latency and init latency
            trace_opcode_latency_initiation_int: (6, 1),
            trace_opcode_latency_initiation_sp: (6, 1),
            trace_opcode_latency_initiation_dp: (8, 8), // (4, 1)
            trace_opcode_latency_initiation_sfu: (14, 4), // (4, 1)
            /// does not have tensor units
            trace_opcode_latency_initiation_tensor: (usize::MAX, 1),
            // trace_opcode_latency_initiation_int: (4, 1),
            // trace_opcode_latency_initiation_sp: (4, 1),
            // trace_opcode_latency_initiation_dp: (20, 8), // (4, 1)
            // trace_opcode_latency_initiation_sfu: (20, 4), // (4, 1)
            // trace_opcode_latency_initiation_tensor: (4, 1),
        }
    }
}
