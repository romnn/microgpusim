/// CacheConfig
///
/// gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.h#565
///
/// TODO: Find out what those values are.
/// sscanf(config, "%c:%u:%u:%u,%c:%c:%c:%c:%c,%c:%u:%u,%u:%u,%u", &ct,
/// &m_nset, &m_line_sz, &m_assoc, &rp, &wp, &ap, &wap, &sif,
/// &mshr_type, &m_mshr_entries, &m_mshr_max_merge,
/// &m_miss_queue_size, &m_result_fifo_entries, &m_data_port_width);
pub struct CacheConfig {
    pub kind: CacheKind,
    pub num_sets: usize,
    pub line_size: usize,
    pub associativity: usize,

    pub replacement_policy: CacheReplacementPolicy,
    pub write_policy: CacheWritePolicy,
    pub allocate_policy: CacheAllocatePolicy,
    pub write_allocate_policy: CacheWriteAllocatePolicy,
    pub set_index_function: CacheSetIndexingFunction,

    pub mshr_kind: MshrKind,
    pub mshr_entries: usize,
    pub mshr_max_merge: usize,

    pub miss_queue_size: usize,
    pub result_fifo_entries: Option<usize>,
    pub data_port_width: Option<usize>,
}

impl CacheConfig {
    // m_line_sz_log2 = LOGB2(m_line_sz);
    // m_nset_log2 = LOGB2(m_nset);
    // m_valid = true;
    // m_atom_sz = (m_cache_type == SECTOR) ? SECTOR_SIZE : m_line_sz;
    // m_sector_sz_log2 = LOGB2(SECTOR_SIZE);
    // original_m_assoc = m_assoc;

    // // detect invalid configuration
    // if ((m_alloc_policy == ON_FILL || m_alloc_policy == STREAMING) and
    //     m_write_policy == WRITE_BACK) {
    //   // A writeback cache with allocate-on-fill policy will inevitably lead to
    //   // deadlock: The deadlock happens when an incoming cache-fill evicts a
    //   // dirty line, generating a writeback request.  If the memory subsystem is
    //   // congested, the interconnection network may not have sufficient buffer
    //   // for the writeback request.  This stalls the incoming cache-fill.  The
    //   // stall may propagate through the memory subsystem back to the output
    //   // port of the same core, creating a deadlock where the wrtieback request
    //   // and the incoming cache-fill are stalling each other.
    //   assert(0 &&
    //          "Invalid cache configuration: Writeback cache cannot allocate new "
    //          "line on fill. ");
    // }
    //
    // if ((m_write_alloc_policy == FETCH_ON_WRITE ||
    //      m_write_alloc_policy == LAZY_FETCH_ON_READ) &&
    //     m_alloc_policy == ON_FILL) {
    //   assert(
    //       0 &&
    //       "Invalid cache configuration: FETCH_ON_WRITE and LAZY_FETCH_ON_READ "
    //       "cannot work properly with ON_FILL policy. Cache must be ON_MISS. ");
    // }
    // if (m_cache_type == SECTOR) {
    //   assert(m_line_sz / SECTOR_SIZE == SECTOR_CHUNCK_SIZE &&
    //          m_line_sz % SECTOR_SIZE == 0);
    // }
    //
    // // default: port to data array width and granularity = line size
    // if (m_data_port_width == 0) {
    //   m_data_port_width = m_line_sz;
    // }
    // assert(m_line_sz % m_data_port_width == 0);
}

pub enum CacheSetIndexingFunction {
    FERMI_HASH_SET_FUNCTION, // H
    HASH_IPOLY_FUNCTION,     // P
    // CUSTOM_SET_FUNCTION, // C
    LINEAR_SET_FUNCTION,     // L
    BITWISE_XORING_FUNCTION, // X
}

pub enum MshrKind {
    TEX_FIFO,        // F
    SECTOR_TEX_FIFO, // T
    ASSOC,           // A
    SECTOR_ASSOC,    // S
}

pub enum CacheKind {
    Normal, // N
    Sector, // S
}

pub enum CacheReplacementPolicy {
    LRU,  // L
    FIFO, // F
}

pub enum CacheWriteAllocatePolicy {
    NO_WRITE_ALLOCATE,  // N
    WRITE_ALLOCATE,     // W
    FETCH_ON_WRITE,     // F
    LAZY_FETCH_ON_READ, // L
}

pub enum CacheWritePolicy {
    READ_ONLY,          // R
    WRITE_BACK,         // B
    WRITE_THROUGH,      // T
    WRITE_EVICT,        // E
    LOCAL_WB_GLOBAL_WT, // L
}

pub enum CacheAllocatePolicy {
    ON_MISS,   // M
    ON_FILL,   // F
    STREAMING, // S
}

pub struct GPUConfig {
    /// The SM number to pass to ptxas when getting register usage for
    /// computing GPU occupancy.
    pub occupancy_sm_number: usize,
    /// num threads per shader core pipeline
    pub max_threads_per_shader: usize,
    /// shader core pipeline warp size
    pub warp_size: usize,
    /// per-shader read-only L1 texture cache config
    pub tex_cache_l1: Option<CacheConfig>,
    /// per-shader read-only L1 constant memory cache config
    pub const_cache_l1: Option<CacheConfig>,
    /// shader L1 instruction cache config
    pub inst_cache_l1: Option<CacheConfig>,
    /// per-shader L1 data cache config
    pub data_cache_l1: Option<CacheConfig>,
    /// unified banked L2 data cache config
    pub data_cache_l2: Option<CacheConfig>,

    /// L1D write ratio
    pub l1_cache_write_ratio: usize,
    /// The number of L1 cache banks
    pub l1_banks: usize,
    /// L1 banks byte interleaving granularity
    pub l1_banks_byte_interleaving: usize,
    // L1 banks hashing function
    pub l1_banks_hashing_function: usize,
    /// L1 Hit Latency
    pub l1_latency: usize,
    /// smem Latency
    pub shared_memory_latency: usize,
    /// implements -Xptxas -dlcm=cg, default=no skip
    pub global_mem_skip_l1_data_cache: bool,
    // -gpgpu_cache:dl1PrefL1                 none # per-shader L1 data cache config  {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq> | none}
    // -gpgpu_cache:dl1PrefShared                 none # per-shader L1 data cache config  {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq> | none}
    /// Number of registers per shader core.
    /// Limits number of concurrent CTAs. (default 8192)
    shader_registers: usize, // 65536
    /// Maximum number of registers per CTA. (default 8192)
    registers_per_block: usize, //  8192
    ignore_resources_limitation: bool, // 0
    /// Maximum number of concurrent CTAs in shader (default 32)
    shader_max_concurrent_cta: usize, // 32
    /// Maximum number of named barriers per CTA (default 16)
    num_cta_barriers: usize, // 16
    /// number of processing clusters
    num_simt_clusters: usize, //  20
    /// number of simd cores per cluster
    num_cores_per_simt_cluster: usize, // 1
    /// number of packets in ejection buffer
    num_cluster_ejection_buffer_size: usize, // 8
    /// number of response packets in ld/st unit ejection buffer
    num_ldst_response_buffer_size: usize, //  2
    /// Size of shared memory per thread block or CTA (default 48kB)
    shared_memory_per_block: usize, // 49152
    /// Size of shared memory per shader core (default 16kB)
    shared_memory_size: usize, // 98304
    /// Option list of shared memory sizes
    shared_memory_option: bool, // 0
    /// Size of unified data cache(L1D + shared memory) in KB
    unified_l1_data_cache_size: bool, //0
    /// adaptive_cache_config
    adaptive_cache_config: bool, // 0
    // Size of shared memory per shader core (default 16kB)
    // shared_memory_size_default: usize, // 16384
    /// Size of shared memory per shader core (default 16kB)
    shared_memory_size_pref_l1: usize, // 16384
    /// Size of shared memory per shader core (default 16kB)
    shared_memory_size_pref_shared: usize, // 16384
    /// Number of banks in the shared memory in each shader core (default 16)
    shared_memory_num_banks: usize, // 32
    /// Limit shared memory to do one broadcast per cycle (default on)
    shared_memory_limited_broadcast: bool, // 0
    /// Number of portions a warp is divided into for shared memory bank conflict check
    shared_memory_warp_parts: usize, // 1
    /// The number of memory transactions allowed per core cycle
    mem_unit_ports: usize, // 1
    /// Specify which shader core to collect the warp size distribution from
    warp_distro_shader_core: i32, // -1
    /// Specify which shader core to collect the warp issue distribution from
    warp_issue_shader_core: i32, // 0
    /// Mapping from local memory space address to simulated GPU physical address space
    local_mem_map: bool, // 1
    /// Number of register banks (default = 8)
    num_reg_banks: usize, // 32
    /// Use warp ID in mapping registers to banks (default = off)
    reg_bank_use_warp_id: bool, // 0
    /// Sub Core Volta/Pascal model (default = off)
    sub_core_model: bool, // 0
    /// Coalescing arch (GT200 = 13, Fermi = 20)
    coalescing_arch: Architecture, // 13
    /// Number of warp schedulers per core
    num_schedulers_per_core: usize, // 2
    /// Max number of instructions that can be issued per warp in one cycle by scheduler (either 1 or 2)
    max_instruction_issue_per_warp: usize, // 2
    /// should dual issue use two different execution unit resources
    dual_issue_diff_exec_units: bool, // 1
    /// Select the simulation order of cores in a cluster
    simt_core_sim_order: SchedulingOrder, // 1
    /// Number if ldst units (default=1) WARNING: not hooked up to anything
    num_mem_units: usize, // 1
    /// Scheduler configuration: < lrr | gto | two_level_active > If two_level_active:<num_active_warps>:<inner_prioritization>:<outer_prioritization>For complete list of prioritization values see shader.h enum scheduler_prioritization_typeDefault: gto
    scheduler: CoreSchedulerKind, // gto
    /// Support concurrent kernels on a SM (default = disabled)
    concurrent_kernel_sm: bool, // 0
    /// perfect inst and const cache mode, so all inst and const hits in the cache(default = disabled)
    perfect_inst_const_cache: bool, // 0
    /// the number of fetched intruction per warp each cycle
    inst_fetch_throughput: usize, // 1
    /// the number ports of the register file
    reg_file_port_throughput: usize, // 1
    /// Fill the L2 cache on memcpy
    fill_l2_on_memcopy: bool, // true
    /// simple_dram_model with fixed latency and BW
    simple_dram_model: bool, // 0
    /// DRAM scheduler kind. 0 = fifo, 1 = FR-FCFS (default)
    dram_scheduler: DRAMSchedulerKind, // 1
    /// DRAM partition queue i2$:$2d:d2$:$2i
    dram_partition_queues: usize, // 8:8:8:8
    /// use a ideal L2 cache that always hit
    ideal_l2: bool, // 0
    /// L2 cache used for texture only
    data_cache_l2_texture_only: bool, // 0
    /// number of memory modules (e.g. memory controllers) in gpu
    num_memory_controllers: usize, // 8
    /// number of memory subpartition in each memory module
    num_sub_partition_per_memory_channel: usize, // 2
    /// number of memory chips per memory controller
    num_memory_chips_per_controller: usize, // 1
    /// track and display latency statistics 0x2 enables MC, 0x4 enables queue logs
    // memory_latency_stat: usize, // 14
    /// DRAM scheduler queue size 0 = unlimited (default); # entries per chip
    frfcfs_dram_sched_queue_size: usize, // 64
    /// 0 = unlimited (default); # entries per chip
    dram_return_queue_size: usize, // 116
    /// default = 4 bytes (8 bytes per cycle at DDR)
    dram_buswidth: usize, // 4
    /// Burst length of each DRAM request (default = 4 data bus cycle)
    dram_burst_length: usize, // 8
    /// Frequency ratio between DRAM data bus and command bus (default = 2 times, i.e. DDR)
    dram_data_command_freq_ratio: usize, // 4
    /// DRAM timing parameters =
    /// {nbk:tCCD:tRRD:tRCD:tRAS:tRP:tRC:CL:WL:tCDLR:tWR:nbkgrp:tCCDL:tRTPL}
    /// dram_timing_opt
    /// nbk=16:CCD=2:RRD=6:RCD=12:RAS=28:RP=12:RC=40: CL=12:WL=4:CDLR=5:WR=12:nbkgrp=1:CCDL=0:RTPL=0
    /// ROP queue latency (default 85)
    l2_rop_latency: usize, // 120
    /// DRAM latency (default 30)
    dram_latency: usize, // 100
    /// dual_bus_interface (default = 0)
    dram_dual_bus_interface: bool, // 0
    /// dram_bnk_indexing_policy
    dram_bank_indexing_policy: DRAMBankIndexPolicy, // 0
    /// dram_bnkgrp_indexing_policy
    dram_bank_group_indexing_policy: DRAMBankGroupIndexPolicy, // 0
    /// Seperate_Write_Queue_Enable
    dram_seperate_write_queue_enable: bool, // 0
    /// write_Queue_Size
    // dram_write_queue_size: usize, // 32:28:16
    /// elimnate_rw_turnaround i.e set tWTR and tRTW = 0
    dram_elimnate_rw_turnaround: bool, // 0
    /// mapping memory address to dram model
    /// {dramid@<start bit>;<memory address map>}
    // memory_addr_mapping: String, // dramid@8;00000000.00000000.00000000.00000000.0000RRRR.RRRRRRRR.RBBBCCCC.BCCSSSSS
    /// run sweep test to check address mapping for aliased address
    // memory_addr_test: bool, // 0
    /// 0 = old addressing mask, 1 = new addressing mask, 2 = new add. mask + flipped bank sel and chip sel bits
    // memory_address_mask: usize, // 1
    memory_partition_indexing: MemoryPartitionIndexingScheme, // 0
    /// Major compute capability version number
    compute_capability_major: usize, // 7
    /// Minor compute capability version number
    compute_capability_minor: usize, // 0
    /// Flush L1 cache at the end of each kernel call
    flush_l1_cache: bool, // 0
    /// Flush L2 cache at the end of each kernel call
    flush_l2_cache: bool, // 0
    /// maximum kernels that can run concurrently on GPU.
    ///
    /// Set this value according to max resident grids for your
    /// compute capability.
    max_concurrent_kernels: usize, // 32
}

/// Memory partition indexing scheme.
///
/// 0 = no indexing, 1 = bitwise xoring, 2 = IPoly, 3 = custom indexing
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryPartitionIndexingScheme {
    None = 0,
    BitwiseXor = 1,
    IPoly = 2,
}

/// DRAM bank group indexing policy.
///
/// 0 = take higher bits, 1 = take lower bits
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DRAMBankGroupIndexPolicy {
    HigherBits = 0,
    LowerBits = 1,
}

/// DRAM bank indexing policy.
///
/// 0 = normal indexing, 1 = Xoring with the higher bits
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DRAMBankIndexPolicy {
    Normal = 0,
    Xor = 1,
}

/// DRAM Scheduler policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DRAMSchedulerKind {
    FIFO = 0,
    FrFcfs = 1,
}

/// Core Scheduler policy.
///
/// If two_level_active:
/// <num_active_warps>:<inner_prioritization>:<outer_prioritization>
///
/// For complete list of prioritization values see shader.h.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CoreSchedulerKind {
    LRR,
    GTO,
    TwoLevelActive,
}

/// GPU microarchitecture generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Architecture {
    GT200 = 13,
    Fermi = 20,
}

/// Scheduling order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SchedulingOrder {
    Fix = 0,
    RoundRobin = 1,
}

impl Default for GPUConfig {
    fn default() -> Self {
        Self {
            occupancy_sm_number: 60,
            max_threads_per_shader: 2048,
            warp_size: 32,
            // N:16:128:24,L:R:m:N:L,F:128:4,128:2
            // {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>:<rf>}
            tex_cache_l1: Some(CacheConfig {
                kind: CacheKind::Normal,
                num_sets: 16,
                line_size: 128,
                associativity: 24,
                replacement_policy: CacheReplacementPolicy::LRU,
                write_policy: CacheWritePolicy::READ_ONLY,
                allocate_policy: CacheAllocatePolicy::ON_MISS,
                write_allocate_policy: CacheWriteAllocatePolicy::NO_WRITE_ALLOCATE,
                set_index_function: CacheSetIndexingFunction::LINEAR_SET_FUNCTION,
                mshr_kind: MshrKind::TEX_FIFO,
                mshr_entries: 128,
                mshr_max_merge: 4,
                miss_queue_size: 128,
                result_fifo_entries: Some(2),
                data_port_width: None,
            }),
            // N:128:64:2,L:R:f:N:L,A:2:64,4
            // {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>}
            const_cache_l1: Some(CacheConfig {
                kind: CacheKind::Normal,
                num_sets: 128,
                line_size: 64,
                associativity: 2,
                replacement_policy: CacheReplacementPolicy::LRU,
                write_policy: CacheWritePolicy::READ_ONLY,
                allocate_policy: CacheAllocatePolicy::ON_FILL,
                write_allocate_policy: CacheWriteAllocatePolicy::NO_WRITE_ALLOCATE,
                set_index_function: CacheSetIndexingFunction::LINEAR_SET_FUNCTION,
                mshr_kind: MshrKind::ASSOC,
                mshr_entries: 2,
                mshr_max_merge: 64,
                miss_queue_size: 4,
                result_fifo_entries: None,
                data_port_width: None,
            }),
            // N:8:128:4,L:R:f:N:L,A:2:48,4
            // {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>}
            inst_cache_l1: Some(CacheConfig {
                kind: CacheKind::Normal,
                num_sets: 8,
                line_size: 128,
                associativity: 4,
                replacement_policy: CacheReplacementPolicy::LRU,
                write_policy: CacheWritePolicy::READ_ONLY,
                allocate_policy: CacheAllocatePolicy::ON_FILL,
                write_allocate_policy: CacheWriteAllocatePolicy::NO_WRITE_ALLOCATE,
                set_index_function: CacheSetIndexingFunction::LINEAR_SET_FUNCTION,
                mshr_kind: MshrKind::ASSOC,
                mshr_entries: 2,
                mshr_max_merge: 48,
                miss_queue_size: 4,
                result_fifo_entries: None,
                data_port_width: None,
            }),
            // N:64:128:6,L:L:m:N:H,A:128:8,8
            // {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq> | none}
            data_cache_l1: Some(CacheConfig {
                kind: CacheKind::Normal,
                num_sets: 64,
                line_size: 128,
                associativity: 6,
                replacement_policy: CacheReplacementPolicy::LRU,
                write_policy: CacheWritePolicy::LOCAL_WB_GLOBAL_WT,
                allocate_policy: CacheAllocatePolicy::ON_MISS,
                write_allocate_policy: CacheWriteAllocatePolicy::NO_WRITE_ALLOCATE,
                set_index_function: CacheSetIndexingFunction::FERMI_HASH_SET_FUNCTION,
                mshr_kind: MshrKind::ASSOC,
                mshr_entries: 128,
                mshr_max_merge: 8,
                miss_queue_size: 4,
                result_fifo_entries: None,
                data_port_width: None,
            }),
            // N:64:128:16,L:B:m:W:L,A:1024:1024,4:0,32
            // {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>}
            data_cache_l2: Some(CacheConfig {
                kind: CacheKind::Normal,
                num_sets: 64,
                line_size: 128,
                associativity: 16,
                replacement_policy: CacheReplacementPolicy::LRU,
                write_policy: CacheWritePolicy::WRITE_BACK,
                allocate_policy: CacheAllocatePolicy::ON_MISS,
                write_allocate_policy: CacheWriteAllocatePolicy::WRITE_ALLOCATE,
                set_index_function: CacheSetIndexingFunction::LINEAR_SET_FUNCTION,
                mshr_kind: MshrKind::ASSOC,
                mshr_entries: 128,
                mshr_max_merge: 8,
                miss_queue_size: 32,
                result_fifo_entries: None,
                data_port_width: None,
            }),
            l1_cache_write_ratio: 0,
            l1_banks: 1,
            l1_banks_byte_interleaving: 32,
            l1_banks_hashing_function: 0,
            l1_latency: 1,
            shared_memory_latency: 3,
            global_mem_skip_l1_data_cache: true,
            shader_registers: 65536,
            registers_per_block: 8192,
            ignore_resources_limitation: false,
            shader_max_concurrent_cta: 32,
            num_cta_barriers: 16,
            num_simt_clusters: 20,
            num_cores_per_simt_cluster: 1,
            num_cluster_ejection_buffer_size: 8,
            num_ldst_response_buffer_size: 2,
            shared_memory_per_block: 49152,
            shared_memory_size: 98304,
            shared_memory_option: false,
            unified_l1_data_cache_size: false,
            adaptive_cache_config: false,
            shared_memory_size_pref_l1: 16384,
            shared_memory_size_pref_shared: 16384,
            shared_memory_num_banks: 32,
            shared_memory_limited_broadcast: false,
            shared_memory_warp_parts: 1,
            mem_unit_ports: 1,
            warp_distro_shader_core: -1,
            warp_issue_shader_core: 0,
            local_mem_map: true,
            num_reg_banks: 32,
            reg_bank_use_warp_id: false,
            sub_core_model: false,
            coalescing_arch: Architecture::GT200,
            num_schedulers_per_core: 2,
            max_instruction_issue_per_warp: 2,
            dual_issue_diff_exec_units: true,
            simt_core_sim_order: SchedulingOrder::RoundRobin,
            num_mem_units: 1,
            scheduler: CoreSchedulerKind::GTO,
            concurrent_kernel_sm: false,
            perfect_inst_const_cache: false,
            inst_fetch_throughput: 1,
            reg_file_port_throughput: 1,
            fill_l2_on_memcopy: true,
            simple_dram_model: false,
            dram_scheduler: DRAMSchedulerKind::FrFcfs,
            /// DRAM partition queue i2$:$2d:d2$:$2i
            dram_partition_queues: 0, // 8:8:8:8
            ideal_l2: false,
            data_cache_l2_texture_only: false,
            num_memory_controllers: 8,
            num_sub_partition_per_memory_channel: 2,
            num_memory_chips_per_controller: 1,
            frfcfs_dram_sched_queue_size: 64,
            dram_return_queue_size: 116,
            dram_buswidth: 4,
            dram_burst_length: 8,
            dram_data_command_freq_ratio: 4,
            l2_rop_latency: 120,
            dram_latency: 100,
            dram_dual_bus_interface: false,
            dram_bank_indexing_policy: DRAMBankIndexPolicy::Normal,
            dram_bank_group_indexing_policy: DRAMBankGroupIndexPolicy::HigherBits,
            dram_seperate_write_queue_enable: false,
            // dram_write_queue_size: usize, // 32:28:16
            dram_elimnate_rw_turnaround: false,
            memory_partition_indexing: MemoryPartitionIndexingScheme::None,
            compute_capability_major: 7,
            compute_capability_minor: 0,
            flush_l1_cache: false,
            flush_l2_cache: false,
            max_concurrent_kernels: 32,
        }
    }
}
