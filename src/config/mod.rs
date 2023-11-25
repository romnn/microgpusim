pub mod accelsim;
pub mod gtx1080;

use crate::{
    address, cache, core::PipelineStage, kernel::Kernel, mcu, mem_sub_partition, mshr, opcodes,
};
use color_eyre::eyre;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

pub use gtx1080::GTX1080;

pub const KB: u64 = 1024;
pub const MB: u64 = 1024 * KB;
pub const GB: u64 = 1024 * MB;
#[allow(non_upper_case_globals)]
pub const Hz: u64 = 1;
#[allow(non_upper_case_globals)]
pub const KHz: u64 = 1000 * Hz;
#[allow(non_upper_case_globals)]
pub const MHz: u64 = 1000 * KHz;
#[allow(non_upper_case_globals)]
pub const GHz: u64 = 1000 * MHz;

/// Memory addressing mask
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MemoryAddressingMask {
    Old,
    New,
    NewFlippedSelectorBits,
}

/// Cache kind
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CacheKind {
    Normal, // N
    Sector, // S
}

#[derive(Debug)]
pub struct L2DCache {
    pub inner: Arc<Cache>,
}

#[derive(Debug)]
pub struct L1DCache {
    /// L1 Hit Latency
    pub l1_latency: usize, // 1
    pub l1_hit_latency: usize, // 80
    /// l1 banks hashing function
    // pub l1_banks_hashing_function: Box<dyn cache::set_index::SetIndexer>, // 0
    // pub l1_banks_hashing_function: CacheSetIndexFunc, // 0
    /// l1 banks byte interleaving granularity
    pub l1_banks_byte_interleaving: usize, // 32
    /// The number of L1 cache banks
    pub l1_banks: usize, // 1

    pub inner: Arc<Cache>,
}

impl L1DCache {
    // // #[inline]
    // #[must_use]
    // pub fn l1_banks_log2(&self) -> u32 {
    //     self.l1_banks.ilog2()
    // }

    // // #[inline]
    // #[must_use]
    // pub fn l1_banks_byte_interleaving_log2(&self) -> u32 {
    //     self.l1_banks_byte_interleaving.ilog2()
    // }

    // // #[inline]
    // #[must_use]
    // pub fn compute_set_bank(&self, addr: address) -> u64 {
    //     log::trace!(
    //         "computing set bank for address {} ({} l1 banks) using hashing function {:?}",
    //         addr,
    //         self.l1_banks,
    //         self.l1_banks_hashing_function
    //     );
    //
    //     // For sector cache, we select one sector per bank (sector interleaving)
    //     // This is what was found in Volta (one sector per bank, sector
    //     // interleaving) otherwise, line interleaving
    //
    //     self.l1_banks_hashing_function.compute_set_index(
    //         addr,
    //         self.l1_banks,
    //         self.l1_banks_byte_interleaving_log2(),
    //         self.l1_banks_log2(),
    //     )
    // }
}

/// `CacheConfig` configures a generic cache
#[derive(Debug)]
pub struct Cache {
    pub kind: CacheKind,
    pub num_sets: usize,
    pub line_size: u32,
    pub associativity: usize,

    pub replacement_policy: cache::config::ReplacementPolicy,
    pub write_policy: cache::config::WritePolicy,
    pub allocate_policy: cache::config::AllocatePolicy,
    pub write_allocate_policy: cache::config::WriteAllocatePolicy,
    // pub set_index_function: CacheSetIndexFunc,
    // pub set_index_function: Box<dyn cache::set_index::SetIndexer>,
    pub mshr_kind: mshr::Kind,
    pub mshr_entries: usize,
    pub mshr_max_merge: usize,

    pub miss_queue_size: usize,
    pub result_fifo_entries: Option<usize>,

    /// L1D write ratio
    pub l1_cache_write_ratio_percent: usize, // 0

    // private (should be used with accessor methods)
    pub data_port_width: Option<usize>,
    // pub accelsim_compat: bool,
}

impl std::fmt::Display for Cache {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let size = human_bytes::human_bytes(self.total_bytes() as f64);
        write!(
            f,
            "{size} ({} set, {}-way, {} byte line)",
            self.num_sets, self.associativity, self.line_size
        )
    }
}

// pub static MAX_DEFAULT_CACHE_SIZE_MULTIPLIER: u8 = 4;

/// TODO: use a builder here so we can fill in the remaining values
/// and do the validation as found below:
impl Cache {
    /// The width if the port to the data array.
    ///
    /// todo: this can be replaced with the builder?
    // #[inline]
    #[must_use]
    pub fn data_port_width(&self) -> usize {
        // default granularity is line size
        let width = self.data_port_width.unwrap_or(self.line_size as usize);
        debug_assert!(self.line_size as usize % width == 0);
        width
    }

    /// The total size of the cache in bytes.
    // #[inline]
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.total_lines() * self.line_size as usize
    }

    /// Number of lines in total.
    // #[inline]
    #[must_use]
    pub fn total_lines(&self) -> usize {
        self.num_sets * self.associativity
    }

    /// Maximum number of lines.
    // #[inline]
    #[must_use]
    pub fn max_num_lines(&self) -> usize {
        self.max_cache_multiplier() as usize * self.num_sets * self.associativity
    }

    /// this is virtual (possibly different)
    // #[inline]
    #[must_use]
    pub fn max_cache_multiplier(&self) -> u8 {
        1
        // MAX_DEFAULT_CACHE_SIZE_MULTIPLIER
    }

    // // #[inline]
    // #[must_use]
    // pub fn line_size_log2(&self) -> u32 {
    //     self.line_size.ilog2()
    // }

    // // #[inline]
    // #[must_use]
    // pub fn num_sets_log2(&self) -> u32 {
    //     self.num_sets.ilog2()
    // }

    // // #[inline]
    // #[must_use]
    // pub fn sector_size(&self) -> u32 {
    //     mem_sub_partition::SECTOR_SIZE
    // }
    //
    // // #[inline]
    // #[must_use]
    // pub fn sector_size_log2(&self) -> u32 {
    //     mcu::logb2(self.sector_size())
    // }

    // #[inline]
    #[must_use]
    pub fn atom_size(&self) -> u32 {
        if self.kind == CacheKind::Sector {
            mem_sub_partition::SECTOR_SIZE
        } else {
            self.line_size
        }
    }

    // // do not use enabled but options
    // // #[inline]
    // #[must_use]
    // pub fn set_index(&self, addr: address) -> u64 {
    //     self.set_index_function.compute_set_index(
    //         addr,
    //         self.num_sets,
    //         self.line_size_log2(),
    //         self.num_sets_log2(),
    //     )
    //     // hash_function(
    //     //     addr,
    //     //     self.num_sets,
    //     //     self.line_size_log2(),
    //     //     self.num_sets_log2(),
    //     //     self.set_index_function,
    //     // )
    // }

    // #[inline]
    #[must_use]
    pub fn tag(&self, addr: address) -> address {
        // For generality, the tag includes both index and tag.
        // This allows for more complex set index calculations that
        // can result in different indexes mapping to the same set,
        // thus the full tag + index is required to check for hit/miss.
        // Tag is now identical to the block address.

        // return addr >> (m_line_sz_log2+m_nset_log2);
        // return addr & ~(new_addr_type)(m_line_sz - 1);
        addr & !u64::from(self.line_size - 1)
    }

    /// Block address
    // #[inline]
    #[must_use]
    pub fn block_addr(&self, addr: address) -> address {
        addr & !u64::from(self.line_size - 1)
    }

    /// Mshr address
    // #[inline]
    #[must_use]
    pub fn mshr_addr(&self, addr: address) -> address {
        addr & !u64::from(self.line_size - 1)
    }

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

/// DRAM Timing Options
///
/// {nbk:tCCD:tRRD:tRCD:tRAS:tRP:tRC:CL:WL:tCDLR:tWR:nbkgrp:tCCDL:tRTPL}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TimingOptions {
    pub num_banks: usize,
    // pub t_ccd: usize,
    // pub t_rrd: usize,
    // pub t_rcd: usize,
    // pub t_ras: usize,
    // pub t_rp: usize,
    // pub t_rc: usize,
    // pub cl: usize,
    // pub wl: usize,
    // pub t_cdlr: usize,
    // pub t_wr: usize,
    // pub num_bank_groups: usize,
    // pub t_ccdl: usize,
    // pub t_rtpl: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum Parallelization {
    Serial,
    #[cfg(feature = "parallel")]
    Deterministic,
    #[cfg(feature = "parallel")]
    Nondeterministic {
        run_ahead: usize,
        interleave: bool,
    },
}

#[allow(clippy::struct_excessive_bools)]
#[derive(Debug)]
pub struct ClockFrequencies {
    pub core_freq_hz: u64,
    pub interconn_freq_hz: u64,
    pub l2_freq_hz: u64,
    pub dram_freq_hz: u64,
    // derived
    pub core_period: f64,
    pub interconn_period: f64,
    pub l2_period: f64,
    pub dram_period: f64,
}

pub struct ClockFrequenciesBuilder {
    pub core_freq_hz: u64,
    pub interconn_freq_hz: u64,
    pub l2_freq_hz: u64,
    pub dram_freq_hz: u64,
}

impl ClockFrequenciesBuilder {
    pub fn build(self) -> ClockFrequencies {
        ClockFrequencies {
            core_freq_hz: self.core_freq_hz,
            interconn_freq_hz: self.interconn_freq_hz,
            dram_freq_hz: self.dram_freq_hz,
            l2_freq_hz: self.l2_freq_hz,
            core_period: 1f64 / self.core_freq_hz as f64,
            interconn_period: 1f64 / self.interconn_freq_hz as f64,
            dram_period: 1f64 / self.dram_freq_hz as f64,
            l2_period: 1f64 / self.l2_freq_hz as f64,
        }
    }
}

#[allow(clippy::struct_excessive_bools)]
#[derive(Debug)]
pub struct GPU {
    /// Log after cycle
    pub log_after_cycle: Option<u64>,
    /// Accelsim compatibility mode.
    ///
    /// This must be set when running lockstep tests.
    /// In other cases, accelsim compat mode is not recommended.
    pub accelsim_compat: bool,
    /// Simulation method
    pub parallelization: Parallelization,
    /// Simulate memory instructions only
    pub memory_only: bool,
    /// Simulate different clock domains of core, memory, and interconnect subsystems.
    pub simulate_clock_domains: bool,
    /// Simulation threads
    ///
    /// If no value is provided, the number of physical cores is used.
    pub simulation_threads: Option<usize>,
    /// Deadlock check
    pub deadlock_check: bool,
    /// Deadlock check
    pub l2_prefetch_percent: Option<f32>,

    pub memory_controller_unit: std::sync::OnceLock<mcu::MemoryControllerUnit>,
    /// The SM number to pass to ptxas when getting register usage for
    /// computing GPU occupancy.
    pub occupancy_sm_number: usize,
    /// num threads per shader core pipeline
    pub max_threads_per_core: usize,
    /// shader core pipeline warp size
    pub warp_size: usize,
    /// Clock frequencies
    pub clock_frequencies: ClockFrequencies,
    /// per-shader read-only L1 texture cache config
    pub tex_cache_l1: Option<Arc<Cache>>,
    /// per-shader read-only L1 constant memory cache config
    pub const_cache_l1: Option<Arc<Cache>>,
    /// shader L1 instruction cache config
    pub inst_cache_l1: Option<Arc<Cache>>,
    /// per-shader L1 data cache config
    pub data_cache_l1: Option<Arc<L1DCache>>,
    /// unified banked L2 data cache config
    pub data_cache_l2: Option<Arc<L2DCache>>,

    /// Shared memory latency
    pub shared_memory_latency: usize,
    /// SP unit max latency
    pub max_sp_latency: usize,
    /// Int unit max latency
    pub max_int_latency: usize,
    /// SFU unit max latency
    pub max_sfu_latency: usize,
    /// DP unit max latency
    pub max_dp_latency: usize,
    /// implements -Xptxas -dlcm=cg (cache global, skipping l1 data cache), default=no skip
    pub global_mem_skip_l1_data_cache: bool,
    /// enable perfect memory mode (no cache miss)
    pub perfect_mem: bool,
    // -gpgpu_cache:dl1PrefL1                 none # per-shader L1 data cache config  {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq> | none}
    // -gpgpu_cache:dl1PrefShared                 none # per-shader L1 data cache config  {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq> | none}
    /// Number of registers per shader core.
    /// Limits number of concurrent CTAs. (default 8192)
    pub shader_registers: usize, // 65536
    /// Maximum number of registers per CTA. (default 8192)
    pub registers_per_block: usize, //  8192
    pub ignore_resources_limitation: bool, // 0
    /// Maximum number of concurrent CTAs in shader (default 32)
    pub max_concurrent_blocks_per_core: usize, // 32
    /// Maximum number of named barriers per CTA (default 16)
    pub max_barriers_per_block: usize, // 16
    /// number of processing clusters
    pub num_simt_clusters: usize, //  20
    /// number of simd cores per cluster
    pub num_cores_per_simt_cluster: usize, // 1
    /// number of packets in ejection buffer
    pub num_cluster_ejection_buffer_size: usize, // 8
    /// number of response packets in ld/st unit ejection buffer
    pub num_ldst_response_buffer_size: usize, //  2
    /// Size of shared memory per thread block or CTA (default 48kB)
    pub shared_memory_per_block: usize, // 49152
    /// Size of shared memory per shader core (default 16kB)
    pub shared_memory_size: u32, // 98304
    /// Option list of shared memory sizes
    pub shared_memory_option: bool, // 0
    /// Size of unified data cache(L1D + shared memory) in KB
    pub unified_l1_data_cache_size: bool, //0
    /// adaptive_cache_config
    pub adaptive_cache_config: bool, // 0
    /// Option list of shared memory sizes
    pub shared_memory_sizes: Vec<u32>, // 0
    // Size of shared memory per shader core (default 16kB)
    // shared_memory_size_default: usize, // 16384
    /// Size of shared memory per shader core (default 16kB)
    pub shared_memory_size_pref_l1: usize, // 16384
    /// Size of shared memory per shader core (default 16kB)
    pub shared_memory_size_pref_shared: usize, // 16384
    /// Number of banks in the shared memory in each shader core (default 16)
    pub shared_memory_num_banks: usize, // 32
    /// Limit shared memory to do one broadcast per cycle (default on)
    pub shared_memory_limited_broadcast: bool, // 0
    /// Number of portions a warp is divided into for shared memory bank conflict check
    pub shared_memory_warp_parts: usize, // 1
    /// The number of memory transactions allowed per core cycle
    pub mem_unit_ports: usize, // 1
    /// Specify which shader core to collect the warp size distribution from
    pub warp_distro_shader_core: i32, // -1
    /// Specify which shader core to collect the warp issue distribution from
    pub warp_issue_shader_core: i32, // 0
    /// Mapping from local memory space address to simulated GPU physical address space
    pub local_mem_map: bool, // 1
    /// Number of register banks (default = 8)
    pub num_reg_banks: usize, // 32
    /// Use warp ID in mapping registers to banks (default = off)
    pub reg_bank_use_warp_id: bool, // 0
    /// Sub Core Volta/Pascal model (default = off)
    pub sub_core_model: bool, // 0
    /// enable_specialized_operand_collector
    pub enable_specialized_operand_collector: bool, // true
    /// number of collector units (default = 4)
    pub operand_collector_num_units_sp: usize, // 4
    /// number of collector units (default = 0)
    pub operand_collector_num_units_dp: usize, // 0
    /// number of collector units (default = 4)
    pub operand_collector_num_units_sfu: usize, // 4
    /// number of collector units (default = 0)
    pub operand_collector_num_units_int: usize, // 0
    /// number of collector units (default = 4)
    pub operand_collector_num_units_tensor_core: usize, // 4
    /// number of collector units (default = 2)
    pub operand_collector_num_units_mem: usize, // 2
    /// number of collector units (default = 0)
    pub operand_collector_num_units_gen: usize, // 0
    /// number of collector unit in ports (default = 1)
    pub operand_collector_num_in_ports_sp: usize, // 1
    /// number of collector unit in ports (default = 0)
    pub operand_collector_num_in_ports_dp: usize, // 0
    /// number of collector unit in ports (default = 1)
    pub operand_collector_num_in_ports_sfu: usize, // 1
    /// number of collector unit in ports (default = 0)
    pub operand_collector_num_in_ports_int: usize, // 0
    /// number of collector unit in ports (default = 1)
    pub operand_collector_num_in_ports_tensor_core: usize, // 1
    /// number of collector unit in ports (default = 1)
    pub operand_collector_num_in_ports_mem: usize, // 1
    /// number of collector unit in ports (default = 0)
    pub operand_collector_num_in_ports_gen: usize, // 0
    /// number of collector unit in ports (default = 1)
    pub operand_collector_num_out_ports_sp: usize, // 1
    /// number of collector unit in ports (default = 0)
    pub operand_collector_num_out_ports_dp: usize, // 0
    /// number of collector unit in ports (default = 1)
    pub operand_collector_num_out_ports_sfu: usize, // 1
    /// number of collector unit in ports (default = 0)
    pub operand_collector_num_out_ports_int: usize, // 0
    /// number of collector unit in ports (default = 1)
    pub operand_collector_num_out_ports_tensor_core: usize, // 1
    /// number of collector unit in ports (default = 1)
    pub operand_collector_num_out_ports_mem: usize, // 1
    /// number of collector unit in ports (default = 0)
    pub operand_collector_num_out_ports_gen: usize, // 0
    /// Coalescing arch (GT200 = 13, Fermi = 20)
    pub coalescing_arch: Architecture, // 13
    /// Number of warp schedulers per core
    pub num_schedulers_per_core: usize, // 2
    /// Max number of instructions that can be issued per warp in one cycle by scheduler (either 1 or 2)
    pub max_instruction_issue_per_warp: usize, // 2
    /// should dual issue use two different execution unit resources
    pub dual_issue_only_to_different_exec_units: bool, // 1
    /// Select the simulation order of cores in a cluster
    pub simt_core_sim_order: SchedulingOrder, // 1
    // Pipeline widths
    //
    // ID_OC_SP,ID_OC_DP,ID_OC_INT,ID_OC_SFU,ID_OC_MEM,OC_EX_SP,OC_EX_DP,
    // OC_EX_INT,OC_EX_SFU,OC_EX_MEM,EX_WB,ID_OC_TENSOR_CORE,OC_EX_TENSOR_CORE
    //
    pub pipeline_widths: HashMap<PipelineStage, usize>, // 4,0,0,1,1,4,0,0,1,1,6
    /// Number of SP units
    pub num_sp_units: usize, //
    /// Number of DP units
    pub num_dp_units: usize, // 0
    /// Number of INT units
    pub num_int_units: usize, // 0

    /// Number of SF units
    pub num_sfu_units: usize, // 1
    /// Number of tensor cores available
    pub num_tensor_core_avail: usize, // 0
    /// Number of tensor_core units
    pub num_tensor_core_units: usize, // 0
    /// Scheduler configuration: < lrr | gto | two_level_active > If two_level_active:<num_active_warps>:<inner_prioritization>:<outer_prioritization>For complete list of prioritization values see shader.h enum scheduler_prioritization_typeDefault: gto
    pub scheduler: CoreSchedulerKind, // gto
    /// Support concurrent kernels on a SM (default = disabled)
    pub concurrent_kernel_sm: bool, // 0
    /// perfect inst and const cache mode, so all inst and const hits in the cache(default = disabled)
    pub perfect_inst_const_cache: bool, // 0
    /// the number of fetched intruction per warp each cycle
    pub inst_fetch_throughput: usize, // 1
    /// the number ports of the register file
    pub reg_file_port_throughput: usize, // 1
    /// Fill the L2 cache on memcpy
    pub fill_l2_on_memcopy: bool, // true
    /// simple_dram_model with fixed latency and BW
    pub simple_dram_model: bool, // 0
    /// DRAM scheduler kind. 0 = fifo, 1 = FR-FCFS (default)
    pub dram_scheduler: DRAMSchedulerKind, // 1
    /// DRAM partition queue
    pub dram_partition_queue_interconn_to_l2: usize, // 8
    pub dram_partition_queue_l2_to_dram: usize,      // 8
    pub dram_partition_queue_dram_to_l2: usize,      // 8
    pub dram_partition_queue_l2_to_interconn: usize, // 8
    /// use a ideal L2 cache that always hit
    pub ideal_l2: bool, // 0
    /// L2 cache used for texture only
    pub data_cache_l2_texture_only: bool, // 0
    /// number of memory modules (e.g. memory controllers) in gpu
    pub num_memory_controllers: usize, // 8
    /// number of memory subpartition in each memory module
    pub num_sub_partitions_per_memory_controller: usize, // 2
    /// number of memory chips per memory controller
    pub num_dram_chips_per_memory_controller: usize, // 1
    /// track and display latency statistics 0x2 enables MC, 0x4 enables queue logs
    // memory_latency_stat: usize, // 14
    /// DRAM scheduler queue size 0 = unlimited (default); # entries per chip
    pub dram_frfcfs_sched_queue_size: usize, // 64
    /// 0 = unlimited (default); # entries per chip
    pub dram_return_queue_size: usize, // 116
    /// default = 4 bytes (8 bytes per cycle at DDR)
    pub dram_buswidth: usize, // 4
    /// Burst length of each DRAM request (default = 4 data bus cycle)
    pub dram_burst_length: usize, // 8
    /// Frequency ratio between DRAM data bus and command bus (default = 2 times, i.e. DDR)
    pub dram_data_command_freq_ratio: usize, // 4
    /// DRAM timing parameters =
    /// {nbk:tCCD:tRRD:tRCD:tRAS:tRP:tRC:CL:WL:tCDLR:tWR:nbkgrp:tCCDL:tRTPL}
    /// nbk=16:CCD=2:RRD=6:RCD=12:RAS=28:RP=12:RC=40: CL=12:WL=4:CDLR=5:WR=12:nbkgrp=1:CCDL=0:RTPL=0
    pub dram_timing_options: TimingOptions,
    /// ROP queue latency (default 85)
    pub l2_rop_latency: u64, // 220
    /// DRAM latency (default 30)
    pub dram_latency: usize, // 100
    /// dual_bus_interface (default = 0)
    pub dram_dual_bus_interface: bool, // 0
    /// dram_bnk_indexing_policy
    pub dram_bank_indexing_policy: DRAMBankIndexPolicy, // 0
    /// dram_bnkgrp_indexing_policy
    pub dram_bank_group_indexing_policy: DRAMBankGroupIndexPolicy, // 0
    /// Seperate_Write_Queue_Enable
    pub dram_seperate_write_queue_enable: bool, // 0
    /// write_Queue_Size
    /// dram_frfcfs_write_queue_size:high_watermark:low_watermark
    pub dram_frfcfs_write_queue_size: usize, // 32:28:16
    /// elimnate_rw_turnaround i.e set tWTR and tRTW = 0
    pub dram_elimnate_rw_turnaround: bool, // 0
    /// mapping memory address to dram model
    /// {dramid@<start bit>;<memory address map>}
    pub memory_addr_mapping: Option<String>, // dramid@8;00000000.00000000.00000000.00000000.0000RRRR.RRRRRRRR.RBBBCCCC.BCCSSSSS
    /// run sweep test to check address mapping for aliased address
    // memory_addr_test: bool, // 0
    /// 0 = old addressing mask, 1 = new addressing mask, 2 = new add. mask + flipped bank sel and chip sel bits
    pub memory_address_mask: MemoryAddressingMask, // 1
    /// 0 = consecutive (no indexing), 1 = bitwise xoring
    /// 2 = IPoly, 3 = pae, 4 = random, 5 = custom"
    pub memory_partition_indexing: MemoryPartitionIndexingScheme, // 0
    /// Major compute capability version number
    pub compute_capability_major: usize, // 7
    /// Minor compute capability version number
    pub compute_capability_minor: usize, // 0
    /// Flush L1 cache at the end of each kernel call
    pub flush_l1_cache: bool, // 0
    /// Flush L2 cache at the end of each kernel call
    pub flush_l2_cache: bool, // 0
    /// maximum kernels that can run concurrently on GPU.
    ///
    /// Set this value according to max resident grids for your
    /// compute capability.
    pub max_concurrent_kernels: usize, // 32
    /// Kernel launch latency in cycles
    pub kernel_launch_latency: usize,
    /// Block launch latency in cycles
    pub block_launch_latency: usize,
    /// Opcode latencies and initiation for integers in trace driven mode (latency,initiation)
    pub trace_opcode_latency_initiation_int: (usize, usize), // 4, 1
    /// Opcode latencies and initiation for sp in trace driven mode (latency,initiation)
    pub trace_opcode_latency_initiation_sp: (usize, usize), // 4, 1
    /// Opcode latencies and initiation for dp in trace driven mode (latency,initiation)
    pub trace_opcode_latency_initiation_dp: (usize, usize), // 4, 1
    /// Opcode latencies and initiation for sfu in trace driven mode (latency,initiation)
    pub trace_opcode_latency_initiation_sfu: (usize, usize), // 4, 1
    /// Opcode latencies and initiation for tensor in trace driven mode (latency,initiation)
    pub trace_opcode_latency_initiation_tensor: (usize, usize), // 4, 1
}

pub static WORD_SIZE: address = 4;

#[must_use]
pub fn pad_to_multiple(n: usize, k: usize) -> usize {
    let rem = n % k;
    if rem == 0 {
        n
    } else {
        ((n / k) + 1) * k
    }
}
impl GPU {
    #[must_use]
    pub fn is_parallel_simulation(&self) -> bool {
        self.parallelization != Parallelization::Serial
    }

    pub fn shared_mem_bank(&self, addr: address) -> address {
        let num_banks = self.shared_memory_num_banks as u64;
        (addr / WORD_SIZE) % num_banks
    }

    pub fn max_warps_per_core(&self) -> usize {
        self.max_threads_per_core / self.warp_size
    }

    pub fn total_cores(&self) -> usize {
        self.num_simt_clusters * self.num_cores_per_simt_cluster
    }

    pub fn global_core_id_to_cluster_id(&self, core_id: usize) -> usize {
        core_id / self.num_cores_per_simt_cluster
    }

    pub fn global_core_id_to_core_id(&self, core_id: usize) -> usize {
        core_id % self.num_cores_per_simt_cluster
    }

    pub fn global_core_id(&self, cluster_id: usize, core_id: usize) -> usize {
        cluster_id * self.num_cores_per_simt_cluster + core_id
    }

    pub fn mem_id_to_device_id(&self, mem_id: usize) -> usize {
        mem_id + self.num_simt_clusters
    }

    pub fn threads_per_block_padded(&self, kernel: &Kernel) -> usize {
        let threads_per_block = kernel.threads_per_block();
        pad_to_multiple(threads_per_block, self.warp_size)
    }

    /// Number of bytes transferred per read or write command.
    pub fn dram_atom_size(&self) -> usize {
        // burst length x bus width x # chips per partition
        self.dram_burst_length * self.dram_buswidth * self.num_dram_chips_per_memory_controller
    }

    /// Compute maximum number of blocks that a kernel can run
    ///
    /// Depends on the following constraints:
    /// -
    pub fn max_blocks(&self, kernel: &Kernel) -> eyre::Result<usize> {
        let threads_per_block = kernel.threads_per_block();
        let threads_per_block = pad_to_multiple(threads_per_block, self.warp_size);
        // limit by n_threads/shader
        let by_thread_limit = self.max_threads_per_core / threads_per_block;

        // limit by shmem/shader
        let by_shared_mem_limit = if kernel.config.shared_mem_bytes > 0 {
            Some(self.shared_memory_size as usize / kernel.config.shared_mem_bytes as usize)
        } else {
            None
        };

        // limit by register count, rounded up to multiple of 4.
        let by_register_limit = if kernel.config.num_registers > 0 {
            Some(
                self.shader_registers
                    / (threads_per_block * ((kernel.config.num_registers + 3) & !3) as usize),
            )
        } else {
            None
        };

        // limit by CTA
        // let _by_block_limit = self.max_concurrent_blocks_per_core;

        // find the minimum
        let mut limit = [
            Some(by_thread_limit),
            by_shared_mem_limit,
            by_register_limit,
        ]
        .into_iter()
        .flatten()
        .min()
        .unwrap_or(usize::MAX);
        // result = gs_min2(result, result_shmem);
        // result = gs_min2(result, result_regs);
        // result = gs_min2(result, result_cta);

        // max blocks per shader is limited by number of blocks
        // if not enough to keep all cores busy
        if kernel.num_blocks() < (limit * self.total_cores()) {
            limit = kernel.num_blocks() / self.total_cores();
            if kernel.num_blocks() % self.total_cores() != 0 {
                limit += 1;
            }
        }
        if limit < 1 {
            return Err(eyre::eyre!(
                "kernel requires more resources than shader has"
            ));
        }

        if self.adaptive_cache_config {
            // more info about adaptive cache, see
            // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
            let total_shared_mem = kernel.config.shared_mem_bytes as usize * limit;
            if let Some(size) = self.shared_memory_sizes.last() {
                assert!(total_shared_mem <= (*size as usize));
            }

            unimplemented!("adaptive cache config")

            // Unified cache config is in KB. Converting to B
            // unsigned total_unified = m_L1D_config.m_unified_cache_size * 1024;
            //
            // bool l1d_configured = false;
            // unsigned max_assoc = m_L1D_config.get_max_assoc();
            //
            // for (std::vector<unsigned>::const_iterator it = shmem_opt_list.begin();
            //      it < shmem_opt_list.end(); it++) {
            //   if (total_shmem <= *it) {
            //     float l1_ratio = 1 - ((float)*(it) / total_unified);
            //     // make sure the ratio is between 0 and 1
            //     assert(0 <= l1_ratio && l1_ratio <= 1);
            //     // round to nearest instead of round down
            //     m_L1D_config.set_assoc(max_assoc * l1_ratio + 0.5f);
            //     l1d_configured = true;
            //     break;
            //   }
            // }
            //
            // assert(l1d_configured && "no shared memory option found");

            // if (m_L1D_config.is_streaming()) {
            //       // for streaming cache, if the whole memory is allocated
            //       // to the L1 cache, then make the allocation to be on_MISS
            //       // otherwise, make it ON_FILL to eliminate line allocation fails
            //       // i.e. MSHR throughput is the same, independent on the L1 cache
            //       // size/associativity
            //       if (total_shmem == 0) {
            //         m_L1D_config.set_allocation_policy(ON_MISS);
            //
            //         if (gpgpu_ctx->accelsim_compat_mode) {
            //           printf("GPGPU-Sim: Reconfigure L1 allocation to ON_MISS\n");
            //         }
            //       } else {
            //         m_L1D_config.set_allocation_policy(ON_FILL);
            //         if (gpgpu_ctx->accelsim_compat_mode) {
            //           printf("GPGPU-Sim: Reconfigure L1 allocation to ON_FILL\n");
            //         }
            //       }
            //     }
            //     if (gpgpu_ctx->accelsim_compat_mode) {
            //       printf("GPGPU-Sim: Reconfigure L1 cache to %uKB\n",
            //              m_L1D_config.get_total_size_inKB());
            //     }
        }

        Ok(limit)
    }

    pub fn get_latencies(&self, arch_op_category: opcodes::ArchOp) -> (usize, usize) {
        use opcodes::ArchOp;

        let mut initiation_interval = 1;
        let mut latency = 1;

        match arch_op_category {
            ArchOp::ALU_OP
            | ArchOp::INT_OP
            | ArchOp::BRANCH_OP
            | ArchOp::CALL_OPS
            | ArchOp::RET_OPS => {
                // integer units
                (latency, initiation_interval) = self.trace_opcode_latency_initiation_int;
            }
            ArchOp::SP_OP => {
                // single precision units
                (latency, initiation_interval) = self.trace_opcode_latency_initiation_sp;
            }
            ArchOp::DP_OP => {
                // double precision units
                (latency, initiation_interval) = self.trace_opcode_latency_initiation_dp;
            }
            ArchOp::SFU_OP => {
                // special function units
                (latency, initiation_interval) = self.trace_opcode_latency_initiation_sfu;
            }
            ArchOp::TENSOR_CORE_OP => {
                (latency, initiation_interval) = self.trace_opcode_latency_initiation_tensor;
            }
            _ => {}
        }

        // ignore special function units for now
        // if (category >= SPEC_UNIT_START_ID) {
        //   unsigned spec_id = category - SPEC_UNIT_START_ID;
        //   assert(spec_id >= 0 && spec_id < SPECIALIZED_UNIT_NUM);
        //   latency = specialized_unit_latency[spec_id];
        //   initiation_interval = specialized_unit_initiation[spec_id];
        // }

        (latency, initiation_interval)
    }
}

/// Cache set indexing function kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CacheSetIndexFunc {
    FERMI_HASH_SET_FUNCTION, // H
    HASH_IPOLY_FUNCTION,     // P
    LINEAR_SET_FUNCTION,     // L
    BITWISE_XORING_FUNCTION, // X
}

/// Memory partition indexing scheme.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub enum MemoryPartitionIndexingScheme {
    Consecutive = 0, // no indexing
    BitwiseXor = 1,
    IPoly = 2,
    PAE = 3,
    Random = 4,
    // Custom = 2,
}

/// DRAM bank group indexing policy.
///
/// 0 = take higher bits, 1 = take lower bits
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DRAMBankGroupIndexPolicy {
    HigherBits = 0,
    LowerBits = 1,
}

/// DRAM bank indexing policy.
///
/// 0 = normal indexing, 1 = Xoring with the higher bits
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DRAMBankIndexPolicy {
    Normal = 0,
    Xor = 1,
}

/// Scheduler kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SchedulerKind {
    LRR = 0,
    TwoLevelActive = 1,
    GTO = 2,
    RRR = 3,
    Old = 4,
    OldestFirst = 5,
    WarpLimiting = 6,
}

/// DRAM Scheduler policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DRAMSchedulerKind {
    FIFO = 0,
    FrFcfs = 1,
}

/// Core Scheduler policy.
///
/// If `two_level_active`:
/// <`num_active_warps>:<inner_prioritization>:<outer_prioritization`>
///
/// For complete list of prioritization values see shader.h.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CoreSchedulerKind {
    LRR,
    GTO,
    TwoLevelActive,
}

/// GPU microarchitecture generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Architecture {
    GT200 = 13,
    Fermi = 20,
    Pascal = 61,
}

/// Scheduling order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SchedulingOrder {
    Fix = 0,
    RoundRobin = 1,
}

impl GPU {
    // pub fn parse() -> eyre::Result<Self> {
    //     let adaptive_cache_config = false;
    //     let shared_memory_sizes_string = "0";
    //     let _shared_memory_sizes: Vec<u32> = if adaptive_cache_config {
    //         let sizes: Result<Vec<u32>, _> = shared_memory_sizes_string
    //             .split(',')
    //             .map(str::parse)
    //             .collect();
    //         let mut sizes: Vec<_> = sizes?.into_iter().map(|size| size * 1024).collect();
    //         sizes.sort_unstable();
    //         sizes
    //     } else {
    //         vec![]
    //     };
    //     Ok(Self::default())
    // }

    pub fn total_sub_partitions(&self) -> usize {
        self.num_memory_controllers * self.num_sub_partitions_per_memory_controller
    }
}

impl Default for GPU {
    fn default() -> Self {
        Self {
            log_after_cycle: None,
            parallelization: Parallelization::Serial,
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
            clock_frequencies: ClockFrequenciesBuilder {
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
            tex_cache_l1: Some(Arc::new(Cache {
                // accelsim_compat,
                kind: CacheKind::Normal,
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
                result_fifo_entries: Some(2),
                l1_cache_write_ratio_percent: 0,
                data_port_width: None,
            })),
            // N:128:64:2,L:R:f:N:L,A:2:64,4
            // {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>}
            const_cache_l1: Some(Arc::new(Cache {
                // accelsim_compat,
                kind: CacheKind::Normal,
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
                result_fifo_entries: None,
                l1_cache_write_ratio_percent: 0,
                data_port_width: None,
            })),
            // N:8:128:4,L:R:f:N:L,A:2:48,4
            // {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>}
            inst_cache_l1: Some(Arc::new(Cache {
                // accelsim_compat,
                kind: CacheKind::Normal,
                num_sets: 8,
                line_size: 128,
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
                result_fifo_entries: None,
                l1_cache_write_ratio_percent: 0,
                data_port_width: None,
            })),
            // N:64:128:6,L:L:m:N:H,A:128:8,8
            // {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq> | none}
            // l1 hit latency = 135.70804974
            // l2 hit latency = 274.3884858
            // l2 miss latency = 474.04434122
            data_cache_l1: Some(Arc::new(L1DCache {
                l1_latency: 1,
                l1_hit_latency: 81,
                // l1_banks_hashing_function: CacheSetIndexFunc::LINEAR_SET_FUNCTION,
                // l1_banks_hashing_function: Box::<cache::set_index::linear::SetIndex>::default(),
                l1_banks_byte_interleaving: 32,
                l1_banks: 1,
                inner: Arc::new(Cache {
                    // accelsim_compat,
                    kind: CacheKind::Sector,
                    // kind: CacheKind::Normal,
                    num_sets: 4, // 64,
                    line_size: 128,
                    associativity: 48, // 6,
                    replacement_policy: cache::config::ReplacementPolicy::LRU,
                    write_policy: cache::config::WritePolicy::LOCAL_WB_GLOBAL_WT,
                    allocate_policy: cache::config::AllocatePolicy::ON_MISS,
                    write_allocate_policy: cache::config::WriteAllocatePolicy::NO_WRITE_ALLOCATE,
                    // set_index_function: CacheSetIndexFunc::FERMI_HASH_SET_FUNCTION,
                    // set_index_function: Box::<cache::set_index::fermi::SetIndex>::default(),
                    mshr_kind: mshr::Kind::ASSOC,
                    // mshr_kind: mshr::Kind::SECTOR_ASSOC,
                    mshr_entries: 128,
                    mshr_max_merge: 8,
                    miss_queue_size: 4,
                    result_fifo_entries: None,
                    l1_cache_write_ratio_percent: 0,
                    data_port_width: None,
                }),
            })),
            // N:64:128:16,L:B:m:W:L,A:1024:1024,4:0,32
            // {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>}
            data_cache_l2: Some(Arc::new(L2DCache {
                inner: Arc::new(Cache {
                    // accelsim_compat,
                    kind: CacheKind::Sector,
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
                    result_fifo_entries: None, // 0 is none?
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
            coalescing_arch: Architecture::Pascal,
            num_schedulers_per_core: 4,
            max_instruction_issue_per_warp: 2,
            dual_issue_only_to_different_exec_units: true,
            simt_core_sim_order: SchedulingOrder::RoundRobin,
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
            scheduler: CoreSchedulerKind::GTO,
            concurrent_kernel_sm: false,
            perfect_inst_const_cache: true,
            inst_fetch_throughput: 1,
            reg_file_port_throughput: 2, // 1 for GTX1080
            fill_l2_on_memcopy: true,
            simple_dram_model: false,
            dram_scheduler: DRAMSchedulerKind::FrFcfs,
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
            dram_timing_options: TimingOptions { num_banks: 16 },
            // this is the l2 latency 216 L2 latency
            // l2_rop_latency: 1,
            // dram_latency: 1,
            l2_rop_latency: 210, // was 120
            dram_latency: 190,   // was 100
            dram_dual_bus_interface: false,
            dram_bank_indexing_policy: DRAMBankIndexPolicy::Normal,
            dram_bank_group_indexing_policy: DRAMBankGroupIndexPolicy::LowerBits,
            dram_seperate_write_queue_enable: false,
            dram_frfcfs_write_queue_size: 32, // 32:28:16
            dram_elimnate_rw_turnaround: false,
            memory_addr_mapping: Some(
                "dramid@8;00000000.00000000.00000000.00000000.0000RRRR.RRRRRRRR.RBBBCCCC.BCCSSSSS"
                    .to_string(),
            ),
            memory_address_mask: MemoryAddressingMask::New, // 1
            memory_partition_indexing: MemoryPartitionIndexingScheme::Consecutive,
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
            trace_opcode_latency_initiation_int: (4, 1),
            trace_opcode_latency_initiation_sp: (4, 1),
            trace_opcode_latency_initiation_dp: (20, 8), // (4, 1)
            trace_opcode_latency_initiation_sfu: (20, 4), // (4, 1)
            trace_opcode_latency_initiation_tensor: (4, 1),
        }
    }
}

#[cfg(test)]
mod tests {
    use playground::bindings;
    // use pretty_assertions_sorted as diff;
    use std::ffi;
    use utils::diff;

    fn parse_cache_config(config: &str) -> bindings::CacheConfig {
        use bindings::parse_cache_config as parse;

        unsafe { parse(config.as_ptr().cast()) }
    }

    #[test]
    fn test_parse_gtx1080_data_l1_cache_config() {
        diff::assert_eq!(
            have: parse_cache_config("N:64:128:6,L:L:m:N:H,A:128:8,8"),
            want: bindings::CacheConfig {
                ct: 'N' as ffi::c_char,
                m_nset: 64,
                m_line_sz: 128,
                m_assoc: 6,
                rp: 'L' as ffi::c_char,
                wp: 'L' as ffi::c_char,
                ap: 'm' as ffi::c_char,
                wap: 'N' as ffi::c_char,
                sif: 'H' as ffi::c_char,
                mshr_type: 'A' as ffi::c_char,
                m_mshr_entries: 128,
                m_mshr_max_merge: 8,
                m_miss_queue_size: 8,
                m_result_fifo_entries: 0,
                m_data_port_width: 0,
            },
        );
    }

    #[test]
    fn test_parse_gtx1080_tex_l1_cache_config() {
        diff::assert_eq!(
            have: parse_cache_config("N:16:128:24,L:R:m:N:L,F:128:4,128:2"),
            want: bindings::CacheConfig {
                ct: 'N' as ffi::c_char,
                m_nset: 16,
                m_line_sz: 128,
                m_assoc: 24,
                rp: 'L' as ffi::c_char,
                wp: 'R' as ffi::c_char,
                ap: 'm' as ffi::c_char,
                wap: 'N' as ffi::c_char,
                sif: 'L' as ffi::c_char,
                mshr_type: 'F' as ffi::c_char,
                m_mshr_entries: 128,
                m_mshr_max_merge: 4,
                m_miss_queue_size: 128,
                m_result_fifo_entries: 2,
                m_data_port_width: 0,
            },
        );
    }

    #[test]
    fn test_parse_gtx1080_inst_l1_cache_config() {
        diff::assert_eq!(
            have: parse_cache_config("N:8:128:4,L:R:f:N:L,A:2:48,4"),
            want: bindings::CacheConfig {
                ct: 'N' as ffi::c_char,
                m_nset: 8,
                m_line_sz: 128,
                m_assoc: 4,
                rp: 'L' as ffi::c_char,
                wp: 'R' as ffi::c_char,
                ap: 'f' as ffi::c_char,
                wap: 'N' as ffi::c_char,
                sif: 'L' as ffi::c_char,
                mshr_type: 'A' as ffi::c_char,
                m_mshr_entries: 2,
                m_mshr_max_merge: 48,
                m_miss_queue_size: 4,
                m_result_fifo_entries: 0,
                m_data_port_width: 0,
            }
        );
    }

    #[test]
    fn test_parse_gtx1080_const_l1_cache_config() {
        diff::assert_eq!(
            have: parse_cache_config("N:128:64:2,L:R:f:N:L,A:2:64,4"),
            want: bindings::CacheConfig {
                ct: 'N' as ffi::c_char,
                m_nset: 128,
                m_line_sz: 64,
                m_assoc: 2,
                rp: 'L' as ffi::c_char,
                wp: 'R' as ffi::c_char,
                ap: 'f' as ffi::c_char,
                wap: 'N' as ffi::c_char,
                sif: 'L' as ffi::c_char,
                mshr_type: 'A' as ffi::c_char,
                m_mshr_entries: 2,
                m_mshr_max_merge: 64,
                m_miss_queue_size: 4,
                m_result_fifo_entries: 0,
                m_data_port_width: 0,
            }
        );
    }

    #[test]
    fn test_parse_gtx1080_data_l2_cache_config() {
        diff::assert_eq!(
            have: parse_cache_config("N:64:128:16,L:B:m:W:L,A:1024:1024,4:0,32"),
            want: bindings::CacheConfig {
                ct: 'N' as ffi::c_char,
                m_nset: 64,
                m_line_sz: 128,
                m_assoc: 16,
                rp: 'L' as ffi::c_char,
                wp: 'B' as ffi::c_char,
                ap: 'm' as ffi::c_char,
                wap: 'W' as ffi::c_char,
                sif: 'L' as ffi::c_char,
                mshr_type: 'A' as ffi::c_char,
                m_mshr_entries: 1024,
                m_mshr_max_merge: 1024,
                m_miss_queue_size: 4,
                m_result_fifo_entries: 0,
                m_data_port_width: 32,
            }
        );
    }

    #[test]
    fn test_l1i_block_addr() {
        let config = super::GPU::default();
        let l1i_cache_config = config.inst_cache_l1.unwrap();
        assert_eq!(l1i_cache_config.block_addr(4_026_531_848), 4_026_531_840);
    }

    #[test]
    fn test_l2d_block_addr() {
        let config = super::GPU::default();
        let l2d_cache_config = config.data_cache_l2.unwrap();
        assert_eq!(
            l2d_cache_config.inner.block_addr(34_887_082_112),
            34_887_082_112
        );
    }

    #[test]
    fn test_l1i_mshr_addr() {
        let config = super::GPU::default();
        let l1i_cache_config = config.inst_cache_l1.unwrap();
        assert_eq!(l1i_cache_config.mshr_addr(4_026_531_848), 4_026_531_840);
        assert_eq!(l1i_cache_config.mshr_addr(4_026_531_992), 4_026_531_968);
    }
}

#[derive(Debug, Default, serde::Deserialize)]
pub struct Input {
    #[serde(rename = "mode")]
    pub parallelism_mode: Option<String>,
    #[serde(rename = "threads")]
    pub parallelism_threads: Option<usize>,
    #[serde(rename = "run_ahead")]
    pub parallelism_run_ahead: Option<usize>,
    pub cores_per_cluster: Option<usize>,
    pub num_clusters: Option<usize>,
    pub memory_only: Option<bool>,
}

impl Input {
    pub fn is_baseline(&self, parallel: bool) -> bool {
        let mut is_baseline = true;
        if !parallel {
            is_baseline &= matches!(self.parallelism_mode.as_deref(), Some("serial") | None);
        }
        is_baseline &= matches!(self.cores_per_cluster, Some(1) | None);
        is_baseline &= matches!(self.num_clusters, Some(28) | None);
        // is_baseline &= matches!(self.memory_only, Some(false) | None);
        is_baseline
    }
}

pub fn parse_input(
    values: &indexmap::IndexMap<String, serde_yaml::Value>,
) -> Result<Input, serde_json::Error> {
    let values: serde_json::Value = serde_json::to_value(&values)?;
    let input: Input = serde_json::from_value(values)?;
    Ok(input)
}
