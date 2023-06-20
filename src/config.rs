use super::ported::{addrdec, address, core::PipelineStage, mem_sub_partition, utils, KernelInfo};
use color_eyre::eyre;
use std::collections::HashMap;
use std::sync::Arc;

/// Cache kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CacheKind {
    Normal, // N
    Sector, // S
}

/// A cache replacement policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CacheReplacementPolicy {
    LRU,  // L
    FIFO, // F
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct L1DCacheConfig {
    /// L1 Hit Latency
    pub l1_latency: usize, // 1
    /// l1 banks hashing function
    pub l1_banks_hashing_function: CacheSetIndexFunc, // 0
    /// l1 banks byte interleaving granularity
    pub l1_banks_byte_interleaving: usize, // 32
    /// The number of L1 cache banks
    pub l1_banks: usize, // 1
    /// L1D write ratio
    pub m_wr_percent: usize, // 0
    pub inner: Arc<CacheConfig>,
}

impl L1DCacheConfig {
    #[inline]
    pub fn l1_banks_log2(&self) -> u32 {
        addrdec::logb2(self.l1_banks as u32)
    }

    #[inline]
    pub fn l1_banks_byte_interleaving_log2(&self) -> u32 {
        addrdec::logb2(self.l1_banks_byte_interleaving as u32)
    }

    #[inline]
    pub fn set_bank(&self, addr: address) -> u64 {
        // For sector cache, we select one sector per bank (sector interleaving)
        // This is what was found in Volta (one sector per bank, sector
        // interleaving) otherwise, line interleaving
        hash_function(
            addr,
            self.l1_banks,
            self.l1_banks_byte_interleaving_log2(),
            self.l1_banks_log2(),
            self.l1_banks_hashing_function,
        )
    }
}

/// CacheConfig
///
/// gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.h#565
///
/// // todo: remove the copy stuff, very expensive otherwise
///
/// TODO: Find out what those values are.
/// sscanf(config, "%c:%u:%u:%u,%c:%c:%c:%c:%c,%c:%u:%u,%u:%u,%u", &ct,
/// &m_nset, &m_line_sz, &m_assoc, &rp, &wp, &ap, &wap, &sif,
/// &mshr_type, &m_mshr_entries, &m_mshr_max_merge,
/// &m_miss_queue_size, &m_result_fifo_entries, &m_data_port_width);
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct CacheConfig {
    pub kind: CacheKind,
    pub num_sets: usize,
    pub line_size: u32,
    pub associativity: usize,

    pub replacement_policy: CacheReplacementPolicy,
    pub write_policy: CacheWritePolicy,
    pub allocate_policy: CacheAllocatePolicy,
    pub write_allocate_policy: CacheWriteAllocatePolicy,
    pub set_index_function: CacheSetIndexFunc,

    pub mshr_kind: MshrKind,
    pub mshr_entries: usize,
    pub mshr_max_merge: usize,

    pub miss_queue_size: usize,
    pub result_fifo_entries: Option<usize>,
    data_port_width: Option<usize>,
    // pub disabled: bool,
}

pub static MAX_DEFAULT_CACHE_SIZE_MULTIPLIER: u8 = 4;

/// TODO: use a builder here so we can fill in the remaining values
/// and do the validation as found below:
impl CacheConfig {
    /// The width if the port to the data array.
    ///
    /// todo: this can be replaced with the builder?
    #[inline]
    pub fn data_port_width(&self) -> usize {
        let width = self.data_port_width.unwrap_or(0);
        let width = if width == 0 {
            // default granularity is line size
            self.line_size as usize
        } else {
            width
        };
        debug_assert!(self.line_size as usize % width == 0);
        width
    }

    /// The total size of the cache in bytes.
    #[inline]
    pub fn total_bytes(&self) -> usize {
        self.line_size as usize * self.num_sets * self.associativity
    }

    /// Number of lines in total.
    #[inline]
    pub fn total_lines(&self) -> usize {
        self.num_sets * self.associativity
    }

    /// Maximum number of lines.
    #[inline]
    pub fn max_num_lines(&self) -> usize {
        self.max_cache_multiplier() as usize * self.num_sets * self.associativity
    }

    /// this is virtual (possibly different)
    #[inline]
    pub fn max_cache_multiplier(&self) -> u8 {
        MAX_DEFAULT_CACHE_SIZE_MULTIPLIER
    }

    #[inline]
    pub fn line_size_log2(&self) -> u32 {
        addrdec::logb2(self.line_size as u32)
    }

    #[inline]
    pub fn num_sets_log2(&self) -> u32 {
        addrdec::logb2(self.num_sets as u32)
    }

    #[inline]
    pub fn sector_size(&self) -> u32 {
        mem_sub_partition::SECTOR_SIZE
    }

    #[inline]
    pub fn sector_size_log2(&self) -> u32 {
        addrdec::logb2(self.sector_size())
    }

    #[inline]
    pub fn atom_size(&self) -> u32 {
        if self.kind == CacheKind::Sector {
            mem_sub_partition::SECTOR_SIZE
        } else {
            self.line_size
        }
    }

    // do not use enabled but options
    #[inline]
    pub fn set_index(&self, addr: address) -> u64 {
        println!("cache_config::set_index({})", addr);
        hash_function(
            addr,
            self.num_sets,
            self.line_size_log2(),
            self.num_sets_log2(),
            self.set_index_function,
        )
    }

    #[inline]
    pub fn tag(&self, addr: address) -> address {
        // For generality, the tag includes both index and tag.
        // This allows for more complex set index calculations that
        // can result in different indexes mapping to the same set,
        // thus the full tag + index is required to check for hit/miss.
        // Tag is now identical to the block address.

        // return addr >> (m_line_sz_log2+m_nset_log2);
        // return addr & ~(new_addr_type)(m_line_sz - 1);
        addr & !((self.line_size - 1) as u64)
    }

    /// Block address
    #[inline]
    pub fn block_addr(&self, addr: address) -> address {
        addr & !((self.line_size - 1) as u64)
    }

    /// Mshr address
    #[inline]
    pub fn mshr_addr(&self, addr: address) -> address {
        addr & !((self.line_size - 1) as u64)
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

fn hash_function(
    addr: address,
    num_sets: usize,
    line_size_log2: u32,
    num_sets_log2: u32,
    set_index_function: CacheSetIndexFunc,
) -> u64 {
    use super::ported::set_index_function as indexing;

    let set_idx: u64 = match set_index_function {
        CacheSetIndexFunc::LINEAR_SET_FUNCTION => (addr >> line_size_log2) & (num_sets as u64 - 1),
        CacheSetIndexFunc::FERMI_HASH_SET_FUNCTION => {
            // Set Indexing function from
            // "A Detailed GPU Cache Model Based on Reuse
            // Distance Theory" Cedric Nugteren et al. HPCA 2014

            // check for incorrect number of sets
            assert!(
                    matches!(num_sets, 32 | 64),
                    "bad cache config: number of sets should be 32 or 64 for the hashing set index function",
                );

            let mut lower_xor = 0;
            let mut upper_xor = 0;

            // lower xor value is bits 7-11
            lower_xor = (addr >> line_size_log2) & 0x1F;

            // upper xor value is bits 13, 14, 15, 17, and 19
            upper_xor = (addr & 0xE000) >> 13; // Bits 13, 14, 15
            upper_xor |= (addr & 0x20000) >> 14; // Bit 17
            upper_xor |= (addr & 0x80000) >> 15; // Bit 19

            let mut set_index = lower_xor ^ upper_xor;

            // 48KB cache prepends the set_index with bit 12
            if num_sets == 64 {
                set_index |= (addr & 0x1000) >> 7;
            }
            set_index
        }
        CacheSetIndexFunc::HASH_IPOLY_FUNCTION => {
            let bits = line_size_log2 + num_sets_log2;
            let higher_bits = addr >> bits;
            let mut index = (addr >> line_size_log2) as usize;
            index &= num_sets - 1;
            indexing::ipoly_hash_function(higher_bits, index, num_sets)
        }

        CacheSetIndexFunc::BITWISE_XORING_FUNCTION => {
            let bits = line_size_log2 + num_sets_log2;
            let higher_bits = addr >> bits;
            let mut index = (addr >> line_size_log2) as usize;
            index &= num_sets - 1;
            indexing::bitwise_hash_function(higher_bits, index, num_sets)
        }
    };

    assert!(
            set_idx < num_sets as u64,
             "Error: Set index out of bounds. This is caused by an incorrect or unimplemented set index function."
        );
    set_idx
}

impl std::fmt::Display for CacheConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let size = human_bytes::human_bytes(self.total_bytes() as f64);
        write!(
            f,
            "{size} ({} set, {}-way, {} byte line)",
            self.num_sets, self.associativity, self.line_size
        )
    }
}

/// todo: remove the copy stuff, very expensive otherwise
#[derive(Debug, PartialEq, Eq)]
pub struct GPUConfig {
    /// The SM number to pass to ptxas when getting register usage for
    /// computing GPU occupancy.
    pub occupancy_sm_number: usize,
    /// num threads per shader core pipeline
    pub max_threads_per_core: usize,
    /// shader core pipeline warp size
    pub warp_size: usize,
    /// per-shader read-only L1 texture cache config
    pub tex_cache_l1: Option<Arc<CacheConfig>>,
    /// per-shader read-only L1 constant memory cache config
    pub const_cache_l1: Option<Arc<CacheConfig>>,
    /// shader L1 instruction cache config
    pub inst_cache_l1: Option<Arc<CacheConfig>>,
    /// per-shader L1 data cache config
    pub data_cache_l1: Option<Arc<L1DCacheConfig>>,
    /// unified banked L2 data cache config
    pub data_cache_l2: Option<Arc<CacheConfig>>,

    /// L1D write ratio
    // pub l1_cache_write_ratio: usize,
    /// The number of L1 cache banks
    // pub l1_banks: usize,
    // /// L1 banks byte interleaving granularity
    // pub l1_banks_byte_interleaving: usize,
    // // L1 banks hashing function
    // pub l1_banks_hashing_function: usize,
    // /// L1 Hit Latency
    // pub l1_latency: usize,
    /// smem Latency
    pub shared_memory_latency: usize,
    /// implements -Xptxas -dlcm=cg, default=no skip
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
    pub num_cta_barriers: usize, // 16
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
    pub dual_issue_diff_exec_units: bool, // 1
    /// Select the simulation order of cores in a cluster
    pub simt_core_sim_order: SchedulingOrder, // 1
    // Pipeline widths
    //
    // ID_OC_SP,ID_OC_DP,ID_OC_INT,ID_OC_SFU,ID_OC_MEM,OC_EX_SP,OC_EX_DP,
    // OC_EX_INT,OC_EX_SFU,OC_EX_MEM,EX_WB,ID_OC_TENSOR_CORE,OC_EX_TENSOR_CORE
    //
    pub pipeline_widths: HashMap<PipelineStage, usize>, // 4,0,0,1,1,4,0,0,1,1,6
    /// Number of SP units
    pub num_sp_units: usize, // 4
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
    /// Number of ldst units
    ///
    /// WARNING: not hooked up to anything
    pub num_mem_units: usize, // 1
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
    pub num_sub_partition_per_memory_channel: usize, // 2
    /// number of memory chips per memory controller
    pub num_memory_chips_per_controller: usize, // 1
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
    /// dram_timing_opt
    /// nbk=16:CCD=2:RRD=6:RCD=12:RAS=28:RP=12:RC=40: CL=12:WL=4:CDLR=5:WR=12:nbkgrp=1:CCDL=0:RTPL=0
    /// ROP queue latency (default 85)
    pub l2_rop_latency: usize, // 120
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
    // memory_addr_mapping: String, // dramid@8;00000000.00000000.00000000.00000000.0000RRRR.RRRRRRRR.RBBBCCCC.BCCSSSSS
    /// run sweep test to check address mapping for aliased address
    // memory_addr_test: bool, // 0
    /// 0 = old addressing mask, 1 = new addressing mask, 2 = new add. mask + flipped bank sel and chip sel bits
    // memory_address_mask: usize, // 1
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
}

pub static WORD_SIZE: address = 4;

impl GPUConfig {
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

    pub fn threads_per_block_padded(&self, kernel: &KernelInfo) -> usize {
        let threads_per_block = kernel.threads_per_block();
        utils::pad_to_multiple(threads_per_block as usize, self.warp_size)
    }

    pub fn max_blocks(&self, kernel: &KernelInfo) -> eyre::Result<usize> {
        let threads_per_block = kernel.threads_per_block();
        let threads_per_block = utils::pad_to_multiple(threads_per_block as usize, self.warp_size);
        // limit by n_threads/shader
        let by_thread_limit = self.max_threads_per_core / threads_per_block as usize;

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
        let by_block_limit = self.max_concurrent_blocks_per_core;

        // find the minimum
        let mut limit = [
            Some(by_thread_limit),
            by_shared_mem_limit,
            by_register_limit,
        ]
        .into_iter()
        .filter_map(|limit| limit)
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

        if self.adaptive_cache_config && !kernel.cache_config_set {
            // more info about adaptive cache, see
            // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
            let total_shared_mem = kernel.config.shared_mem_bytes as usize * limit;
            assert!(
                total_shared_mem >= 0
                    && self
                        .shared_memory_sizes
                        .last()
                        .map(|size| total_shared_mem <= (*size as usize))
                        .unwrap_or(true)
            );
        }

        Ok(limit)

        // if (adaptive_cache_config && !k.cache_config_set) {
        //   // For more info about adaptive cache, see
        //   // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
        //   unsigned total_shmem = kernel_info->smem * result;
        //   assert(total_shmem >= 0 && total_shmem <= shmem_opt_list.back());
        //
        //   // Unified cache config is in KB. Converting to B
        //   unsigned total_unified = m_L1D_config.m_unified_cache_size * 1024;
        //
        //   bool l1d_configured = false;
        //   unsigned max_assoc = m_L1D_config.get_max_assoc();
        //
        //   for (std::vector<unsigned>::const_iterator it = shmem_opt_list.begin();
        //        it < shmem_opt_list.end(); it++) {
        //     if (total_shmem <= *it) {
        //       float l1_ratio = 1 - ((float)*(it) / total_unified);
        //       // make sure the ratio is between 0 and 1
        //       assert(0 <= l1_ratio && l1_ratio <= 1);
        //       // round to nearest instead of round down
        //       m_L1D_config.set_assoc(max_assoc * l1_ratio + 0.5f);
        //       l1d_configured = true;
        //       break;
        //     }
        //   }
        //
        //   assert(l1d_configured && "no shared memory option found");
        //
        //   if (m_L1D_config.is_streaming()) {
        //     // for streaming cache, if the whole memory is allocated
        //     // to the L1 cache, then make the allocation to be on_MISS
        //     // otherwise, make it ON_FILL to eliminate line allocation fails
        //     // i.e. MSHR throughput is the same, independent on the L1 cache
        //     // size/associativity
        //     if (total_shmem == 0) {
        //       m_L1D_config.set_allocation_policy(ON_MISS);
        //       printf("GPGPU-Sim: Reconfigure L1 allocation to ON_MISS\n");
        //     } else {
        //       m_L1D_config.set_allocation_policy(ON_FILL);
        //       printf("GPGPU-Sim: Reconfigure L1 allocation to ON_FILL\n");
        //     }
        //   }
        //   printf("GPGPU-Sim: Reconfigure L1 cache to %uKB\n",
        //          m_L1D_config.get_total_size_inKB());
        //
        //   k.cache_config_set = true;
        // }

        // unsigned threads_per_cta = k.threads_per_cta();
        // const class function_info *kernel = k.entry();
        // unsigned int padded_cta_size = threads_per_cta;
        // if (padded_cta_size % warp_size)
        //   padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);
        //
        // // Limit by n_threads/shader
        // unsigned int result_thread = n_thread_per_shader / padded_cta_size;
        //
        // const struct gpgpu_ptx_sim_info *kernel_info = ptx_sim_kernel_info(kernel);
        //
        // // Limit by shmem/shader
        // unsigned int result_shmem = (unsigned)-1;
        // if (kernel_info->smem > 0)
        //   result_shmem = gpgpu_shmem_size / kernel_info->smem;
        //
        // // Limit by register count, rounded up to multiple of 4.
        // unsigned int result_regs = (unsigned)-1;
        // if (kernel_info->regs > 0)
        //   result_regs = gpgpu_shader_registers /
        //                 (padded_cta_size * ((kernel_info->regs + 3) & ~3));
        //
        // // Limit by CTA
        // unsigned int result_cta = max_cta_per_core;
        //
        // unsigned result = result_thread;
        // result = gs_min2(result, result_shmem);
        // result = gs_min2(result, result_regs);
        // result = gs_min2(result, result_cta);
        //
        // static const struct gpgpu_ptx_sim_info *last_kinfo = NULL;
        // if (last_kinfo !=
        //     kernel_info) {  // Only print out stats if kernel_info struct changes
        //   last_kinfo = kernel_info;
        //   printf("GPGPU-Sim uArch: CTA/core = %u, limited by:", result);
        //   if (result == result_thread) printf(" threads");
        //   if (result == result_shmem) printf(" shmem");
        //   if (result == result_regs) printf(" regs");
        //   if (result == result_cta) printf(" cta_limit");
        //   printf("\n");
        // }
        //
        // // gpu_max_cta_per_shader is limited by number of CTAs if not enough to keep
        // // all cores busy
        // if (k.num_blocks() < result * num_shader()) {
        //   result = k.num_blocks() / num_shader();
        //   if (k.num_blocks() % num_shader()) result++;
        // }
        //
        // assert(result <= MAX_CTA_PER_SHADER);
        // if (result < 1) {
        //   printf(
        //       "GPGPU-Sim uArch: ERROR ** Kernel requires more resources than shader "
        //       "has.\n");
        //   if (gpgpu_ignore_resources_limitation) {
        //     printf(
        //         "GPGPU-Sim uArch: gpgpu_ignore_resources_limitation is set, ignore "
        //         "the ERROR!\n");
        //     return 1;
        //   }
        //   abort();
        // }
        //
        // if (adaptive_cache_config && !k.cache_config_set) {
        //   // For more info about adaptive cache, see
        //   // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
        //   unsigned total_shmem = kernel_info->smem * result;
        //   assert(total_shmem >= 0 && total_shmem <= shmem_opt_list.back());
        //
        //   // Unified cache config is in KB. Converting to B
        //   unsigned total_unified = m_L1D_config.m_unified_cache_size * 1024;
        //
        //   bool l1d_configured = false;
        //   unsigned max_assoc = m_L1D_config.get_max_assoc();
        //
        //   for (std::vector<unsigned>::const_iterator it = shmem_opt_list.begin();
        //        it < shmem_opt_list.end(); it++) {
        //     if (total_shmem <= *it) {
        //       float l1_ratio = 1 - ((float)*(it) / total_unified);
        //       // make sure the ratio is between 0 and 1
        //       assert(0 <= l1_ratio && l1_ratio <= 1);
        //       // round to nearest instead of round down
        //       m_L1D_config.set_assoc(max_assoc * l1_ratio + 0.5f);
        //       l1d_configured = true;
        //       break;
        //     }
        //   }
        //
        //   assert(l1d_configured && "no shared memory option found");
        //
        //   if (m_L1D_config.is_streaming()) {
        //     // for streaming cache, if the whole memory is allocated
        //     // to the L1 cache, then make the allocation to be on_MISS
        //     // otherwise, make it ON_FILL to eliminate line allocation fails
        //     // i.e. MSHR throughput is the same, independent on the L1 cache
        //     // size/associativity
        //     if (total_shmem == 0) {
        //       m_L1D_config.set_allocation_policy(ON_MISS);
        //       printf("GPGPU-Sim: Reconfigure L1 allocation to ON_MISS\n");
        //     } else {
        //       m_L1D_config.set_allocation_policy(ON_FILL);
        //       printf("GPGPU-Sim: Reconfigure L1 allocation to ON_FILL\n");
        //     }
        //   }
        //   printf("GPGPU-Sim: Reconfigure L1 cache to %uKB\n",
        //          m_L1D_config.get_total_size_inKB());
        //
        //   k.cache_config_set = true;
        // }
        //
        // return result;
    }
}
//     pub fn new() -> Self {
//         Self {}
//         // gpu_stat_sample_freq = 10000;
//         // gpu_runtime_stat_flag = 0;
//         // sscanf(gpgpu_runtime_stat, "%d:%x", &gpu_stat_sample_freq,
//         //        &gpu_runtime_stat_flag);
//         // m_shader_config.init();
//         // ptx_set_tex_cache_linesize(m_shader_config.m_L1T_config.get_line_sz());
//         // m_memory_config.init();
//         // init_clock_domains();
//         // power_config::init();
//         // Trace::init();
//     }
// }

/// Cache set indexing function kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CacheSetIndexFunc {
    FERMI_HASH_SET_FUNCTION, // H
    HASH_IPOLY_FUNCTION,     // P
    // CUSTOM_SET_FUNCTION, // C
    LINEAR_SET_FUNCTION,     // L
    BITWISE_XORING_FUNCTION, // X
}

/// Miss status handlign register kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MshrKind {
    TEX_FIFO,        // F
    SECTOR_TEX_FIFO, // T
    ASSOC,           // A
    SECTOR_ASSOC,    // S
}
///
/// Cache write-allocate policy.
///
/// For more details about difference between FETCH_ON_WRITE and WRITE
/// VALIDAE policies Read: Jouppi, Norman P. "Cache write policies and
/// performance". ISCA 93. WRITE_ALLOCATE is the old write policy in
/// GPGPU-sim 3.x, that send WRITE and READ for every write request
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CacheWriteAllocatePolicy {
    NO_WRITE_ALLOCATE,  // N
    WRITE_ALLOCATE,     // W
    FETCH_ON_WRITE,     // F
    LAZY_FETCH_ON_READ, // L
}

/// A cache write policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CacheWritePolicy {
    READ_ONLY,          // R
    WRITE_BACK,         // B
    WRITE_THROUGH,      // T
    WRITE_EVICT,        // E
    LOCAL_WB_GLOBAL_WT, // L
}

/// A cache allocate policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CacheAllocatePolicy {
    ON_MISS,   // M
    ON_FILL,   // F
    STREAMING, // S
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

/// Scheduler kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

impl GPUConfig {
    pub fn parse() -> eyre::Result<Self> {
        let adaptive_cache_config = false;
        let shared_memory_sizes_string = "0";
        let shared_memory_sizes: Vec<u32> = if adaptive_cache_config {
            let sizes: Result<Vec<u32>, _> = shared_memory_sizes_string
                .split(",")
                .map(str::parse)
                .collect();
            let mut sizes: Vec<_> = sizes?.into_iter().map(|size| size * 1024).collect();
            sizes.sort();
            sizes
        } else {
            vec![]
        };
        Ok(Self::default())
    }

    pub fn total_sub_partitions(&self) -> usize {
        self.num_mem_units * self.num_sub_partition_per_memory_channel
    }

    pub fn address_mapping(&self) -> addrdec::LinearToRawAddressTranslation {
        addrdec::LinearToRawAddressTranslation::new(
            self.num_memory_controllers,
            self.num_sub_partition_per_memory_channel,
        )
    }
}

// opp, "-gpgpu_pipeline_widths", OPT_CSTR, &pipeline_widths_string,
//       "Pipeline widths "
//       "ID_OC_SP,ID_OC_DP,ID_OC_INT,ID_OC_SFU,ID_OC_MEM,OC_EX_SP,OC_EX_DP,OC_EX_"
//       "INT,OC_EX_SFU,OC_EX_MEM,EX_WB,ID_OC_TENSOR_CORE,OC_EX_TENSOR_CORE",
//       "1,1,1,1,1,1,1,1,1,1,1,1,1")

impl Default for GPUConfig {
    fn default() -> Self {
        Self {
            occupancy_sm_number: 60,
            max_threads_per_core: 2048,
            warp_size: 32,
            // N:16:128:24,L:R:m:N:L,F:128:4,128:2
            // {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>:<rf>}
            tex_cache_l1: Some(Arc::new(CacheConfig {
                kind: CacheKind::Normal,
                num_sets: 16,
                line_size: 128,
                associativity: 24,
                replacement_policy: CacheReplacementPolicy::LRU,
                write_policy: CacheWritePolicy::READ_ONLY,
                allocate_policy: CacheAllocatePolicy::ON_MISS,
                write_allocate_policy: CacheWriteAllocatePolicy::NO_WRITE_ALLOCATE,
                set_index_function: CacheSetIndexFunc::LINEAR_SET_FUNCTION,
                mshr_kind: MshrKind::TEX_FIFO,
                mshr_entries: 128,
                mshr_max_merge: 4,
                miss_queue_size: 128,
                result_fifo_entries: Some(2),
                data_port_width: None,
            })),
            // N:128:64:2,L:R:f:N:L,A:2:64,4
            // {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>}
            const_cache_l1: Some(Arc::new(CacheConfig {
                kind: CacheKind::Normal,
                num_sets: 128,
                line_size: 64,
                associativity: 2,
                replacement_policy: CacheReplacementPolicy::LRU,
                write_policy: CacheWritePolicy::READ_ONLY,
                allocate_policy: CacheAllocatePolicy::ON_FILL,
                write_allocate_policy: CacheWriteAllocatePolicy::NO_WRITE_ALLOCATE,
                set_index_function: CacheSetIndexFunc::LINEAR_SET_FUNCTION,
                mshr_kind: MshrKind::ASSOC,
                mshr_entries: 2,
                mshr_max_merge: 64,
                miss_queue_size: 4,
                result_fifo_entries: None,
                data_port_width: None,
            })),
            // N:8:128:4,L:R:f:N:L,A:2:48,4
            // {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>}
            inst_cache_l1: Some(Arc::new(CacheConfig {
                kind: CacheKind::Normal,
                num_sets: 8,
                line_size: 128,
                associativity: 4,
                replacement_policy: CacheReplacementPolicy::LRU,
                write_policy: CacheWritePolicy::READ_ONLY,
                allocate_policy: CacheAllocatePolicy::ON_FILL,
                write_allocate_policy: CacheWriteAllocatePolicy::NO_WRITE_ALLOCATE,
                set_index_function: CacheSetIndexFunc::LINEAR_SET_FUNCTION,
                mshr_kind: MshrKind::ASSOC,
                mshr_entries: 2,
                mshr_max_merge: 48,
                miss_queue_size: 4,
                result_fifo_entries: None,
                data_port_width: None,
            })),
            // N:64:128:6,L:L:m:N:H,A:128:8,8
            // {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq> | none}
            data_cache_l1: Some(Arc::new(L1DCacheConfig {
                l1_latency: 1,
                l1_banks_hashing_function: CacheSetIndexFunc::FERMI_HASH_SET_FUNCTION,
                l1_banks_byte_interleaving: 32,
                l1_banks: 1,
                m_wr_percent: 0,
                inner: Arc::new(CacheConfig {
                    kind: CacheKind::Normal,
                    num_sets: 64,
                    line_size: 128,
                    associativity: 6,
                    replacement_policy: CacheReplacementPolicy::LRU,
                    write_policy: CacheWritePolicy::LOCAL_WB_GLOBAL_WT,
                    allocate_policy: CacheAllocatePolicy::ON_MISS,
                    write_allocate_policy: CacheWriteAllocatePolicy::NO_WRITE_ALLOCATE,
                    set_index_function: CacheSetIndexFunc::FERMI_HASH_SET_FUNCTION,
                    mshr_kind: MshrKind::ASSOC,
                    mshr_entries: 128,
                    mshr_max_merge: 8,
                    miss_queue_size: 4,
                    result_fifo_entries: None,
                    data_port_width: None,
                }),
            })),
            // N:64:128:16,L:B:m:W:L,A:1024:1024,4:0,32
            // {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>}
            data_cache_l2: Some(Arc::new(CacheConfig {
                kind: CacheKind::Normal,
                num_sets: 64,
                line_size: 128,
                associativity: 16,
                replacement_policy: CacheReplacementPolicy::LRU,
                write_policy: CacheWritePolicy::WRITE_BACK,
                allocate_policy: CacheAllocatePolicy::ON_MISS,
                write_allocate_policy: CacheWriteAllocatePolicy::WRITE_ALLOCATE,
                set_index_function: CacheSetIndexFunc::LINEAR_SET_FUNCTION,
                mshr_kind: MshrKind::ASSOC,
                mshr_entries: 128,
                mshr_max_merge: 8,
                miss_queue_size: 32,
                result_fifo_entries: None,
                data_port_width: None,
            })),
            // l1_cache_write_ratio: 0,
            // l1_banks: 1,
            // l1_banks_byte_interleaving: 32,
            // l1_banks_hashing_function: 0,
            // l1_latency: 1,
            shared_memory_latency: 3,
            global_mem_skip_l1_data_cache: true,
            perfect_mem: false,
            shader_registers: 65536,
            registers_per_block: 8192,
            ignore_resources_limitation: false,
            max_concurrent_blocks_per_core: 32,
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
            shared_memory_sizes: vec![],
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
            enable_specialized_operand_collector: true,
            operand_collector_num_units_sp: 20, // 4,
            operand_collector_num_units_dp: 0,
            operand_collector_num_units_sfu: 4,
            operand_collector_num_units_int: 0,
            operand_collector_num_units_tensor_core: 4,
            operand_collector_num_units_mem: 8, // 2,
            operand_collector_num_units_gen: 0,
            operand_collector_num_in_ports_sp: 4, // 1,
            operand_collector_num_in_ports_dp: 0,
            operand_collector_num_in_ports_sfu: 1,
            operand_collector_num_in_ports_int: 0,
            operand_collector_num_in_ports_tensor_core: 1,
            operand_collector_num_in_ports_mem: 1,
            operand_collector_num_in_ports_gen: 0,
            operand_collector_num_out_ports_sp: 4, // 1,
            operand_collector_num_out_ports_dp: 0,
            operand_collector_num_out_ports_sfu: 1,
            operand_collector_num_out_ports_int: 0,
            operand_collector_num_out_ports_tensor_core: 1,
            operand_collector_num_out_ports_mem: 1,
            operand_collector_num_out_ports_gen: 0,
            coalescing_arch: Architecture::GT200,
            num_schedulers_per_core: 2,
            max_instruction_issue_per_warp: 2,
            dual_issue_diff_exec_units: true,
            simt_core_sim_order: SchedulingOrder::RoundRobin,
            pipeline_widths: HashMap::from_iter([
                (PipelineStage::ID_OC_SP, 4),
                (PipelineStage::ID_OC_DP, 0),
                (PipelineStage::ID_OC_INT, 0),
                (PipelineStage::ID_OC_SFU, 1),
                (PipelineStage::ID_OC_MEM, 1),
                (PipelineStage::OC_EX_SP, 4),
                (PipelineStage::OC_EX_DP, 0),
                (PipelineStage::OC_EX_INT, 0),
                (PipelineStage::OC_EX_SFU, 1),
                (PipelineStage::OC_EX_MEM, 1),
                (PipelineStage::EX_WB, 6),
                // don't have tensor cores
                (PipelineStage::ID_OC_TENSOR_CORE, 0),
                (PipelineStage::OC_EX_TENSOR_CORE, 0),
            ]),
            num_sp_units: 4,
            num_dp_units: 0,
            num_int_units: 0,
            num_sfu_units: 1,
            num_tensor_core_avail: 0,
            num_tensor_core_units: 0,
            num_mem_units: 1,
            scheduler: CoreSchedulerKind::GTO,
            concurrent_kernel_sm: false,
            perfect_inst_const_cache: false,
            inst_fetch_throughput: 1,
            reg_file_port_throughput: 1,
            fill_l2_on_memcopy: true,
            simple_dram_model: false,
            dram_scheduler: DRAMSchedulerKind::FrFcfs,
            dram_partition_queue_interconn_to_l2: 8,
            dram_partition_queue_l2_to_dram: 8,
            dram_partition_queue_dram_to_l2: 8,
            dram_partition_queue_l2_to_interconn: 8,
            ideal_l2: false,
            data_cache_l2_texture_only: false,
            num_memory_controllers: 8,
            num_sub_partition_per_memory_channel: 2,
            num_memory_chips_per_controller: 1,
            dram_frfcfs_sched_queue_size: 64,
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
            dram_frfcfs_write_queue_size: 32, // 32:28:16
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

#[cfg(test)]
mod tests {
    use playground::bindings;
    use pretty_assertions::assert_eq;
    use std::ffi;
    use std::mem::MaybeUninit;

    #[test]
    fn test_scanf_cache_config() {
        let config = "N:16:128:24,L:R:m:N:L,F:128:4,128:2";
        let config = ffi::CString::new(config).unwrap();
        let mut cache_config = MaybeUninit::uninit();
        unsafe {
            bindings::parse_cache_config(config.as_ptr() as *mut _, cache_config.as_mut_ptr());
        }
        let cache_config = unsafe { cache_config.assume_init() };
        assert_eq!(
            cache_config,
            bindings::CacheConfig {
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
            }
        );
    }

    #[test]
    fn test_block_addr() {
        let config = super::GPUConfig::default();
        let instr_cache_config = config.inst_cache_l1.unwrap();
        assert_eq!(instr_cache_config.block_addr(4026531848), 4026531840);
    }

    #[test]
    fn test_mshr_addr() {
        let config = super::GPUConfig::default();
        let instr_cache_config = config.inst_cache_l1.unwrap();
        assert_eq!(instr_cache_config.mshr_addr(4026531848), 4026531840);
        assert_eq!(instr_cache_config.mshr_addr(4026531992), 4026531968);
    }
}
