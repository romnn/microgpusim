use crate::bindings;
use std::collections::HashMap;

impl From<stats::mem::Accesses> for self::Accesses {
    fn from(other: stats::mem::Accesses) -> Self {
        Self {
            num_mem_write: other.num_writes(),
            num_mem_read: other.num_reads(),
            num_mem_const: other
                .get(&stats::mem::AccessKind::CONST_ACC_R)
                .copied()
                .unwrap_or(0),
            num_mem_texture: other
                .get(&stats::mem::AccessKind::TEXTURE_ACC_R)
                .copied()
                .unwrap_or(0),
            num_mem_read_global: other
                .get(&stats::mem::AccessKind::GLOBAL_ACC_R)
                .copied()
                .unwrap_or(0),
            num_mem_write_global: other
                .get(&stats::mem::AccessKind::GLOBAL_ACC_W)
                .copied()
                .unwrap_or(0),
            num_mem_read_local: other
                .get(&stats::mem::AccessKind::LOCAL_ACC_R)
                .copied()
                .unwrap_or(0),
            num_mem_write_local: other
                .get(&stats::mem::AccessKind::LOCAL_ACC_W)
                .copied()
                .unwrap_or(0),
            num_mem_l2_writeback: other
                .get(&stats::mem::AccessKind::L2_WRBK_ACC)
                .copied()
                .unwrap_or(0),
            num_mem_l1_write_allocate: other
                .get(&stats::mem::AccessKind::L1_WR_ALLOC_R)
                .copied()
                .unwrap_or(0),
            num_mem_l2_write_allocate: other
                .get(&stats::mem::AccessKind::L2_WR_ALLOC_R)
                .copied()
                .unwrap_or(0),
        }
    }
}

impl From<stats::dram::DRAM> for self::DRAM {
    fn from(other: stats::dram::DRAM) -> Self {
        Self {
            total_reads: other.total_reads() as usize,
            total_writes: other.total_writes() as usize,
        }
    }
}

impl From<stats::instructions::InstructionCounts> for self::Instructions {
    fn from(other: stats::instructions::InstructionCounts) -> Self {
        let num_global_loads = other
            .get(&(stats::instructions::MemorySpace::Global, false))
            .copied()
            .unwrap_or(0);
        let num_local_loads = other
            .get(&(stats::instructions::MemorySpace::Local, false))
            .copied()
            .unwrap_or(0);
        let num_global_stores = other
            .get(&(stats::instructions::MemorySpace::Global, true))
            .copied()
            .unwrap_or(0);
        let num_local_stores = other
            .get(&(stats::instructions::MemorySpace::Local, true))
            .copied()
            .unwrap_or(0);
        let num_shmem = other.get_total(stats::instructions::MemorySpace::Shared);
        let num_tex = other.get_total(stats::instructions::MemorySpace::Texture);
        let num_const = other.get_total(stats::instructions::MemorySpace::Constant);

        Self {
            num_load_instructions: num_local_loads + num_global_loads,
            num_store_instructions: num_local_stores + num_global_stores,
            num_shared_mem_instructions: num_shmem,
            num_sstarr_instructions: 0,
            num_texture_instructions: num_tex,
            num_const_instructions: num_const,
            num_param_instructions: 0,
        }
    }
}

impl From<stats::sim::Sim> for self::Sim {
    fn from(other: stats::sim::Sim) -> Self {
        Self {
            cycle: other.cycles,
            instructions: other.instructions,
        }
    }
}

impl PartialEq<stats::Stats> for self::Stats {
    fn eq(&self, other: &stats::Stats) -> bool {
        use stats::ConvertHashMap;
        if !(stats::PerCache(self.l1i_stats.clone().convert()) == other.l1i_stats)
            && (stats::PerCache(self.l1d_stats.clone().convert()) == other.l1d_stats)
            && (stats::PerCache(self.l1t_stats.clone().convert()) == other.l1t_stats)
            && (stats::PerCache(self.l1c_stats.clone().convert()) == other.l1c_stats)
            && (stats::PerCache(self.l2d_stats.clone().convert()) == other.l2d_stats)
        {
            return false;
        }

        if self.accesses != self::Accesses::from(other.accesses.clone()) {
            return false;
        }

        if self.dram != self::DRAM::from(other.dram.clone()) {
            return false;
        }

        if self.instructions != self::Instructions::from(other.instructions.clone()) {
            return false;
        }

        self.sim == self::Sim::from(other.sim.clone())
    }
}

pub type RequestStatus = bindings::cache_request_status;
pub type AccessType = bindings::mem_access_type;
pub type ReservationFailure = bindings::cache_reservation_fail_reason;

#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub accesses: HashMap<(AccessType, AccessStat), u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessStat {
    ReservationFailure(ReservationFailure),
    Status(RequestStatus),
}

/// DRAM stats
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DRAM {
    // total accesses are always zero (never set by accelsim)
    // we ignore them to spare confusion
    // pub total_accesses: usize,
    pub total_reads: usize,
    pub total_writes: usize,
}

/// Memory accesses
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Accesses {
    pub num_mem_write: usize,
    pub num_mem_read: usize,
    pub num_mem_const: usize,
    pub num_mem_texture: usize,
    pub num_mem_read_global: usize,
    pub num_mem_write_global: usize,
    pub num_mem_read_local: usize,
    pub num_mem_write_local: usize,
    pub num_mem_l2_writeback: usize,
    pub num_mem_l1_write_allocate: usize,
    pub num_mem_l2_write_allocate: usize,
}

/// Instruction counts
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Instructions {
    pub num_load_instructions: usize,
    pub num_store_instructions: usize,
    pub num_shared_mem_instructions: usize,
    pub num_sstarr_instructions: usize,
    pub num_texture_instructions: usize,
    pub num_const_instructions: usize,
    pub num_param_instructions: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Sim {
    pub cycle: usize,
    pub instructions: usize,
}

#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub accesses: Accesses,
    pub instructions: Instructions,
    pub sim: Sim,
    pub dram: DRAM,

    // per cache stats
    pub l1i_stats: HashMap<usize, CacheStats>,
    pub l1c_stats: HashMap<usize, CacheStats>,
    pub l1t_stats: HashMap<usize, CacheStats>,
    pub l1d_stats: HashMap<usize, CacheStats>,
    pub l2d_stats: HashMap<usize, CacheStats>,
}

impl From<CacheStats> for stats::Cache {
    fn from(stats: CacheStats) -> Self {
        Self {
            accesses: stats
                .accesses
                .into_iter()
                .map(|((access_kind, access_stat), count)| {
                    (
                        (access_kind.into(), access_stat.into()),
                        count.try_into().unwrap(),
                    )
                })
                .collect(),
        }
    }
}

impl From<AccessType> for stats::mem::AccessKind {
    fn from(kind: AccessType) -> Self {
        match kind {
            AccessType::GLOBAL_ACC_R => stats::mem::AccessKind::GLOBAL_ACC_R,
            AccessType::LOCAL_ACC_R => stats::mem::AccessKind::LOCAL_ACC_R,
            AccessType::CONST_ACC_R => stats::mem::AccessKind::CONST_ACC_R,
            AccessType::TEXTURE_ACC_R => stats::mem::AccessKind::TEXTURE_ACC_R,
            AccessType::GLOBAL_ACC_W => stats::mem::AccessKind::GLOBAL_ACC_W,
            AccessType::LOCAL_ACC_W => stats::mem::AccessKind::LOCAL_ACC_W,
            AccessType::L1_WRBK_ACC => stats::mem::AccessKind::L1_WRBK_ACC,
            AccessType::L2_WRBK_ACC => stats::mem::AccessKind::L2_WRBK_ACC,
            AccessType::INST_ACC_R => stats::mem::AccessKind::INST_ACC_R,
            AccessType::L1_WR_ALLOC_R => stats::mem::AccessKind::L1_WR_ALLOC_R,
            AccessType::L2_WR_ALLOC_R => stats::mem::AccessKind::L2_WR_ALLOC_R,
            other @ AccessType::NUM_MEM_ACCESS_TYPE => {
                panic!("bad mem access type: {:?}", other)
            }
        }
    }
}

impl From<AccessStat> for stats::cache::AccessStat {
    fn from(stat: AccessStat) -> Self {
        match stat {
            AccessStat::Status(status) => stats::cache::AccessStat::Status(status.into()),
            AccessStat::ReservationFailure(failure) => {
                stats::cache::AccessStat::ReservationFailure(failure.into())
            }
        }
    }
}

impl From<ReservationFailure> for stats::cache::ReservationFailure {
    fn from(failure: ReservationFailure) -> Self {
        match failure {
            ReservationFailure::LINE_ALLOC_FAIL => {
                stats::cache::ReservationFailure::LINE_ALLOC_FAIL
            }
            ReservationFailure::MISS_QUEUE_FULL => {
                stats::cache::ReservationFailure::MISS_QUEUE_FULL
            }
            ReservationFailure::MSHR_ENRTY_FAIL => {
                stats::cache::ReservationFailure::MSHR_ENTRY_FAIL
            }
            ReservationFailure::MSHR_MERGE_ENRTY_FAIL => {
                stats::cache::ReservationFailure::MSHR_MERGE_ENTRY_FAIL
            }
            ReservationFailure::MSHR_RW_PENDING => {
                stats::cache::ReservationFailure::MSHR_RW_PENDING
            }
            other @ ReservationFailure::NUM_CACHE_RESERVATION_FAIL_STATUS => {
                panic!("bad cache request status: {:?}", other)
            }
        }
    }
}

impl From<RequestStatus> for stats::cache::RequestStatus {
    fn from(status: RequestStatus) -> Self {
        match status {
            RequestStatus::HIT => stats::cache::RequestStatus::HIT,
            RequestStatus::HIT_RESERVED => stats::cache::RequestStatus::HIT_RESERVED,
            RequestStatus::MISS => stats::cache::RequestStatus::MISS,
            RequestStatus::RESERVATION_FAIL => stats::cache::RequestStatus::RESERVATION_FAIL,
            RequestStatus::SECTOR_MISS => stats::cache::RequestStatus::SECTOR_MISS,
            RequestStatus::MSHR_HIT => stats::cache::RequestStatus::MSHR_HIT,
            other @ RequestStatus::NUM_CACHE_REQUEST_STATUS => {
                panic!("bad cache request status: {:?}", other)
            }
        }
    }
}

impl Stats {
    pub fn add_accesses(
        &mut self,
        cache_kind: CacheKind,
        cache_index: usize,
        kind: u32,
        status: u32,
        failed: bool,
        num_accesses: u64,
    ) {
        let cache = match cache_kind {
            CacheKind::L1T => &mut self.l1t_stats,
            CacheKind::L1I => &mut self.l1i_stats,
            CacheKind::L1C => &mut self.l1c_stats,
            CacheKind::L1D => &mut self.l1d_stats,
            CacheKind::L2D => &mut self.l2d_stats,
            _ => {
                return;
            }
        };

        let kind: AccessType = unsafe { std::mem::transmute(kind) };
        let status = if failed {
            AccessStat::ReservationFailure(unsafe { std::mem::transmute(status) })
        } else {
            AccessStat::Status(unsafe { std::mem::transmute(status) })
        };
        let cache = cache.entry(cache_index).or_default();
        *cache.accesses.entry((kind, status)).or_insert(0) += num_accesses;
    }

    // dram stats
    pub fn set_total_dram_accesses(&mut self, v: usize) {
        // self.dram.total_accesses = v;
    }
    pub fn set_total_dram_reads(&mut self, v: usize) {
        self.dram.total_reads = v;
    }
    pub fn set_total_dram_writes(&mut self, v: usize) {
        self.dram.total_writes = v;
    }

    // sim stats
    pub fn set_sim_cycle(&mut self, v: usize) {
        self.sim.cycle = v;
    }
    pub fn set_sim_instructions(&mut self, v: usize) {
        self.sim.instructions = v;
    }

    // memory accesses
    pub fn set_num_mem_write(&mut self, v: usize) {
        self.accesses.num_mem_write = v;
    }
    pub fn set_num_mem_read(&mut self, v: usize) {
        self.accesses.num_mem_read = v;
    }
    pub fn set_num_mem_const(&mut self, v: usize) {
        self.accesses.num_mem_const = v;
    }
    pub fn set_num_mem_texture(&mut self, v: usize) {
        self.accesses.num_mem_texture = v;
    }
    pub fn set_num_mem_read_global(&mut self, v: usize) {
        self.accesses.num_mem_read_global = v;
    }
    pub fn set_num_mem_write_global(&mut self, v: usize) {
        self.accesses.num_mem_write_global = v;
    }
    pub fn set_num_mem_read_local(&mut self, v: usize) {
        self.accesses.num_mem_read_local = v;
    }
    pub fn set_num_mem_write_local(&mut self, v: usize) {
        self.accesses.num_mem_write_local = v;
    }
    pub fn set_num_mem_l2_writeback(&mut self, v: usize) {
        self.accesses.num_mem_l2_writeback = v;
    }
    pub fn set_num_mem_l1_write_allocate(&mut self, v: usize) {
        self.accesses.num_mem_l1_write_allocate = v;
    }
    pub fn set_num_mem_l2_write_allocate(&mut self, v: usize) {
        self.accesses.num_mem_l2_write_allocate = v;
    }

    // instruction counts
    pub fn set_num_load_instructions(&mut self, v: usize) {
        self.instructions.num_load_instructions = v;
    }
    pub fn set_num_store_instructions(&mut self, v: usize) {
        self.instructions.num_store_instructions = v;
    }
    pub fn set_num_shared_mem_instructions(&mut self, v: usize) {
        self.instructions.num_shared_mem_instructions = v;
    }
    pub fn set_num_sstarr_instructions(&mut self, v: usize) {
        self.instructions.num_sstarr_instructions = v;
    }
    pub fn set_num_texture_instructions(&mut self, v: usize) {
        self.instructions.num_texture_instructions = v;
    }
    pub fn set_num_const_instructions(&mut self, v: usize) {
        self.instructions.num_const_instructions = v;
    }
    pub fn set_num_param_instructions(&mut self, v: usize) {
        self.instructions.num_param_instructions = v;
    }
}

#[cxx::bridge]
mod default {

    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
    pub enum CacheKind {
        // l1 caches
        L1T,
        L1I,
        L1C,
        L1D,
        // l2 cache
        L2D,
    }

    extern "Rust" {
        type Stats;

        fn add_accesses(
            self: &mut Stats,
            cache_kind: CacheKind,
            cache_index: usize,
            kind: u32,
            status: u32,
            failed: bool,
            num_accesses: u64,
        );

        // dram stats
        fn set_total_dram_accesses(self: &mut Stats, v: usize);
        fn set_total_dram_reads(self: &mut Stats, v: usize);
        fn set_total_dram_writes(self: &mut Stats, v: usize);

        // sim stats
        fn set_sim_cycle(self: &mut Stats, v: usize);
        fn set_sim_instructions(self: &mut Stats, v: usize);

        // memory accesses
        fn set_num_mem_write(self: &mut Stats, v: usize);
        fn set_num_mem_read(self: &mut Stats, v: usize);
        fn set_num_mem_const(self: &mut Stats, v: usize);
        fn set_num_mem_texture(self: &mut Stats, v: usize);
        fn set_num_mem_read_global(self: &mut Stats, v: usize);
        fn set_num_mem_write_global(self: &mut Stats, v: usize);
        fn set_num_mem_read_local(self: &mut Stats, v: usize);
        fn set_num_mem_write_local(self: &mut Stats, v: usize);
        fn set_num_mem_l2_writeback(self: &mut Stats, v: usize);
        fn set_num_mem_l1_write_allocate(self: &mut Stats, v: usize);
        fn set_num_mem_l2_write_allocate(self: &mut Stats, v: usize);

        // instruction counts
        fn set_num_load_instructions(self: &mut Stats, v: usize);
        fn set_num_store_instructions(self: &mut Stats, v: usize);
        fn set_num_shared_mem_instructions(self: &mut Stats, v: usize);
        fn set_num_sstarr_instructions(self: &mut Stats, v: usize);
        fn set_num_texture_instructions(self: &mut Stats, v: usize);
        fn set_num_const_instructions(self: &mut Stats, v: usize);
        fn set_num_param_instructions(self: &mut Stats, v: usize);

    }

    unsafe extern "C++" {
        include!("playground/src/ref/bridge/main.hpp");

        type accelsim_bridge = crate::bridge::main::accelsim_bridge;

        fn transfer_stats(self: &accelsim_bridge, stats: &mut Stats);
    }
}

pub use default::CacheKind;
