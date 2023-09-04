use crate::bindings;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub type RequestStatus = bindings::cache_request_status;
pub type AccessType = bindings::mem_access_type;
pub type ReservationFailure = bindings::cache_reservation_fail_reason;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Cache {
    pub accesses: HashMap<(AccessType, AccessStat), u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AccessStat {
    ReservationFailure(ReservationFailure),
    Status(RequestStatus),
}

/// DRAM stats
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DRAM {
    // total accesses are always zero (never set by accelsim)
    // we ignore them to spare confusion
    // pub total_accesses: usize,
    pub total_reads: u64,
    pub total_writes: u64,
}

/// Memory accesses
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Accesses {
    pub num_mem_write: u64,
    pub num_mem_read: u64,
    pub num_mem_const: u64,
    pub num_mem_texture: u64,
    pub num_mem_read_global: u64,
    pub num_mem_write_global: u64,
    pub num_mem_read_local: u64,
    pub num_mem_write_local: u64,
    pub num_mem_l2_writeback: u64,
    pub num_mem_l1_write_allocate: u64,
    pub num_mem_l2_write_allocate: u64,
}

/// Instruction counts
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstructionCounts {
    pub num_load_instructions: u64,
    pub num_store_instructions: u64,
    pub num_shared_mem_instructions: u64,
    pub num_sstarr_instructions: u64,
    pub num_texture_instructions: u64,
    pub num_const_instructions: u64,
    pub num_param_instructions: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sim {
    pub cycles: u64,
    pub instructions: u64,
}

#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone, Serialize)]
pub struct StatsBridge {
    pub accesses: Accesses,
    pub instructions: InstructionCounts,
    pub sim: Sim,
    pub dram: DRAM,

    // per cache stats
    pub l1i_stats: Box<[Cache]>,
    pub l1c_stats: Box<[Cache]>,
    pub l1t_stats: Box<[Cache]>,
    pub l1d_stats: Box<[Cache]>,
    pub l2d_stats: Box<[Cache]>,
}

impl StatsBridge {
    #[must_use]
    pub fn new(num_cores: usize, num_sub_partitions: usize) -> Self {
        Self {
            accesses: Accesses::default(),
            instructions: InstructionCounts::default(),
            sim: Sim::default(),
            dram: DRAM::default(),
            // per cache stats
            l1i_stats: vec![Cache::default(); num_cores].into_boxed_slice(),
            l1c_stats: vec![Cache::default(); num_cores].into_boxed_slice(),
            l1t_stats: vec![Cache::default(); num_cores].into_boxed_slice(),
            l1d_stats: vec![Cache::default(); num_cores].into_boxed_slice(),
            l2d_stats: vec![Cache::default(); num_sub_partitions].into_boxed_slice(),
        }
    }

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
        *cache[cache_index]
            .accesses
            .entry((kind, status))
            .or_insert(0) += num_accesses;
    }
}

// dram stats
impl StatsBridge {
    pub fn set_total_dram_accesses(&mut self, _v: u64) {
        // self.dram.total_accesses = v;
    }
    pub fn set_total_dram_reads(&mut self, v: u64) {
        self.dram.total_reads = v;
    }
    pub fn set_total_dram_writes(&mut self, v: u64) {
        self.dram.total_writes = v;
    }
}

// sim stats
impl StatsBridge {
    pub fn set_sim_cycle(&mut self, v: u64) {
        self.sim.cycles = v;
    }
    pub fn set_sim_instructions(&mut self, v: u64) {
        self.sim.instructions = v;
    }
}

// memory accesses
impl StatsBridge {
    pub fn set_num_mem_write(&mut self, v: u64) {
        self.accesses.num_mem_write = v;
    }
    pub fn set_num_mem_read(&mut self, v: u64) {
        self.accesses.num_mem_read = v;
    }
    pub fn set_num_mem_const(&mut self, v: u64) {
        self.accesses.num_mem_const = v;
    }
    pub fn set_num_mem_texture(&mut self, v: u64) {
        self.accesses.num_mem_texture = v;
    }
    pub fn set_num_mem_read_global(&mut self, v: u64) {
        self.accesses.num_mem_read_global = v;
    }
    pub fn set_num_mem_write_global(&mut self, v: u64) {
        self.accesses.num_mem_write_global = v;
    }
    pub fn set_num_mem_read_local(&mut self, v: u64) {
        self.accesses.num_mem_read_local = v;
    }
    pub fn set_num_mem_write_local(&mut self, v: u64) {
        self.accesses.num_mem_write_local = v;
    }
    pub fn set_num_mem_l2_writeback(&mut self, v: u64) {
        self.accesses.num_mem_l2_writeback = v;
    }
    pub fn set_num_mem_l1_write_allocate(&mut self, v: u64) {
        self.accesses.num_mem_l1_write_allocate = v;
    }
    pub fn set_num_mem_l2_write_allocate(&mut self, v: u64) {
        self.accesses.num_mem_l2_write_allocate = v;
    }
}

// instruction counts
impl StatsBridge {
    pub fn set_num_load_instructions(&mut self, v: u64) {
        self.instructions.num_load_instructions = v;
    }
    pub fn set_num_store_instructions(&mut self, v: u64) {
        self.instructions.num_store_instructions = v;
    }
    pub fn set_num_shared_mem_instructions(&mut self, v: u64) {
        self.instructions.num_shared_mem_instructions = v;
    }
    pub fn set_num_sstarr_instructions(&mut self, v: u64) {
        self.instructions.num_sstarr_instructions = v;
    }
    pub fn set_num_texture_instructions(&mut self, v: u64) {
        self.instructions.num_texture_instructions = v;
    }
    pub fn set_num_const_instructions(&mut self, v: u64) {
        self.instructions.num_const_instructions = v;
    }
    pub fn set_num_param_instructions(&mut self, v: u64) {
        self.instructions.num_param_instructions = v;
    }
}

#[cxx::bridge]
mod ffi {

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
        type StatsBridge;

        fn add_accesses(
            self: &mut StatsBridge,
            cache_kind: CacheKind,
            cache_index: usize,
            kind: u32,
            status: u32,
            failed: bool,
            num_accesses: u64,
        );

        // dram stats
        fn set_total_dram_accesses(self: &mut StatsBridge, v: u64);
        fn set_total_dram_reads(self: &mut StatsBridge, v: u64);
        fn set_total_dram_writes(self: &mut StatsBridge, v: u64);

        // sim stats
        fn set_sim_cycle(self: &mut StatsBridge, v: u64);
        fn set_sim_instructions(self: &mut StatsBridge, v: u64);

        // memory accesses
        fn set_num_mem_write(self: &mut StatsBridge, v: u64);
        fn set_num_mem_read(self: &mut StatsBridge, v: u64);
        fn set_num_mem_const(self: &mut StatsBridge, v: u64);
        fn set_num_mem_texture(self: &mut StatsBridge, v: u64);
        fn set_num_mem_read_global(self: &mut StatsBridge, v: u64);
        fn set_num_mem_write_global(self: &mut StatsBridge, v: u64);
        fn set_num_mem_read_local(self: &mut StatsBridge, v: u64);
        fn set_num_mem_write_local(self: &mut StatsBridge, v: u64);
        fn set_num_mem_l2_writeback(self: &mut StatsBridge, v: u64);
        fn set_num_mem_l1_write_allocate(self: &mut StatsBridge, v: u64);
        fn set_num_mem_l2_write_allocate(self: &mut StatsBridge, v: u64);

        // instruction counts
        fn set_num_load_instructions(self: &mut StatsBridge, v: u64);
        fn set_num_store_instructions(self: &mut StatsBridge, v: u64);
        fn set_num_shared_mem_instructions(self: &mut StatsBridge, v: u64);
        fn set_num_sstarr_instructions(self: &mut StatsBridge, v: u64);
        fn set_num_texture_instructions(self: &mut StatsBridge, v: u64);
        fn set_num_const_instructions(self: &mut StatsBridge, v: u64);
        fn set_num_param_instructions(self: &mut StatsBridge, v: u64);

    }

    unsafe extern "C++" {
        include!("playground-sys/src/ref/bridge/main.hpp");

        type accelsim_bridge = crate::bridge::main::accelsim_bridge;

        fn transfer_stats(self: &accelsim_bridge, stats: &mut StatsBridge);
        fn transfer_dram_stats(self: &accelsim_bridge, stats: &mut StatsBridge);
        fn transfer_general_stats(self: &accelsim_bridge, stats: &mut StatsBridge);
        fn transfer_core_cache_stats(self: &accelsim_bridge, stats: &mut StatsBridge);
        fn transfer_l2d_stats(self: &accelsim_bridge, stats: &mut StatsBridge);

    }
}

pub use ffi::*;
