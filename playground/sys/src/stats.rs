pub use crate::bridge::stats::StatsBridge as Stats;
pub use crate::bridge::stats::*;
use stats::box_slice;

impl From<Cache> for stats::Cache {
    fn from(stats: crate::bridge::stats::Cache) -> Self {
        Self {
            accesses: stats
                .accesses
                .into_iter()
                .map(|((access_kind, access_stat), count)| {
                    (
                        (
                            None,
                            stats::cache::Access((access_kind.into(), access_stat.into())),
                        ),
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
                panic!("bad mem access type: {other:?}")
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
                panic!("bad cache request status: {other:?}")
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
                panic!("bad cache request status: {other:?}")
            }
        }
    }
}

impl From<stats::mem::Accesses> for Accesses {
    fn from(other: stats::mem::Accesses) -> Self {
        let num_mem_write = other.num_writes();
        let num_mem_read = other.num_reads();
        Self {
            num_mem_write,
            num_mem_read,
            num_mem_const: other.num_accesses(stats::mem::AccessKind::CONST_ACC_R),
            num_mem_texture: other.num_accesses(stats::mem::AccessKind::TEXTURE_ACC_R),
            num_mem_read_global: other.num_accesses(stats::mem::AccessKind::GLOBAL_ACC_R),
            num_mem_write_global: other.num_accesses(stats::mem::AccessKind::GLOBAL_ACC_W),
            num_mem_read_local: other.num_accesses(stats::mem::AccessKind::LOCAL_ACC_R),
            num_mem_write_local: other.num_accesses(stats::mem::AccessKind::LOCAL_ACC_W),
            num_mem_l2_writeback: other.num_accesses(stats::mem::AccessKind::L2_WRBK_ACC),
            num_mem_l1_write_allocate: other.num_accesses(stats::mem::AccessKind::L1_WR_ALLOC_R),
            num_mem_l2_write_allocate: other.num_accesses(stats::mem::AccessKind::L2_WR_ALLOC_R),
            // num_mem_texture: other
            //     .get(&stats::mem::AccessKind::TEXTURE_ACC_R)
            //     .copied()
            //     .unwrap_or(0),
            // num_mem_read_global: other
            //     .get(&stats::mem::AccessKind::GLOBAL_ACC_R)
            //     .copied()
            //     .unwrap_or(0),
            // num_mem_write_global: other
            //     .get(&stats::mem::AccessKind::GLOBAL_ACC_W)
            //     .copied()
            //     .unwrap_or(0),
            // num_mem_read_local: other
            //     .get(&stats::mem::AccessKind::LOCAL_ACC_R)
            //     .copied()
            //     .unwrap_or(0),
            // num_mem_write_local: other
            //     .get(&stats::mem::AccessKind::LOCAL_ACC_W)
            //     .copied()
            //     .unwrap_or(0),
            // num_mem_l2_writeback: other
            //     .get(&stats::mem::AccessKind::L2_WRBK_ACC)
            //     .copied()
            //     .unwrap_or(0),
            // num_mem_l1_write_allocate: other
            //     .get(&stats::mem::AccessKind::L1_WR_ALLOC_R)
            //     .copied()
            //     .unwrap_or(0),
            // num_mem_l2_write_allocate: other
            //     .get(&stats::mem::AccessKind::L2_WR_ALLOC_R)
            //     .copied()
            //     .unwrap_or(0),
        }
    }
}

impl From<Accesses> for stats::mem::Accesses {
    fn from(_acc: Accesses) -> Self {
        Self::default()
    }
}

impl From<stats::dram::DRAM> for DRAM {
    fn from(other: stats::dram::DRAM) -> Self {
        Self {
            total_reads: other.total_reads(),
            total_writes: other.total_writes(),
        }
    }
}

impl From<DRAM> for stats::dram::DRAM {
    fn from(dram: DRAM) -> Self {
        Self {
            bank_writes: box_slice![box_slice![box_slice![dram.total_writes]]],
            bank_reads: box_slice![box_slice![box_slice![dram.total_reads]]],
            total_bank_writes: box_slice![box_slice![dram.total_writes]],
            total_bank_reads: box_slice![box_slice![dram.total_reads]],
            // we only have total numbers
            num_banks: 1,
            num_cores: 1,
            num_chips: 1,
        }
    }
}

impl From<stats::instructions::InstructionCounts> for InstructionCounts {
    fn from(other: stats::instructions::InstructionCounts) -> Self {
        use stats::instructions::MemorySpace;
        let num_global_loads = other.num_instructions(MemorySpace::Global, false);
        let num_local_loads = other.num_instructions(MemorySpace::Local, false);
        let num_global_stores = other.num_instructions(MemorySpace::Global, true);
        let num_local_stores = other.num_instructions(MemorySpace::Local, true);
        // let num_global_loads = other
        //     .get(&(stats::instructions::MemorySpace::Global, false))
        //     .copied()
        //     .unwrap_or(0);
        // let num_local_loads = other
        //     .get(&(stats::instructions::MemorySpace::Local, false))
        //     .copied()
        //     .unwrap_or(0);
        // let num_global_stores = other
        //     .get(&(stats::instructions::MemorySpace::Global, true))
        //     .copied()
        //     .unwrap_or(0);
        // let num_local_stores = other
        //     .get(&(stats::instructions::MemorySpace::Local, true))
        //     .copied()
        //     .unwrap_or(0);
        let num_shmem = other.get_total(MemorySpace::Shared);
        let num_tex = other.get_total(MemorySpace::Texture);
        let num_const = other.get_total(MemorySpace::Constant);

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

impl From<InstructionCounts> for stats::instructions::InstructionCounts {
    fn from(_inst: InstructionCounts) -> Self {
        // TODO

        Self::default()
        // num_load_instructions: inst.num_load_instructions,
        // num_store_instructions: inst.num_store_instructions,
        // num_shared_mem_instructions: inst.num_shared_mem_instructions,
        // num_sstarr_instruction: inst.num_sstarr_instructions,
        // num_texture_instructions: inst.num_texture_instructions,
        // num_const_instructions: inst.num_const_instructions,
        // num_param_instructions: inst.num_param_instructions,
    }
}

impl From<stats::sim::Sim> for Sim {
    fn from(sim: stats::sim::Sim) -> Self {
        Self {
            cycles: sim.cycles,
            instructions: sim.instructions,
            num_blocks: sim.num_blocks,
        }
    }
}

impl From<Sim> for stats::sim::Sim {
    fn from(sim: Sim) -> Self {
        Self {
            kernel_name: "".to_string(),
            kernel_launch_id: 0,
            cycles: sim.cycles,
            instructions: sim.instructions,
            num_blocks: sim.num_blocks,
        }
    }
}

impl From<StatsBridge> for stats::Stats {
    fn from(stats: StatsBridge) -> Self {
        Self {
            accesses: stats.accesses.into(),
            instructions: stats.instructions.into(),
            sim: stats.sim.into(),
            dram: stats.dram.into(),
            l1i_stats: stats.l1i_stats.iter().cloned().collect(),
            l1c_stats: stats.l1c_stats.iter().cloned().collect(),
            l1t_stats: stats.l1t_stats.iter().cloned().collect(),
            l1d_stats: stats.l1d_stats.iter().cloned().collect(),
            l2d_stats: stats.l2d_stats.iter().cloned().collect(),
            stall_dram_full: 0,
        }
    }
}

// impl PartialEq<stats::Stats> for StatsBridge {
//     fn eq(&self, other: &stats::Stats) -> bool {
//         // use stats::ConvertHashMap;
//         if !(stats::PerCache(self.l1i_stats.clone().convert()) == other.l1i_stats)
//             && (stats::PerCache(self.l1d_stats.clone().convert()) == other.l1d_stats)
//             && (stats::PerCache(self.l1t_stats.clone().convert()) == other.l1t_stats)
//             && (stats::PerCache(self.l1c_stats.clone().convert()) == other.l1c_stats)
//             && (stats::PerCache(self.l2d_stats.clone().convert()) == other.l2d_stats)
//         {
//             return false;
//         }
//
//         if self.accesses != Accesses::from(other.accesses.clone()) {
//             return false;
//         }
//
//         if self.dram != DRAM::from(other.dram.clone()) {
//             return false;
//         }
//
//         if self.instructions != InstructionCounts::from(other.instructions.clone()) {
//             return false;
//         }
//
//         self.sim == Sim::from(other.sim.clone())
//     }
// }
