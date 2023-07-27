use color_eyre::eyre;
use std::collections::HashMap;

type StatsMap = indexmap::IndexMap<(String, u16, String), f64>;

/// Stats
#[repr(transparent)]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Stats(StatsMap);

impl Stats {
    pub fn into_inner(self) -> StatsMap {
        self.0
    }
}

impl std::ops::Deref for Stats {
    type Target = StatsMap;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for Stats {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut stats = self.0.clone();
        stats.sort_keys();

        let mut s = f.debug_struct("Stats");
        for ((current_kernel, running_kcount, stat_name), value) in stats.iter() {
            s.field(
                &format!("{} / {} / {}", current_kernel, running_kcount, stat_name),
                value,
            );
        }
        s.finish_non_exhaustive()
    }
}

impl TryFrom<Stats> for stats::Stats {
    type Error = eyre::Report;

    fn try_from(stats: Stats) -> Result<Self, Self::Error> {
        use stats::{
            cache::{AccessStat, RequestStatus, ReservationFailure},
            mem::AccessKind,
        };
        use strum::IntoEnumIterator;

        macro_rules! key {
            ($stat:expr) => {
                ("final_kernel".to_string(), 0, $stat.to_string())
            };
        }

        let accesses: HashMap<AccessKind, u64> = [
            (AccessKind::GLOBAL_ACC_R, 0),
            (AccessKind::LOCAL_ACC_R, 0),
            (AccessKind::CONST_ACC_R, 0),
            (AccessKind::TEXTURE_ACC_R, 0),
            (AccessKind::GLOBAL_ACC_W, 0),
            (AccessKind::LOCAL_ACC_W, 0),
            (AccessKind::L1_WRBK_ACC, 0),
            (AccessKind::L2_WRBK_ACC, 0),
            (AccessKind::INST_ACC_R, 0),
            (AccessKind::L1_WR_ALLOC_R, 0),
            (AccessKind::L2_WR_ALLOC_R, 0),
        ]
        .into_iter()
        .collect();

        let mut l2d_stats = stats::PerCache::default();
        let l2d_total = l2d_stats.entry(0).or_default();
        for kind in AccessKind::iter() {
            for reservation_failure in ReservationFailure::iter() {
                l2d_total.accesses.insert(
                    (kind, AccessStat::ReservationFailure(reservation_failure)),
                    stats
                        .get(&key!(format!("l2_cache_{kind:?}_{reservation_failure:?}")))
                        .copied()
                        .unwrap_or(0.0) as usize,
                );
            }
            for status in RequestStatus::iter() {
                l2d_total.accesses.insert(
                    (kind, AccessStat::Status(status)),
                    stats
                        .get(&key!(format!("l2_cache_{kind:?}_{status:?}")))
                        .copied()
                        .unwrap_or(0.0) as usize,
                );
            }
        }

        // let l2d: HashMap<AccessKind, u64> = [
        //     (AccessKind::GLOBAL_ACC_R, 0),
        //     (AccessKind::LOCAL_ACC_R, 0),
        //     (AccessKind::CONST_ACC_R, 0),
        //     (AccessKind::TEXTURE_ACC_R, 0),
        //     (AccessKind::GLOBAL_ACC_W, 0),
        //     (AccessKind::LOCAL_ACC_W, 0),
        //     (AccessKind::L1_WRBK_ACC, 0),
        //     (AccessKind::L2_WRBK_ACC, 0),
        //     (AccessKind::INST_ACC_R, 0),
        //     (AccessKind::L1_WR_ALLOC_R, 0),
        //     (AccessKind::L2_WR_ALLOC_R, 0),
        // ]
        // .into_iter()
        // .collect();
        let total_dram_reads = stats.get(&key!("total_dram_reads")).copied().unwrap_or(0.0) as u64;
        let total_dram_writes = stats
            .get(&key!("total_dram_writes"))
            .copied()
            .unwrap_or(0.0) as u64;

        Ok(Self {
            sim: stats::Sim {
                cycles: stats
                    .get(&key!("gpu_tot_sim_cycle"))
                    .copied()
                    .unwrap_or(0.0) as u64,
                instructions: stats
                    .get(&key!("gpu_total_instructions"))
                    .copied()
                    .unwrap_or(0.0) as u64,
            },
            accesses: stats::Accesses(accesses),
            dram: stats::DRAM {
                bank_writes: vec![vec![vec![total_dram_writes]]],
                bank_reads: vec![vec![vec![total_dram_reads]]],
                total_bank_writes: vec![vec![total_dram_writes]],
                total_bank_reads: vec![vec![total_dram_reads]],
            },
            instructions: stats::InstructionCounts::default(),
            l1i_stats: stats::PerCache::default(),
            l1t_stats: stats::PerCache::default(),
            l1c_stats: stats::PerCache::default(),
            l1d_stats: stats::PerCache::default(),
            l2d_stats,
        })
    }
}
