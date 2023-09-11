use color_eyre::eyre;
use stats::{
    cache::{Access, AccessStat, RequestStatus, ReservationFailure},
    mem::AccessKind,
};
use std::collections::HashMap;
use strum::IntoEnumIterator;
use utils::box_slice;

pub type Stat = (String, u16, String);
pub type Map = indexmap::IndexMap<Stat, f64>;

/// Stats
#[repr(transparent)]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Stats(Map);

impl IntoIterator for Stats {
    type Item = (Stat, f64);
    type IntoIter = indexmap::map::IntoIter<Stat, f64>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl FromIterator<(Stat, f64)> for Stats {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (Stat, f64)>,
    {
        Self(iter.into_iter().collect())
    }
}

impl Stats {
    pub fn find_stat(&self, name: impl AsRef<str>) -> Option<&f64> {
        self.0.iter().find_map(|((_, _, stat_name), value)| {
            if stat_name == name.as_ref() {
                Some(value)
            } else {
                None
            }
        })
    }
}

impl std::ops::Deref for Stats {
    type Target = Map;

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
        for ((current_kernel, running_kcount, stat_name), value) in &stats {
            s.field(
                &format!("{current_kernel} / {running_kcount} / {stat_name}"),
                value,
            );
        }
        s.finish_non_exhaustive()
    }
}

macro_rules! key {
    ($stat:expr) => {
        ("final_kernel".to_string(), 0, $stat.to_string())
    };
}

#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
fn convert_cache_stats(cache_name: &str, stats: &Stats) -> stats::PerCache {
    let mut cache_stats = stats::Cache::default();
    for kind in AccessKind::iter() {
        for reservation_failure in ReservationFailure::iter() {
            let per_cache_stat = stats.get(&key!(format!(
                "{cache_name}_{kind:?}_{reservation_failure:?}"
            )));
            cache_stats.accesses.insert(
                (
                    None,
                    Access((kind, AccessStat::ReservationFailure(reservation_failure))),
                ),
                per_cache_stat.copied().unwrap_or(0.0) as usize,
            );
        }
        for status in RequestStatus::iter() {
            let per_cache_stat = stats.get(&key!(format!("{cache_name}_{kind:?}_{status:?}")));
            cache_stats.accesses.insert(
                (None, Access((kind, AccessStat::Status(status)))),
                per_cache_stat.copied().unwrap_or(0.0) as usize,
            );
        }
    }

    // dbg!(&cache_stats);
    // dbg!(format!("{cache_name}_total_accesses"));
    // dbg!(stats.get(&key!(format!("{cache_name}_total_accesses"))));
    //
    // if let Some(total_accesses) = stats.get(&key!(format!("{cache_name}_total_accesses"))) {
    //     assert_eq!(*total_accesses, cache_stats.total_accesses() as f64);
    // }

    // accelsim only reports the sum of all cache statistics
    stats::PerCache(box_slice![cache_stats])
}

impl TryFrom<Stats> for stats::Stats {
    type Error = eyre::Report;

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn try_from(stats: Stats) -> Result<Self, Self::Error> {
        // this is wrong
        // let accesses: HashMap<AccessKind, u64> = AccessKind::iter()
        //     .map(|kind| {
        //         let hits = stats
        //             .get(&key!(format!("total_core_cache_{kind:?}_HIT")))
        //             .copied()
        //             .unwrap_or(0.0);
        //         let misses = stats
        //             .get(&key!(format!("total_core_cache_{kind:?}_HIT")))
        //             .copied()
        //             .unwrap_or(0.0);
        //         (kind, hits as u64 + misses as u64)
        //     })
        //     .collect();

        let accesses: HashMap<(Option<usize>, AccessKind), u64> = [
            (None, AccessKind::GLOBAL_ACC_R, "num_global_mem_read"),
            (None, AccessKind::LOCAL_ACC_R, "num_local_mem_read"),
            (
                None,
                AccessKind::CONST_ACC_R,
                "num_const_mem_total_accesses",
            ),
            (
                None,
                AccessKind::TEXTURE_ACC_R,
                "num_tex_mem_total_accesses",
            ),
            (None, AccessKind::GLOBAL_ACC_W, "num_global_mem_write"),
            (None, AccessKind::LOCAL_ACC_W, "num_local_mem_write"),
            // the following metrics are not printed out by accelsim (internal?)
            // (AccessKind::L1_WRBK_ACC, 0),
            // (AccessKind::L2_WRBK_ACC, 0),
            // (AccessKind::INST_ACC_R, 0),
            // (AccessKind::L1_WR_ALLOC_R, 0),
            // (AccessKind::L2_WR_ALLOC_R, 0),
        ]
        .into_iter()
        .map(|(alloc_id, kind, stat)| {
            (
                (alloc_id, kind),
                stats.get(&key!(stat)).copied().unwrap_or(0.0) as u64,
            )
        })
        .collect();

        // dbg!(&stats);

        // todo
        let instructions = stats::InstructionCounts::default();

        let l2_data_stats = convert_cache_stats("l2_cache", &stats);
        let l1_inst_stats = convert_cache_stats("l1_inst_cache", &stats);
        let l1_data_stats = convert_cache_stats("l1_data_cache", &stats);
        let l1_const_stats = convert_cache_stats("l1_const_cache", &stats);
        let l1_tex_stats = convert_cache_stats("l1_tex_cache", &stats);

        let total_dram_reads = stats.get(&key!("total_dram_reads")).copied().unwrap_or(0.0) as u64;
        let total_dram_writes = stats
            .get(&key!("total_dram_writes"))
            .copied()
            .unwrap_or(0.0) as u64;
        let dram = stats::DRAM {
            bank_writes: box_slice![box_slice![box_slice![total_dram_writes]]],
            bank_reads: box_slice![box_slice![box_slice![total_dram_reads]]],
            total_bank_writes: box_slice![box_slice![total_dram_writes]],
            total_bank_reads: box_slice![box_slice![total_dram_reads]],
            // we only have total numbers
            num_banks: 1,
            num_cores: 1,
            num_chips: 1,
        };

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
                num_blocks: stats
                    .get(&key!("num_issued_blocks"))
                    .copied()
                    .unwrap_or(0.0) as u64,
            },
            accesses: stats::Accesses(accesses),
            dram,
            instructions,
            l1i_stats: l1_inst_stats,
            l1t_stats: l1_tex_stats,
            l1c_stats: l1_const_stats,
            l1d_stats: l1_data_stats,
            l2d_stats: l2_data_stats,
            stall_dram_full: 0, // todo
        })
    }
}
