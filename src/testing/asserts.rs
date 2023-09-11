use crate::testing::stats::rel_err;

use stats::{
    cache::{Access, RequestStatus},
    mem::AccessKind,
};
use strum::IntoEnumIterator;
use utils::diff;

#[macro_export]
macro_rules! status {
    ($kind:ident, $status:ident) => {{
        Access((AccessKind::$kind, RequestStatus::$status.into()))
    }};
}

#[inline]
pub fn stats_match(
    play_stats: &playground::stats::StatsBridge,
    box_stats: &stats::Stats,
    max_rel_err: Option<f64>,
    abs_threshold: Option<f64>,
    check_cycles: bool,
) {
    let mut play_l1_inst_stats = stats::PerCache::from_iter(play_stats.l1i_stats.to_vec());
    play_l1_inst_stats.shave();
    let mut play_l1_data_stats = stats::PerCache::from_iter(play_stats.l1d_stats.to_vec());
    play_l1_data_stats.shave();
    let mut play_l1_tex_stats = stats::PerCache::from_iter(play_stats.l1t_stats.to_vec());
    play_l1_tex_stats.shave();
    let mut play_l1_const_stats = stats::PerCache::from_iter(play_stats.l1c_stats.to_vec());
    play_l1_const_stats.shave();
    let mut play_l2_data_stats = stats::PerCache::from_iter(play_stats.l2d_stats.to_vec());
    play_l2_data_stats.shave();

    if max_rel_err.is_some() || abs_threshold.is_some() {
        // compare reduced cache stats
        let max_rel_err = max_rel_err.unwrap_or(0.0);
        let abs_threshold = abs_threshold.unwrap_or(0.0);
        {
            let play_l1_inst_stats = play_l1_inst_stats.reduce();
            let box_l1_inst_stats = box_stats.l1i_stats.reduce();
            dbg!(&play_l1_inst_stats, &box_l1_inst_stats);
            if play_l1_inst_stats != box_l1_inst_stats {
                diff::diff!(play: &play_l1_inst_stats, box: &box_l1_inst_stats);
            }
            let rel_err =
                super::stats::cache_rel_err(&play_l1_inst_stats, &box_l1_inst_stats, abs_threshold);
            dbg!(&rel_err);
            assert!(rel_err.into_iter().all(|(_, err)| err <= max_rel_err));
        }
        {
            let play_l1_data_stats = play_l1_data_stats.reduce();
            let box_l1_data_stats = box_stats.l1d_stats.reduce();
            dbg!(&play_l1_data_stats, &box_l1_data_stats);
            if play_l1_data_stats != box_l1_data_stats {
                diff::diff!(play: &play_l1_data_stats, box: &box_l1_data_stats);
            }
            let rel_err =
                super::stats::cache_rel_err(&play_l1_data_stats, &box_l1_data_stats, abs_threshold);
            dbg!(&rel_err);
            assert!(rel_err.into_iter().all(|(_, err)| err <= max_rel_err));
        }
        {
            let play_l1_tex_stats = play_l1_tex_stats.reduce();
            let box_l1_tex_stats = box_stats.l1t_stats.reduce();
            dbg!(&play_l1_tex_stats, &box_l1_tex_stats);
            if play_l1_tex_stats != box_l1_tex_stats {
                diff::diff!(play: &play_l1_tex_stats, box: &box_l1_tex_stats);
            }
            let rel_err =
                super::stats::cache_rel_err(&play_l1_tex_stats, &box_l1_tex_stats, abs_threshold);
            dbg!(&rel_err);
            assert!(rel_err.into_iter().all(|(_, err)| err <= max_rel_err));
        }
        {
            let play_l1_const_stats = play_l1_const_stats.reduce();
            let box_l1_const_stats = box_stats.l1c_stats.reduce();
            dbg!(&play_l1_const_stats, &box_l1_const_stats);
            if play_l1_const_stats != box_l1_const_stats {
                diff::diff!(play: &play_l1_const_stats, box: &box_l1_const_stats);
            }
            let rel_err = super::stats::cache_rel_err(
                &play_l1_const_stats,
                &box_l1_const_stats,
                abs_threshold,
            );
            dbg!(&rel_err);
            assert!(rel_err.into_iter().all(|(_, err)| err <= max_rel_err));
        }
        let l2_rel_err = {
            let box_l2_data_stats = &box_stats.l2d_stats;

            dbg!(&play_l2_data_stats, &box_l2_data_stats);
            if play_l2_data_stats != *box_l2_data_stats {
                // diff::diff!(play: &play_l2_data_stats, box: &box_l2_data_stats);
                // pretty_assertions_sorted::assert_eq!(
                //     play_l2_data_stats
                //         .into_iter()
                //         .enumerate()
                //         .collect::<Vec<_>>(),
                //     box_l2_data_stats
                //         .into_iter()
                //         .enumerate()
                //         .collect::<Vec<_>>(),
                // );
            }

            let mut play_l2_data_stats = play_l2_data_stats.reduce();
            let mut box_l2_data_stats = box_l2_data_stats.reduce();

            dbg!(&play_l2_data_stats, &box_l2_data_stats);
            if play_l2_data_stats != box_l2_data_stats {
                diff::diff!(play: &play_l2_data_stats, box: &box_l2_data_stats);
            }

            // fix some things up
            // for l2_stats in [&mut play_l2_data_stats, &mut box_l2_data_stats] {
            //     for kind in AccessKind::iter() {
            //         let mut hits = l2_stats.accesses[&Access((kind, RequestStatus::HIT.into()))];
            //         // add hit reserved
            //         hits += l2_stats.accesses[&Access((kind, RequestStatus::HIT_RESERVED.into()))];
            //         // add mshr hits
            //         hits += l2_stats.accesses[&Access((kind, RequestStatus::MSHR_HIT.into()))];
            //         l2_stats
            //             .accesses
            //             .insert(Access((kind, RequestStatus::HIT.into())), hits);
            //         // l2_stats
            //         //     .accesses
            //         //     .insert(Access((kind, RequestStatus::HIT_RESERVED.into())), hits);
            //         // l2_stats
            //         //     .accesses
            //         //     .insert(Access((kind, RequestStatus::MSHR_HIT.into())), hits);
            //     }
            // }

            let compare_stats: Vec<_> = AccessKind::iter()
                .flat_map(|kind| {
                    [
                        Access((kind, RequestStatus::HIT.into())),
                        Access((kind, RequestStatus::MISS.into())),
                    ]
                })
                .collect();
            let play_l2_data_stats = stats::Cache::new(
                compare_stats
                    .iter()
                    .map(|stat| ((None, *stat), play_l2_data_stats.num_accesses(stat)))
                    // .map(|stat| (*stat, play_l2_data_stats.accesses[stat]))
                    .collect(),
            );
            let box_l2_data_stats = stats::Cache::new(
                compare_stats
                    .iter()
                    .map(|stat| ((None, *stat), box_l2_data_stats.num_accesses(stat)))
                    // .map(|stat| (*stat, box_l2_data_stats.accesses[stat]))
                    .collect(),
            );
            dbg!(&play_l2_data_stats, &box_l2_data_stats);
            for kind in AccessKind::iter() {
                dbg!(&kind);
                // diff::diff!(
                //     play: play_l2_data_stats.accesses[&Access((kind, RequestStatus::HIT.into()))] +
                //         play_l2_data_stats.accesses[&Access((kind, RequestStatus::MISS.into()))],
                //     box: box_l2_data_stats.accesses[&Access((kind, RequestStatus::HIT.into()))] +
                //         box_l2_data_stats.accesses[&Access((kind, RequestStatus::MISS.into()))],
                // );
                diff::diff!(
                    play: play_l2_data_stats.num_accesses(&Access((kind, RequestStatus::HIT.into()))) +
                        play_l2_data_stats.num_accesses(&Access((kind, RequestStatus::MISS.into()))),
                    box: box_l2_data_stats.num_accesses(&Access((kind, RequestStatus::HIT.into()))) +
                        box_l2_data_stats.num_accesses(&Access((kind, RequestStatus::MISS.into()))),
                );
            }

            if play_l2_data_stats != box_l2_data_stats {
                diff::diff!(play: &play_l2_data_stats, box: &box_l2_data_stats);
            }
            let rel_err =
                super::stats::cache_rel_err(&play_l2_data_stats, &box_l2_data_stats, abs_threshold);
            dbg!(&rel_err);

            let play_l2_sum: usize = play_l2_data_stats.as_ref().values().sum();
            let box_l2_sum: usize = box_l2_data_stats.as_ref().values().sum();
            diff::diff!(play_sum: play_l2_sum, box_sum: box_l2_sum);

            rel_err
        };

        let dram_rel_err = {
            // compare DRAM stats
            let box_dram_stats = playground::stats::DRAM::from(box_stats.dram.clone());
            dbg!(&play_stats.dram, &box_dram_stats);
            if play_stats.dram != box_dram_stats {
                diff::diff!(play: &play_stats.dram, box: &box_dram_stats);
            }
            let rel_err =
                super::stats::dram_rel_err(&play_stats.dram, &box_dram_stats, abs_threshold);
            dbg!(&rel_err);
            rel_err
        };

        assert!(l2_rel_err.into_iter().all(|(_, err)| err <= max_rel_err));
        assert!(dram_rel_err.into_iter().all(|(_, err)| err <= max_rel_err));
    } else {
        dbg!(&play_l1_inst_stats, &box_stats.l1i_stats);
        diff::assert_eq!(play: &play_l1_inst_stats, box: &box_stats.l1i_stats);
        dbg!(&play_l1_data_stats, &box_stats.l1d_stats);
        diff::assert_eq!( play: &play_l1_data_stats, box: &box_stats.l1d_stats);
        dbg!(&play_l1_tex_stats, &box_stats.l1t_stats);
        diff::assert_eq!( play: &play_l1_tex_stats, box: &box_stats.l1t_stats);
        dbg!(&play_l1_const_stats, &box_stats.l1c_stats);
        diff::assert_eq!( play: &play_l1_const_stats, box: &box_stats.l1c_stats);
        dbg!(&play_l2_data_stats, &box_stats.l2d_stats);
        diff::assert_eq!( play: &play_l2_data_stats, box: &box_stats.l2d_stats);

        // compare DRAM stats
        let box_dram_stats = playground::stats::DRAM::from(box_stats.dram.clone());
        dbg!(&play_stats.dram, &box_dram_stats);
        diff::assert_eq!(play: &play_stats.dram, box: &box_dram_stats);
    }

    // compare accesses
    let box_accesses = playground::stats::Accesses::from(box_stats.accesses.clone());
    dbg!(&play_stats.accesses, &box_stats.accesses);
    diff::assert_eq!(play: play_stats.accesses, box: box_accesses);

    // compare instruction stats
    let box_instructions =
        playground::stats::InstructionCounts::from(box_stats.instructions.clone());
    dbg!(&play_stats.instructions, &box_instructions);
    diff::assert_eq!(play: &play_stats.instructions, box: &box_instructions);

    // compate simulation stats
    let box_sim_stats = playground::stats::Sim::from(box_stats.sim.clone());
    dbg!(&play_stats.sim, &box_sim_stats);

    if check_cycles {
        diff::assert_eq!(play: &play_stats.sim, box: &box_sim_stats);
    } else {
        diff::diff!(play: &play_stats.sim, box: &box_sim_stats);
        dbg!(rel_err(play_stats.sim.cycles, box_sim_stats.cycles, 20.0));
        assert_eq!(play_stats.sim.instructions, box_sim_stats.instructions);
    }

    // this uses our custom PartialEq::eq implementation
    // assert_eq!(&play_stats, &box_stats);
    // assert!(false);
}
