use crate::sync::{Arc, Mutex, RwLock};
use crate::{config, core::WarpIssuer, scoreboard::Scoreboard, warp};
// use std::collections::VecDeque;

#[derive(Debug)]
pub struct Scheduler {
    // pub struct Scheduler<'a> {
    inner: super::Base,
    // inner: super::Base<'a>,
}

impl Scheduler {
    // impl<'a> Scheduler<'a> {
    pub fn new(
        id: usize,
        cluster_id: usize,
        core_id: usize,
        // warps: Vec<warp::Ref>,
        // scoreboard: Arc<RwLock<Scoreboard>>,
        // stats: Arc<Mutex<stats::scheduler::Scheduler>>,
        config: Arc<config::GPU>,
    ) -> Self {
        // let inner = super::Base::new(id, cluster_id, core_id, warps, scoreboard, stats, config);
        let inner = super::Base::new(
            id, cluster_id, core_id, // scoreboard,
            config,
        );
        Self { inner }
    }
}

// impl Scheduler {
//     // impl<'a> Scheduler<'a> {
//     fn debug_warp_ids(&self, warps: &[(usize, &mut warp::Warp)]) -> Vec<usize> {
//         // self.inner
//         //     .next_cycle_prioritized_warps
//         warps
//             .iter()
//             .map(
//                 |(_idx, w)| w.warp_id, // w.try_lock().warp_id
//             )
//             .collect()
//     }
//
//     fn debug_dynamic_warp_ids(&self, warps: &[(usize, &mut warp::Warp)]) -> Vec<usize> {
//         // self.inner
//         //     .next_cycle_prioritized_warps
//         warps
//             .iter()
//             .map(|(_idx, w)| w.dynamic_warp_id())
//             // .map(|(_idx, w)| w.try_lock().dynamic_warp_id())
//             .collect()
//     }
// }

impl super::Scheduler for Scheduler {
    // impl<'a> super::Scheduler for Scheduler<'a> {
    // fn order_warps(&mut self, core: &dyn WarpIssuer, warps: &[&warp::Warp]) {
    //     self.inner.order_by_priority(
    //         warps,
    //         super::ordering::Ordering::GREEDY_THEN_PRIORITY_FUNC,
    //         // |lhs: &(usize, warp::Ref), rhs: &(usize, warp::Ref)| {
    //         |lhs: &(usize, &warp::Warp), rhs: &(usize, &warp::Warp)| {
    //             super::ordering::sort_warps_by_oldest_dynamic_id(lhs, rhs, core)
    //         },
    //     );
    // }

    // fn add_supervised_warp(&mut self, warp: warp::Ref) {
    // fn add_supervised_warp(&mut self, warp: &'a warp::Warp) {
    //     self.inner.supervised_warps.push_back(warp);
    // }

    fn prioritized_warp_ids(&self) -> &Vec<(usize, usize)> {
        self.inner.prioritized_warp_ids()
    }

    // fn prioritized_warps(&self) -> &VecDeque<(usize, warp::Ref)> {
    //     self.inner.prioritized_warps()
    // }

    fn issue_to(&mut self, core: &mut dyn WarpIssuer, mut warps: Vec<&mut warp::Warp>, cycle: u64) {
        log::debug!(
            // eprintln!(
            "gto scheduler[{}, core {}]: BEFORE: prioritized warp ids: {:?}",
            self.inner.id,
            self.inner.core_id,
            self.prioritized_warp_ids(),
            // self.debug_warp_ids(warps)
        );

        // hack: work around removing memory barriers
        for warp in warps.iter_mut() {
            if warp.waiting_for_memory_barrier && !core.warp_waiting_at_mem_barrier(&*warp) {
                // clear memory barrier
                warp.waiting_for_memory_barrier = false;
            }
        }

        // log::debug!(
        //     "gto scheduler[{}]: BEFORE: prioritized dynamic warp ids: {:?}",
        //     self.inner.id,
        //     self.debug_dynamic_warp_ids(warps)
        // );
        log::debug!(
            "gto scheduler[{}, core {}]: last issued from {} (index {})",
            self.inner.id,
            self.inner.core_id,
            warps
                .get(self.inner.last_supervised_issued_idx)
                .map(|w| w.dynamic_warp_id)
                .unwrap_or_default(),
            self.inner.last_supervised_issued_idx
        );

        // self.order_warps(core, warps);
        let prioritized_warps = self.inner.order_by_priority(
            warps,
            super::ordering::Ordering::GREEDY_THEN_PRIORITY_FUNC,
            // |lhs: &(usize, warp::Ref), rhs: &(usize, warp::Ref)| {
            core,
            |lhs: &(usize, &mut warp::Warp), rhs: &(usize, &mut warp::Warp)| {
                super::ordering::sort_warps_by_oldest_dynamic_id(lhs, rhs, core)
            },
        );

        // log::debug!(
        //     "gto scheduler[{}]: AFTER: prioritized dynamic warp ids: {:?}",
        //     self.inner.id,
        //     self.debug_dynamic_warp_ids(&prioritized_warps)
        // );

        self.inner.prioritized_warps_ids.clear();
        self.inner.prioritized_warps_ids.extend(
            prioritized_warps
                .iter()
                .map(|(_, warp)| (warp.warp_id, warp.dynamic_warp_id)),
        );

        #[cfg(debug_assertions)]
        {
            let left = prioritized_warps
                .iter()
                .map(|(_, warp)| (warp.warp_id, warp.dynamic_warp_id))
                .collect::<Vec<_>>();
            let right = self.inner.prioritized_warp_ids();
            let valid = left.len().min(right.len());
            debug_assert_eq!(left[..valid], right[..valid]);
            // if self.inner.core_id == 0 {
            //     dbg!(right);
            // }
        }

        log::debug!(
            // eprintln!(
            "gto scheduler[{}, core {}]: AFTER: prioritized warp ids: {:?}",
            self.inner.id,
            self.inner.core_id,
            self.prioritized_warp_ids(),
            // self.debug_warp_ids(&prioritized_warps)
        );

        self.inner.issue_to(core, prioritized_warps, cycle);
    }
}
