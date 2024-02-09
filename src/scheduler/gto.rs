use crate::sync::{Arc, Mutex, RwLock};
use crate::{config, core::WarpIssuer, scoreboard::Scoreboard, warp};
// use std::collections::VecDeque;
use smallvec::SmallVec;

#[derive(Debug)]
pub struct Scheduler {
    // pub struct Scheduler<'a> {
    pub inner: super::Base,
    // inner: super::Base<'a>,
}

impl Scheduler {
    // impl<'a> Scheduler<'a> {
    pub fn new(
        id: usize,
        global_core_id: usize,
        cluster_id: usize,
        // warps: Vec<warp::Ref>,
        // scoreboard: Arc<RwLock<Scoreboard>>,
        // stats: Arc<Mutex<stats::scheduler::Scheduler>>,
        config: Arc<config::GPU>,
    ) -> Self {
        // let inner = super::Base::new(id, cluster_id, core_id, warps, scoreboard, stats, config);
        let inner = super::Base::new(id, global_core_id, cluster_id, config);
        Self { inner }
    }
}

impl Scheduler {
    #[allow(dead_code)]
    fn debug_warp_ids(&self, warps: &[(usize, &mut warp::Warp)]) -> Vec<usize> {
        // self.inner
        //     .next_cycle_prioritized_warps
        warps
            .iter()
            .map(
                |(_idx, w)| w.warp_id, // w.try_lock().warp_id
            )
            .collect()
    }

    #[allow(dead_code)]
    fn debug_dynamic_warp_ids(&self, warps: &[(usize, &mut warp::Warp)]) -> Vec<usize> {
        // self.inner
        //     .next_cycle_prioritized_warps
        warps
            .iter()
            .map(|(_idx, w)| w.dynamic_warp_id())
            // .map(|(_idx, w)| w.try_lock().dynamic_warp_id())
            .collect()
    }
}

impl<I> super::ScheduleWarps<I> for Scheduler
where
    I: WarpIssuer,
{
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

    // fn prioritized_warps(&self) -> &VecDeque<(usize, warp::Ref)> {
    //     self.inner.prioritized_warps()
    // }

    // fn issue_to(&mut self, core: &mut dyn WarpIssuer, mut warps: Vec<&mut warp::Warp>, cycle: u64) {
    #[inline]
    fn issue_to<'a>(
        &mut self,
        core: &mut I,
        // core: &mut dyn WarpIssuer,
        warps: &mut [(usize, &'a mut warp::Warp)],
        // warps: impl Iterator<Item = &'a mut warp::Warp>,
        // mut warps: SmallVec<[&mut warp::Warp; 64]>,
        cycle: u64,
    ) {
        log::debug!(
            // eprintln!(
            "gto scheduler[{}, core {}]: BEFORE: prioritized warp ids: {:?}",
            self.inner.id,
            self.inner.global_core_id,
            // <self::Scheduler as super::Scheduler<I>>::prioritized_warp_ids(self)
            self.prioritized_warp_ids()
                .iter()
                .map(|(_, dyn_warp_id)| dyn_warp_id)
                .collect::<Vec<_>>(),
            // self.debug_dynamic_warp_ids(&self.prioritized_warp_ids()),
            // self.debug_warp_ids(warps)
        );

        // hack: work around removing memory barriers
        for (_, warp) in warps.iter_mut() {
            if warp.waiting_for_memory_barrier && !core.warp_waiting_at_mem_barrier(&*warp) {
                // clear memory barrier
                warp.waiting_for_memory_barrier = false;
                // todo: inform the core that the memory barrier is reached
                //if (m_gpu->get_config().flush_l1()) {
                // Mahmoud fixed this on Nov 2019
                // Invalidate L1 cache
                // Based on Nvidia Doc, at MEM barrier, we have to
                //(1) wait for all pending writes till they are acked
                //(2) invalidate L1 cache to ensure coherence and avoid reading
                // stall
                // data
                // cache_invalidate();
                // TO DO: you need to stall the SM for 5k cycles.
                // }
            }
        }

        // eprintln!(
        //     // log::debug!(
        //     "gto scheduler[{}]: BEFORE: prioritized dynamic warp ids: {:?}",
        //     self.inner.id,
        //     self.debug_dynamic_warp_ids(warps)
        // );

        // eprintln!(
        log::debug!(
            "gto scheduler[{}, core {}]: last issued from {} (index {})",
            self.inner.id,
            self.inner.global_core_id,
            warps
                .get(self.inner.last_supervised_issued_idx)
                .map(|(_, w)| w.dynamic_warp_id)
                .unwrap_or_default(),
            self.inner.last_supervised_issued_idx
        );

        // let prioritized_warps: SmallVec<[_; 64]> = crate::timeit!(
        crate::timeit!(
            "core::issue::order_by_priority",
            self.inner.order_by_priority(
                warps,
                // warps,
                super::ordering::Ordering::GREEDY_THEN_PRIORITY_FUNC,
                // |lhs: &(usize, warp::Ref), rhs: &(usize, warp::Ref)| {
                core,
                |lhs: &(usize, &mut warp::Warp), rhs: &(usize, &mut warp::Warp)| {
                    super::ordering::sort_warps_by_oldest_dynamic_id(lhs, rhs, core)
                },
            ) // .collect()
        );

        // log::debug!(
        //     "gto scheduler[{}]: AFTER: prioritized dynamic warp ids: {:?}",
        //     self.inner.id,
        //     self.debug_dynamic_warp_ids(&prioritized_warps)
        // );

        self.inner.prioritized_warps_ids.clear();
        self.inner.prioritized_warps_ids.extend(
            // prioritized_warps
            warps
                .iter()
                .map(|(_, warp)| (warp.warp_id, warp.dynamic_warp_id)),
        );

        // #[cfg(debug_assertions)]
        // {
        //     let left = prioritized_warps
        //         .iter()
        //         .map(|(_, warp)| (warp.warp_id, warp.dynamic_warp_id))
        //         .collect::<Vec<_>>();
        //     let right = self.inner.prioritized_warp_ids();
        //     let valid = left.len().min(right.len());
        //     debug_assert_eq!(left[..valid], right[..valid]);
        //     // if self.inner.core_id == 0 {
        //     //     dbg!(right);
        //     // }
        // }

        log::debug!(
            // eprintln!(
            "gto scheduler[{}, core {}]: AFTER: prioritized warp ids: {:?}",
            self.inner.id,
            self.inner.global_core_id,
            // self.prioritized_warp_ids(),
            // <self::Scheduler as super::Scheduler<I>>::prioritized_warp_ids(self)
            self.prioritized_warp_ids()
                .iter()
                .map(|(_, dyn_warp_id)| dyn_warp_id)
                .collect::<Vec<_>>(),
            // self.debug_dynamic_warp_ids(&warps),
            // self.inner.prioritized_warps_ids,
            // self.debug_warp_ids(&prioritized_warps)
        );

        // #[cfg(feature = "timings")]
        // crate::TIMINGS
        // .lock()
        // .entry("serial::total")
        // .or_default()
        // .add(start.elapsed());

        crate::timeit!(
            "core::issue::issue_warps",
            self.inner.issue_to(core, warps, cycle)
        );
    }
}

impl Scheduler {
    pub fn prioritized_warp_ids(&self) -> &Vec<(usize, usize)> {
        self.inner.prioritized_warp_ids()
    }
}
