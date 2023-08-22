use crate::sync::{Arc, Mutex, RwLock};
use crate::{config, core::WarpIssuer, scoreboard::Scoreboard, warp};
use std::collections::VecDeque;

#[derive(Debug)]
pub struct Scheduler {
    inner: super::Base,
}

impl Scheduler {
    pub fn new(
        id: usize,
        cluster_id: usize,
        core_id: usize,
        warps: Vec<warp::Ref>,
        scoreboard: Arc<RwLock<Scoreboard>>,
        stats: Arc<Mutex<stats::scheduler::Scheduler>>,
        config: Arc<config::GPU>,
    ) -> Self {
        let inner = super::Base::new(id, cluster_id, core_id, warps, scoreboard, stats, config);
        Self { inner }
    }
}

impl Scheduler {
    fn debug_warp_ids(&self) -> Vec<usize> {
        self.inner
            .next_cycle_prioritized_warps
            .iter()
            .map(|(_idx, w)| w.try_lock().warp_id)
            .collect()
    }

    fn debug_dynamic_warp_ids(&self) -> Vec<usize> {
        self.inner
            .next_cycle_prioritized_warps
            .iter()
            .map(|(_idx, w)| w.try_lock().dynamic_warp_id())
            .collect()
    }
}

impl super::Scheduler for Scheduler {
    fn order_warps(&mut self, core: &dyn WarpIssuer) {
        self.inner.order_by_priority(
            super::ordering::Ordering::GREEDY_THEN_PRIORITY_FUNC,
            |lhs: &warp::Ref, rhs: &warp::Ref| {
                super::ordering::sort_warps_by_oldest_dynamic_id(lhs, rhs, core)
            },
        );
    }

    fn add_supervised_warp(&mut self, warp: warp::Ref) {
        self.inner.supervised_warps.push_back(warp);
    }

    fn prioritized_warps(&self) -> &VecDeque<(usize, warp::Ref)> {
        self.inner.prioritized_warps()
    }

    fn issue_to(&mut self, core: &dyn WarpIssuer, cycle: u64) {
        log::debug!(
            "gto scheduler[{}]: BEFORE: prioritized warp ids: {:?}",
            self.inner.id,
            self.debug_warp_ids()
        );
        log::debug!(
            "gto scheduler[{}]: BEFORE: prioritized dynamic warp ids: {:?}",
            self.inner.id,
            self.debug_dynamic_warp_ids()
        );

        self.order_warps(core);

        log::debug!(
            "gto scheduler[{}]: AFTER: prioritized warp ids: {:?}",
            self.inner.id,
            self.debug_warp_ids()
        );
        log::debug!(
            "gto scheduler[{}]: AFTER: prioritized dynamic warp ids: {:?}",
            self.inner.id,
            self.debug_dynamic_warp_ids()
        );

        self.inner.issue_to(core, cycle);
    }
}
