use crate::warp;
use smallvec::SmallVec;

// use crate::sync::{Arc, Mutex};

// #[must_use]
// pub fn all_different<T>(values: &[Arc<Mutex<T>>]) -> bool {
//     for (vi, v) in values.iter().enumerate() {
//         for (vii, vv) in values.iter().enumerate() {
//             let should_be_equal = vi == vii;
//             let are_equal = Arc::ptr_eq(v, vv);
//             if should_be_equal && !are_equal {
//                 return false;
//             }
//             if !should_be_equal && are_equal {
//                 return false;
//             }
//         }
//     }
//     true
// }

pub fn sort_warps_by_oldest_dynamic_id(
    // lhs: &(usize, warp::Ref),
    // rhs: &(usize, warp::Ref),
    lhs: &(usize, &mut warp::Warp),
    rhs: &(usize, &mut warp::Warp),
    issuer: &dyn crate::core::WarpIssuer,
) -> std::cmp::Ordering {
    let lhs_warp = &lhs.1;
    let rhs_warp = &rhs.1;
    // let mut lhs_warp = lhs.1.try_lock();
    // let mut rhs_warp = rhs.1.try_lock();
    let lhs_blocked = lhs_warp.done_exit()
        || lhs_warp.waiting()
        || issuer.warp_waiting_at_barrier(lhs_warp.warp_id)
        // || issuer.warp_waiting_at_mem_barrier(lhs_warp.warp_id);
    || issuer.warp_waiting_at_mem_barrier(&lhs_warp);
    let rhs_blocked = rhs_warp.done_exit()
        || rhs_warp.waiting()
        || issuer.warp_waiting_at_barrier(rhs_warp.warp_id)
        // || issuer.warp_waiting_at_mem_barrier(rhs_warp.warp_id);
    || issuer.warp_waiting_at_mem_barrier(&rhs_warp);

    match (lhs_blocked, rhs_blocked) {
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (true, true) => {
            // both blocked
            (lhs.0).cmp(&(rhs.0))
        }
        (false, false) => {
            // both unblocked
            (lhs_warp.dynamic_warp_id(), lhs.0).cmp(&(rhs_warp.dynamic_warp_id(), rhs.0))
        }
    }

    // the following is sufficient when STABLE sorting is used (requires allocation).
    // if lhs_blocked {
    //     std::cmp::Ordering::Greater
    // } else if rhs_blocked {
    //     std::cmp::Ordering::Less
    // } else {
    //     (lhs_warp.dynamic_warp_id(), lhs.0).cmp(&(rhs_warp.dynamic_warp_id(), rhs.0))
    // }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ordering {
    // The item that issued last is prioritized first then the
    // sorted result of the priority_function
    GREEDY_THEN_PRIORITY_FUNC = 0,
    // No greedy scheduling based on last to issue.
    //
    // Only the priority function determines priority
    PRIORITY_FUNC_ONLY,
}

impl super::Base {
    // impl<'a> super::Base<'a> {
    pub fn order_by_priority<'a, F, const N: usize>(
        &self,
        // &mut self,
        warps: SmallVec<[&'a mut warp::Warp; N]>,
        // warps: SmallVec<[&'a mut warp::Warp; N]>,
        // warps: Vec<&'a mut warp::Warp>,
        // warps: &mut [&'a mut warp::Warp],
        // warps: &'b [&'a mut warp::Warp],
        ordering: Ordering,
        _core: &dyn crate::core::WarpIssuer,
        priority_func: F,
    ) -> impl Iterator<Item = (usize, &'a mut warp::Warp)>
    // ) -> Vec<(usize, &'a mut warp::Warp)>
    where
        F: FnMut(&(usize, &mut warp::Warp), &(usize, &mut warp::Warp)) -> std::cmp::Ordering,
        // F: FnMut(&(usize, warp::Ref), &(usize, warp::Ref)) -> std::cmp::Ordering,
    {
        let num_warps_to_add = warps.len();
        let mut out: SmallVec<[(usize, &mut warp::Warp); 64]> = SmallVec::new();
        // let mut out: Vec<(usize, &mut warp::Warp)> = Vec::new();
        // let out = &mut self.next_cycle_prioritized_warps;

        // debug_assert!(num_warps_to_add <= self.warps.len());
        // out.clear();

        // debug_assert!(all_different(self.supervised_warps.make_contiguous()));
        //
        // let mut last_issued_iter = warps
        //     // self
        //     // .supervised_warps
        //     .iter()
        //     .enumerate()
        //     .skip(self.last_supervised_issued_idx);

        // self.last_supervised_issued_idx

        #[cfg(debug_assertions)]
        {
            let mut last_issued_iter = warps
                // self
                // .supervised_warps
                .iter()
                .enumerate()
                .skip(self.last_supervised_issued_idx);

            let last_issued_idx: Option<usize> = last_issued_iter.next().map(|(idx, _)| idx);
            drop(last_issued_iter);

            if let Some(last_issued_idx) = last_issued_idx {
                assert_eq!(self.last_supervised_issued_idx, last_issued_idx);
            }
        }

        // dbg!(self.core_id, self.last_supervised_issued_idx);

        // let last_issued_ids: Vec<_> = last_issued_iter.map(|(idx, _)| idx).collect();
        // if last_issued_ids.len() > 0 {
        //     assert_eq!(self.last_supervised_issued_idx, last_issued_ids[0]);
        // }

        // debug_assert!(all_different(&self.warps));

        // sort a copy of the supervised warps for stability
        // self.supervised_warps_sorted.clear();
        // self.supervised_warps_sorted
        //     .extend(self.supervised_warps.iter().cloned().enumerate());
        //
        // self.supervised_warps_sorted.sort_unstable_by(priority_func);

        // let mut warps_sorted: Vec<(usize, &mut warp::Warp)> = warps
        let mut warps_sorted = warps
            .into_iter()
            .enumerate()
            // .map(|(idx, warp)| (idx, *warp))
            .collect::<SmallVec<[(usize, &mut warp::Warp); 64]>>();

        // use crate::scoreboard::Access;
        // let (_first_idx, first_warp) = &warps_sorted[0];
        // let (_second_idx, second_warp) = &warps_sorted[1];
        // log::warn!(
        //     "first: {:?} done={} waiting={} at barrier={} at mem barrier={} (has barrier={} outstanding stores={})",
        //     (first_warp.warp_id, first_warp.dynamic_warp_id),
        //     first_warp.done_exit(),
        //     first_warp.waiting(),
        //     core.warp_waiting_at_barrier(first_warp.warp_id),
        //     // core.warp_waiting_at_mem_barrier(first_warp.warp_id),
        //     core.warp_waiting_at_mem_barrier(first_warp),
        //     first_warp.waiting_for_memory_barrier,
        //     self.scoreboard.try_read().pending_writes(first_warp.warp_id).len()
        // );
        // log::warn!(
        //     "second: {:?} done={} waiting={} at barrier={} at mem barrier={} (has barrier={} outstanding stores={})",
        //     (second_warp.warp_id, second_warp.dynamic_warp_id),
        //     second_warp.done_exit(),
        //     second_warp.waiting(),
        //     core.warp_waiting_at_barrier(second_warp.warp_id),
        //     // core.warp_waiting_at_mem_barrier(second_warp.warp_id),
        //     core.warp_waiting_at_mem_barrier(second_warp),
        //     second_warp.waiting_for_memory_barrier,
        //     self.scoreboard.try_read().pending_writes(second_warp.warp_id).len()
        // );
        //
        // log::warn!(
        //     "{:?} vs {:?}: {:?}",
        //     (first_warp.warp_id, first_warp.dynamic_warp_id),
        //     (second_warp.warp_id, second_warp.dynamic_warp_id),
        //     // (warps_sorted[0].0, warps_sorted[0].1.dynamic_warp_id),
        //     // (warps_sorted[1].0, warps_sorted[1].1.dynamic_warp_id),
        //     priority_func(&warps_sorted[0], &warps_sorted[1]),
        //     // sort_warps_by_oldest_dynamic_id(&warps[0], &warps[1], core)
        // );

        // warps.clone().into_iter().enumerate().copied().collect();
        // .iter().enumerate()..collect();
        warps_sorted.sort_unstable_by(priority_func);

        log::debug!(
            "gto scheduler[{}, core {}]: greedy={:?} sorted by priority: {:?}",
            self.id,
            self.global_core_id,
            last_issued_idx,
            warps_sorted
                .iter()
                .map(|(_, w)| (w.warp_id, w.dynamic_warp_id))
                .collect::<Vec<_>>(),
        );

        //
        // debug_assert!(all_different(
        //     &self
        //         .supervised_warps_sorted
        //         .clone()
        //         .into_iter()
        //         .map(|(_, w)| w)
        //         .collect::<Vec<_>>()
        // ));
        //
        match ordering {
            Ordering::GREEDY_THEN_PRIORITY_FUNC => {
                // move greedy warp to the start
                // let greedy_warp = last_issued_iter.next();
                // if let Some((greedy_idx, _)) = greedy_warp {
                if let Some(greedy_idx) = last_issued_idx {
                    let greedy_idx = warps_sorted.iter().position(|(idx, _)| greedy_idx == *idx);
                    if let Some(greedy_idx) = greedy_idx {
                        let greedy = warps_sorted.remove(greedy_idx);
                        log::debug!("added greedy warp: {}", greedy.1.dynamic_warp_id);
                        warps_sorted.insert(0, greedy);
                    }
                }

                // let greedy_warp = last_issued_iter.next();
                // if let Some((idx, warp)) = greedy_warp {
                //     out.push((idx, *warp));
                //     // out.push_back((idx, Arc::clone(warp)));
                // }
                //
                // log::debug!(
                //     "added greedy warp (last supervised issued idx={}): {:?}",
                //     self.last_supervised_issued_idx,
                //     &greedy_warp.map(|(idx, w)| (idx, w.dynamic_warp_id)) // &greedy_warp.map(|(idx, w)| (idx, w.try_lock().dynamic_warp_id))
                // );

                out.extend(
                    // self.supervised_warps_sorted
                    warps_sorted.drain(..).take(num_warps_to_add),
                    // .filter(|(idx, warp)| {
                    //     if let Some((greedy_idx, greedy_warp)) = greedy_warp {
                    //         // let already_added = Arc::ptr_eq(greedy_warp, warp);
                    //         // assert_eq!(already_added, *idx == greedy_idx);
                    //         // !already_added
                    //         let already_added = *idx == greedy_idx;
                    //         !already_added
                    //     } else {
                    //         true
                    //     }
                    // }),
                );
            }
            Ordering::PRIORITY_FUNC_ONLY => {
                out.extend(
                    // self.supervised_warps_sorted
                    warps_sorted.drain(..).take(num_warps_to_add),
                );
            }
        }

        assert_eq!(
            num_warps_to_add,
            out.len(),
            "either too few supervised warps or greedy warp not in supervised warps"
        );

        // out
        out.into_iter()
    }
}
