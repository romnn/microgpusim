use crate::warp;

use crate::sync::{Arc, Mutex};

#[must_use]
pub fn all_different<T>(values: &[Arc<Mutex<T>>]) -> bool {
    for (vi, v) in values.iter().enumerate() {
        for (vii, vv) in values.iter().enumerate() {
            let should_be_equal = vi == vii;
            let are_equal = Arc::ptr_eq(v, vv);
            if should_be_equal && !are_equal {
                return false;
            }
            if !should_be_equal && are_equal {
                return false;
            }
        }
    }
    true
}

pub fn sort_warps_by_oldest_dynamic_id(
    lhs: &(usize, warp::Ref),
    rhs: &(usize, warp::Ref),
    issuer: &dyn crate::core::WarpIssuer,
) -> std::cmp::Ordering {
    let mut lhs_warp = lhs.1.try_lock();
    let mut rhs_warp = rhs.1.try_lock();
    let lhs_blocked = lhs_warp.done_exit()
        || lhs_warp.waiting()
        || issuer.warp_waiting_at_barrier(lhs_warp.warp_id)
        || issuer.warp_waiting_at_mem_barrier(&mut lhs_warp);
    let rhs_blocked = rhs_warp.done_exit()
        || rhs_warp.waiting()
        || issuer.warp_waiting_at_barrier(rhs_warp.warp_id)
        || issuer.warp_waiting_at_mem_barrier(&mut rhs_warp);

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
    // NUM_ORDERING,
}

impl super::Base {
    pub fn order_by_priority<F>(&mut self, ordering: Ordering, priority_func: F)
    where
        F: FnMut(&(usize, warp::Ref), &(usize, warp::Ref)) -> std::cmp::Ordering,
    {
        let num_warps_to_add = self.supervised_warps.len();
        let out = &mut self.next_cycle_prioritized_warps;

        debug_assert!(num_warps_to_add <= self.warps.len());
        out.clear();

        debug_assert!(all_different(self.supervised_warps.make_contiguous()));

        let mut last_issued_iter = self
            .supervised_warps
            .iter()
            .enumerate()
            .skip(self.last_supervised_issued_idx);
        debug_assert!(all_different(&self.warps));

        // sort a copy of the supervised warps for stability
        self.supervised_warps_sorted.clear();
        self.supervised_warps_sorted
            .extend(self.supervised_warps.iter().cloned().enumerate());

        self.supervised_warps_sorted.sort_unstable_by(priority_func);

        debug_assert!(all_different(
            &self
                .supervised_warps_sorted
                .clone()
                .into_iter()
                .map(|(_, w)| w)
                .collect::<Vec<_>>()
        ));

        match ordering {
            Ordering::GREEDY_THEN_PRIORITY_FUNC => {
                let greedy_warp = last_issued_iter.next();
                if let Some((idx, warp)) = greedy_warp {
                    out.push_back((idx, Arc::clone(warp)));
                }

                log::debug!(
                    "added greedy warp (last supervised issued idx={}): {:?}",
                    self.last_supervised_issued_idx,
                    &greedy_warp.map(|(idx, w)| (idx, w.try_lock().dynamic_warp_id))
                );

                out.extend(
                    self.supervised_warps_sorted
                        .drain(..)
                        .take(num_warps_to_add)
                        .filter(|(idx, warp)| {
                            if let Some((greedy_idx, greedy_warp)) = greedy_warp {
                                let already_added = Arc::ptr_eq(greedy_warp, warp);
                                assert_eq!(already_added, *idx == greedy_idx);
                                !already_added
                            } else {
                                true
                            }
                        }),
                );
            }
            Ordering::PRIORITY_FUNC_ONLY => {
                out.extend(
                    self.supervised_warps_sorted
                        .drain(..)
                        .take(num_warps_to_add),
                );
            }
        }
        assert_eq!(
            num_warps_to_add,
            out.len(),
            "either too few supervised warps or greedy warp not in supervised warps"
        );
    }
}
