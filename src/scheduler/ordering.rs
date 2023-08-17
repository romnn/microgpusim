use crate::warp;

use std::sync::{Arc, Mutex};

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

pub fn sort_warps_by_oldest_dynamic_id(lhs: &warp::Ref, rhs: &warp::Ref) -> std::cmp::Ordering {
    let lhs = lhs.try_lock().unwrap();
    let rhs = rhs.try_lock().unwrap();
    if lhs.done_exit() || lhs.waiting() {
        std::cmp::Ordering::Greater
    } else if rhs.done_exit() || rhs.waiting() {
        std::cmp::Ordering::Less
    } else {
        lhs.dynamic_warp_id().cmp(&rhs.dynamic_warp_id())
    }
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
    pub fn order_by_priority<F>(&mut self, ordering: Ordering, mut priority_func: F)
    where
        F: FnMut(&warp::Ref, &warp::Ref) -> std::cmp::Ordering,
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

        // sort a copy of the supervised warps reorder those for stability
        let mut supervised_warps_sorted: Vec<_> = self
            .supervised_warps
            .clone()
            .into_iter()
            .enumerate()
            .collect();
        supervised_warps_sorted.sort_by(|(_, a), (_, b)| priority_func(a, b));

        debug_assert!(all_different(
            &supervised_warps_sorted
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
                    &greedy_warp.map(|(idx, w)| (idx, w.try_lock().unwrap().dynamic_warp_id))
                );

                out.extend(
                    supervised_warps_sorted
                        .into_iter()
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
                out.extend(supervised_warps_sorted.into_iter().take(num_warps_to_add));
            }
        }
        assert_eq!(
            num_warps_to_add,
            out.len(),
            "either too few supervised warps or greedy warp not in supervised warps"
        );
    }
}
