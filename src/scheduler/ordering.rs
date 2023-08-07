use super::{BaseSchedulerUnit, WarpRef};

use std::cell::RefCell;
use std::rc::Rc;

pub fn all_different<T>(values: &[Rc<RefCell<T>>]) -> bool {
    for (vi, v) in values.iter().enumerate() {
        for (vii, vv) in values.iter().enumerate() {
            let should_be_equal = vi == vii;
            let are_equal = Rc::ptr_eq(v, vv);
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

pub fn sort_warps_by_oldest_dynamic_id(lhs: &WarpRef, rhs: &WarpRef) -> std::cmp::Ordering {
    let lhs = lhs.try_borrow().unwrap();
    let rhs = rhs.try_borrow().unwrap();
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

impl BaseSchedulerUnit {
    pub fn order_by_priority<F>(&mut self, ordering: Ordering, priority_func: F)
    where
        F: FnMut(&WarpRef, &WarpRef) -> std::cmp::Ordering,
    {
        let num_warps_to_add = self.supervised_warps.len();
        let out = &mut self.next_cycle_prioritized_warps;

        debug_assert!(num_warps_to_add <= self.warps.len());
        out.clear();

        debug_assert!(all_different(self.supervised_warps.make_contiguous()));

        let mut last_issued_iter = self
            .supervised_warps
            .iter()
            .skip(self.last_supervised_issued_idx);
        debug_assert!(all_different(&self.warps));

        // sort a copy of the supervised warps reorder those for stability
        let mut supervised_warps_sorted: Vec<_> =
            self.supervised_warps.clone().into_iter().collect();
        supervised_warps_sorted.sort_by(priority_func);

        debug_assert!(all_different(&supervised_warps_sorted));

        match ordering {
            Ordering::GREEDY_THEN_PRIORITY_FUNC => {
                let greedy_value = last_issued_iter.next();
                if let Some(greedy) = greedy_value {
                    out.push_back(Rc::clone(greedy));
                }

                log::debug!(
                    "added greedy warp (last supervised issued idx={}): {:?}",
                    self.last_supervised_issued_idx,
                    &greedy_value.map(|w| w.borrow().dynamic_warp_id)
                );

                out.extend(
                    supervised_warps_sorted
                        .into_iter()
                        .take(num_warps_to_add)
                        .filter(|warp| {
                            if let Some(greedy) = greedy_value {
                                let already_added = Rc::ptr_eq(greedy, warp);
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
