use super::instruction::WarpInstruction;

/// Register that can hold multiple instructions.
#[derive(Debug)]
pub struct RegisterSet {
    name: String,
    regs: Vec<Option<WarpInstruction>>,
}

impl RegisterSet {
    pub fn new(size: usize, name: String) -> Self {
        let regs = (0..size).map(|_| None).collect();
        Self { regs, name }
    }

    pub fn has_free(&self) -> bool {
        self.regs.iter().any(Option::is_none)
        // self.regs.iter().any(|r| match r {
        //     Some(r) => r.empty(),
        //     None => true,
        // })
    }

    pub fn has_free_sub_core(&self, sub_core_model: bool, reg_id: usize) -> bool {
        // in subcore model, each sched has a one specific
        // reg to use (based on sched id)
        if !sub_core_model {
            return self.has_free();
        }

        debug_assert!(reg_id < self.regs.len());
        // self.regs[reg_id].is_none()
        let Some(reg) = self.regs.get(reg_id) else {
            return false;
        };

        reg.as_ref()
            // .and_then(Option::as_ref)
            // .flatten()
            .is_none()
        // .map(|r| r.empty())
        // .unwrap_or(false)
    }

    pub fn has_ready(&self) -> bool {
        self.regs.iter().any(Option::is_some)
        // self.regs.iter().any(|r| match r {
        //     Some(r) => !r.empty(),
        //     None => false,
        // })
    }

    // pub fn has_ready_sub_core(&self, sub_core_model: bool, reg_id: usize) -> bool {
    pub fn has_ready_sub_core(&mut self, reg_id: usize) -> bool {
        // if !sub_core_model {
        //     return self.has_ready();
        // }

        debug_assert!(reg_id < self.regs.len());
        match self.get_ready_sub_core(reg_id) {
            Some(ready) => !ready.empty(),
            None => true,
        }
    }

    pub fn ready_reg_id(&self) -> Option<usize> {
        // for sub core model we need to figure which reg_id has
        // the ready warp this function should only be called
        // if has_ready() was true
        debug_assert!(self.has_ready());
        let mut non_empty = self
            .regs
            .iter()
            .map(Option::as_ref)
            .filter_map(|r| r)
            .filter(|r| !r.empty());

        let mut ready: Option<&WarpInstruction> = None;
        let mut reg_id = None;
        for (i, reg) in non_empty.enumerate() {
            match ready {
                Some(ready) if ready.warp_id < reg.warp_id => {
                    // ready is oldest
                }
                _ => {
                    ready.insert(reg);
                    reg_id = Some(i);
                }
            }
        }
        reg_id
    }

    pub fn schd_id(&self, reg_id: usize) -> Option<usize> {
        match self.regs.get(reg_id).map(Option::as_ref).flatten() {
            Some(r) => {
                debug_assert!(!r.empty());
                Some(r.scheduler_id())
            }
            None => None,
        }
    }

    pub fn get_ready_sub_core(&mut self, reg_id: usize) -> Option<&mut WarpInstruction> {
        debug_assert!(reg_id < self.regs.len());
        self.regs // [reg_id]
            .get_mut(reg_id)
            .map(Option::as_mut)
            .flatten()
        // .filter(Option::is_some)
        // .filter(|r| r.empty())
    }

    pub fn get_free(&mut self) -> Option<&mut Option<WarpInstruction>> {
        let mut free = self
            .regs
            .iter_mut()
            // .map(Option::as_mut)
            .filter(|r| r.is_none());
        // .filter(Option::is_none);
        // .filter_map(|r| r.as_ref())
        // .filter_map(|r| r.as_ref())
        // .filter(|r| r.empty());
        free.next()
    }

    pub fn get_free_sub_core(&mut self, reg_id: usize) -> Option<&mut Option<WarpInstruction>> {
        // in subcore model, each sched has a one specific reg
        // to use (based on sched id)
        debug_assert!(reg_id < self.regs.len());
        self.regs.get_mut(reg_id) // .and_then(Option::as_ref)
                                  // .filter(|r| r.empty())
    }

    pub fn size(&self) -> usize {
        self.regs.len()
    }

    pub fn empty(&self) -> bool {
        todo!("RegisterSet::empty")
    }

    pub fn move_in(&mut self, src: WarpInstruction) {
        if let Some(free) = self.get_free() {
            free.insert(src);
        }
        // move_warp(src, free);
    }

    // pub fn move_in(&bool sub_core_model, unsigned reg_id, warp_inst_t *&src) {
    //   warp_inst_t **free;
    //   if (!sub_core_model) {
    //     free = get_free();
    //   } else {
    //     assert(reg_id < regs.size());
    //     free = get_free(sub_core_model, reg_id);
    //   }
    //   move_warp(*free, src);
    // }

    pub fn move_out_to(&mut self, dest: WarpInstruction) {
        // warp_inst_t * *ready = get_ready();
        // move_warp(dest, *ready);
    }

    // void move_out_to(bool sub_core_model, unsigned reg_id, warp_inst_t *&dest) {
    //   if (!sub_core_model) {
    //     return move_out_to(dest);
    //   }
    //   warp_inst_t **ready = get_ready(sub_core_model, reg_id);
    //   assert(ready != NULL);
    //   move_warp(dest, *ready);
    // }
}

impl std::fmt::Display for RegisterSet {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_list().entries(self.regs.iter()).finish()
    }
}

fn swap<T>(x: &mut [T], i: usize, j: usize) {
    let (lo, hi) = match i.cmp(&j) {
        // no swapping necessary
        std::cmp::Ordering::Equal => return,

        // get the smallest and largest of the two indices
        std::cmp::Ordering::Less => (i, j),
        std::cmp::Ordering::Greater => (j, i),
    };

    let (init, tail) = x.split_at_mut(hi);
    std::mem::swap(&mut init[lo], &mut tail[0]);
}

// fn move_warp<T>(from: &mut Option<T>, to: &mut Option<T>) {
fn move_warp<T>(from: &mut T, to: &mut T) {
    // fn move_warp<T>(x: &mut [T], from: usize, to: usize) {
    // debug_assert!(
    std::mem::swap(from, to);
}

// void move_warp(warp_inst_t *&dst, warp_inst_t *&src) {
//   assert(dst->empty());
//   warp_inst_t *temp = dst;
//   dst = src;
//   src = temp;
//   src->clear();
// }

//   void move_in(warp_inst_t *&src) {
//     warp_inst_t **free = get_free();
//     move_warp(*free, src);
//   }
//
//   void move_in(bool sub_core_model, unsigned reg_id, warp_inst_t *&src) {
//     warp_inst_t **free;
//     if (!sub_core_model) {
//       free = get_free();
//     } else {
//       assert(reg_id < regs.size());
//       free = get_free(sub_core_model, reg_id);
//     }
//     move_warp(*free, src);
//   }
//
//   void move_out_to(warp_inst_t *&dest) {
//     warp_inst_t **ready = get_ready();
//     move_warp(dest, *ready);
//   }
//
//   void move_out_to(bool sub_core_model, unsigned reg_id, warp_inst_t *&dest) {
//     if (!sub_core_model) {
//       return move_out_to(dest);
//     }
//     warp_inst_t **ready = get_ready(sub_core_model, reg_id);
//     assert(ready != NULL);
//     move_warp(dest, *ready);
//   }
//
//   warp_inst_t **get_ready() {
//     warp_inst_t **ready;
//     ready = NULL;
//     for (unsigned i = 0; i < regs.size(); i++) {
//       if (not regs[i]->empty()) {
//         if (ready and (*ready)->get_uid() < regs[i]->get_uid()) {
//           // ready is oldest
//         } else {
//           ready = &regs[i];
//         }
//       }
//     }
//     return ready;
//   }
