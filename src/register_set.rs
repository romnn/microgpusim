use super::instruction::WarpInstruction;

/// Register set that can hold multiple instructions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisterSet {
    pub stage: super::PipelineStage,
    pub regs: Box<[Option<WarpInstruction>]>,
    pub id: usize,
}

/// Find oldest instruction based on unique ID
fn oldest_instruction_reducer<'a>(
    mut oldest: (usize, &'a Option<WarpInstruction>),
    ready: (usize, &'a Option<WarpInstruction>),
) -> (usize, &'a Option<WarpInstruction>) {
    match (&oldest, &ready) {
        ((_, Some(o)), (_, Some(r))) if o.uid < r.uid => {
            // ready is newer, so nothing to do here
        }
        _ => oldest = ready,
    }
    oldest
}

/// Find oldest instruction based on unique ID
fn oldest_instruction_reducer_mut<'a>(
    mut oldest: (usize, &'a mut Option<WarpInstruction>),
    ready: (usize, &'a mut Option<WarpInstruction>),
) -> (usize, &'a mut Option<WarpInstruction>) {
    if let ((_, Some(o)), (_, Some(r))) = (&oldest, &ready) {
        log::trace!(
            "oldest={} uid = {}  <  ready={} uid = {}",
            o,
            o.uid,
            r,
            r.uid
        );
    }
    match (&oldest, &ready) {
        ((_, Some(o)), (_, Some(r))) if o.uid < r.uid => {
            // ready is newer, so nothing to do here
        }
        _ => oldest = ready,
    }
    oldest
}

/// Trait for accessing the register set.
///
/// TODO: split this up
pub trait Access<I> {
    #[must_use]
    fn get(&self, reg_id: usize) -> Option<&Option<I>>;

    #[must_use]
    fn get_mut(&mut self, reg_id: usize) -> Option<&mut Option<I>>;

    #[must_use]
    fn has_free(&self) -> bool;

    #[must_use]
    fn has_free_sub_core(&self, reg_id: usize) -> bool {
        self.get(reg_id).map(Option::as_ref).flatten().is_none()
    }

    #[must_use]
    fn has_ready(&self) -> bool;

    #[must_use]
    fn size(&self) -> usize;

    #[must_use]
    fn is_empty(&self) -> bool {
        self.size() == 0
    }

    #[must_use]
    fn occupied(&self) -> Box<dyn Iterator<Item = (usize, &Option<I>)> + '_>;

    #[must_use]
    fn occupied_mut(&mut self) -> Box<dyn Iterator<Item = (usize, &mut Option<I>)> + '_>;

    #[must_use]
    fn get_ready(&self) -> Option<(usize, &Option<I>)>;

    #[must_use]
    fn get_ready_mut(&mut self) -> Option<(usize, &mut Option<I>)>;

    #[must_use]
    fn get_ready_sub_core_mut(
        &mut self,
        reg_id: usize,
    ) -> Option<(usize, &mut Option<WarpInstruction>)>;

    #[must_use]
    fn free(&self) -> Box<dyn Iterator<Item = &Option<I>> + '_>;

    #[must_use]
    fn free_mut(&mut self) -> Box<dyn Iterator<Item = (usize, &mut Option<I>)> + '_>;

    fn get_free_mut(&mut self) -> Option<(usize, &mut Option<I>)> {
        self.free_mut().next()
    }

    fn get_free_sub_core_mut(&mut self, scheduler_id: usize) -> Option<(usize, &mut Option<I>)>;

    fn scheduler_id(&self, reg_id: usize) -> Option<usize>;

    fn move_in_from(&mut self, src: Option<I>) {
        let (_, free) = self.get_free_mut().unwrap();
        move_warp(src, free);
    }

    fn move_out_to(&mut self, dest: &mut Option<I>) {
        let (_, ready) = self.get_ready_mut().unwrap();
        move_warp(ready.take(), dest);
    }
}

impl Access<WarpInstruction> for RegisterSet {
    fn get(&self, reg_id: usize) -> Option<&Option<WarpInstruction>> {
        self.regs.get(reg_id)
    }

    fn get_mut(&mut self, reg_id: usize) -> Option<&mut Option<WarpInstruction>> {
        self.regs.get_mut(reg_id)
    }

    fn has_free(&self) -> bool {
        self.regs.iter().any(Option::is_none)
    }

    fn has_ready(&self) -> bool {
        self.regs.iter().any(Option::is_some)
    }

    fn size(&self) -> usize {
        self.regs.len()
    }

    fn occupied(&self) -> Box<dyn Iterator<Item = (usize, &Option<WarpInstruction>)> + '_> {
        Box::new(self.regs.iter().enumerate().filter(|(_, r)| r.is_some()))
    }

    fn occupied_mut(
        &mut self,
    ) -> Box<dyn Iterator<Item = (usize, &mut Option<WarpInstruction>)> + '_> {
        Box::new(self.iter_occupied_mut())
    }

    fn get_free_sub_core_mut(
        &mut self,
        scheduler_id: usize,
    ) -> Option<(usize, &mut Option<WarpInstruction>)> {
        if self.regs[scheduler_id].is_none() {
            log::trace!("found free register at index {}", &scheduler_id);
            Some((scheduler_id, &mut self.regs[scheduler_id]))
        } else {
            None
        }
    }

    fn free(&self) -> Box<dyn Iterator<Item = &Option<WarpInstruction>> + '_> {
        Box::new(self.regs.iter().filter(|r| r.is_none()))
    }

    fn free_mut(&mut self) -> Box<dyn Iterator<Item = (usize, &mut Option<WarpInstruction>)> + '_> {
        Box::new(
            self.regs
                .iter_mut()
                .enumerate()
                .filter(|(_i, r)| r.is_none()),
        )
    }

    fn get_ready(&self) -> Option<(usize, &Option<WarpInstruction>)> {
        self.occupied().reduce(oldest_instruction_reducer)
    }

    fn get_ready_mut(&mut self) -> Option<(usize, &mut Option<WarpInstruction>)> {
        self.iter_occupied_mut()
            .reduce(oldest_instruction_reducer_mut)
    }

    fn get_ready_sub_core_mut(
        &mut self,
        reg_id: usize,
    ) -> Option<(usize, &mut Option<WarpInstruction>)> {
        if self.regs[reg_id].is_some() {
            Some((reg_id, &mut self.regs[reg_id]))
        } else {
            None
        }
    }

    fn scheduler_id(&self, reg_id: usize) -> Option<usize> {
        match self.regs.get(reg_id).and_then(Option::as_ref) {
            Some(r) => r.scheduler_id,
            None => None,
        }
    }
}

impl RegisterSet {
    #[must_use]
    pub fn new(stage: super::PipelineStage, size: usize, id: usize) -> Self {
        let regs = (0..size).map(|_| None).collect();
        Self { stage, regs, id }
    }

    pub fn iter_occupied_mut(
        &mut self,
    ) -> impl Iterator<Item = (usize, &mut Option<WarpInstruction>)> + '_ {
        self.regs
            .iter_mut()
            .enumerate()
            .filter(|(_, r)| r.is_some())
    }
}

impl std::fmt::Display for RegisterSet {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let instructions = self
            .regs
            .iter()
            .map(|inst| inst.as_ref().map(std::string::ToString::to_string));
        f.debug_list().entries(instructions).finish()
    }
}

pub fn move_warp<T>(from: Option<T>, to: &mut Option<T>) {
    *to = from;
}
