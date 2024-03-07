use crate::instruction::WarpInstruction;
use std::collections::VecDeque;
pub use trace_model::{active_mask::Inner as ActiveMaskInner, ActiveMask, WARP_SIZE};

#[derive(Debug)]
pub struct Warp {
    pub block_id: u64,
    pub dynamic_warp_id: usize,
    pub warp_id: usize,

    pub kernel_id: Option<u64>,
    pub trace_pc: usize,
    pub active_mask: ActiveMask,
    pub trace_instructions: VecDeque<WarpInstruction>,

    pub done_exit: bool,
    pub num_instr_in_pipeline: usize,
    pub num_outstanding_stores: usize,
    pub num_outstanding_atomics: usize,
    pub waiting_for_memory_barrier: bool,
    pub has_imiss_pending: bool,

    pub instr_buffer: InstructionBuffer,
}

impl std::fmt::Display for Warp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Warp[warp_id={}]", self.warp_id)
    }
}

const IBUFFER_SIZE: usize = 2;

#[derive(Debug)]
pub struct InstructionBuffer {
    inner: Box<[Option<WarpInstruction>]>,
    next: usize,
}

impl InstructionBuffer {
    pub fn new(size: usize) -> Self {
        Self {
            inner: utils::box_slice![None; size],
            next: 0,
        }
    }
}

impl InstructionBuffer {
    pub fn fill(&mut self, slot: usize, instr: WarpInstruction) {
        debug_assert!(slot < self.inner.len());
        self.inner[slot] = Some(instr);
        self.next = 0;
    }

    #[must_use]
    pub fn iter_filled(&self) -> impl Iterator<Item = &WarpInstruction> + '_ {
        self.inner.iter().filter_map(Option::as_ref)
    }

    #[must_use]
    pub fn size(&self) -> usize {
        self.inner.iter().filter(|x| x.is_some()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.iter().all(Option::is_none)
    }

    pub fn flush(&mut self) -> usize {
        let mut num_flushed = 0;
        for i in self.inner.iter_mut() {
            if i.is_some() {
                num_flushed += 1;
            }
            *i = None;
        }
        num_flushed
    }

    pub fn pos(&self) -> usize {
        self.next
    }

    pub fn peek(&self) -> Option<&WarpInstruction> {
        self.inner[self.next].as_ref()
    }

    pub fn take(&mut self) -> Option<WarpInstruction> {
        self.inner[self.next].take()
    }

    pub fn step(&mut self) {
        self.next = (self.next + 1) % self.inner.len();
    }

    pub fn reset(&mut self) {
        self.inner.fill(None);
        self.next = 0;
    }
}

impl Default for Warp {
    fn default() -> Self {
        let instr_buffer = InstructionBuffer::new(IBUFFER_SIZE);
        Self {
            block_id: 0,
            dynamic_warp_id: u32::MAX as usize,
            warp_id: u32::MAX as usize,
            kernel_id: None,
            trace_pc: 0,
            trace_instructions: VecDeque::new(),
            active_mask: ActiveMask::ZERO,
            done_exit: false,
            num_instr_in_pipeline: 0,
            num_outstanding_stores: 0,
            num_outstanding_atomics: 0,
            has_imiss_pending: false,
            waiting_for_memory_barrier: false,
            instr_buffer,
        }
    }
}

impl Warp {
    pub fn init(
        &mut self,
        block_id: u64,
        warp_id: usize,
        dynamic_warp_id: usize,
        active_mask: ActiveMask,
        kernel_id: u64,
    ) {
        self.block_id = block_id;
        self.warp_id = warp_id;
        self.dynamic_warp_id = dynamic_warp_id;
        self.done_exit = false;
        self.kernel_id = Some(kernel_id);
        self.active_mask = active_mask;
    }

    pub fn reset(&mut self) {
        debug_assert_eq!(self.num_outstanding_stores, 0);
        debug_assert_eq!(self.num_instr_in_pipeline, 0);
        self.has_imiss_pending = false;
        self.warp_id = u32::MAX as usize;
        self.dynamic_warp_id = u32::MAX as usize;

        self.active_mask.fill(false);
        self.done_exit = true;
        self.instr_buffer.reset();
    }

    #[must_use]
    pub fn current_instr(&self) -> Option<&WarpInstruction> {
        self.trace_instructions.get(self.trace_pc)
    }

    pub fn push_trace_instruction(&mut self, instr: WarpInstruction) {
        self.trace_instructions.push_back(instr);
    }

    pub fn next_trace_inst(&mut self) -> Option<&WarpInstruction> {
        let trace_instr = self.trace_instructions.get(self.trace_pc)?;
        self.trace_pc += 1;
        Some(trace_instr)
    }

    #[must_use]
    pub fn instruction_count(&self) -> usize {
        self.trace_instructions.len()
    }

    #[must_use]
    pub fn pc(&self) -> Option<usize> {
        debug_assert!(self.trace_pc <= self.instruction_count());
        self.trace_instructions
            .get(self.trace_pc)
            .map(|instr| instr.pc)
    }

    #[must_use]
    pub fn done(&self) -> bool {
        self.trace_pc == self.instruction_count()
    }

    pub fn clear(&mut self) {
        self.trace_pc = 0;
        self.trace_instructions.clear();
    }

    #[must_use]
    pub fn done_exit(&self) -> bool {
        self.done_exit
    }

    #[must_use]
    pub fn hardware_done(&self) -> bool {
        self.functional_done() && self.stores_done() && self.num_instr_in_pipeline == 0
    }

    #[must_use]
    pub fn stores_done(&self) -> bool {
        self.num_outstanding_stores == 0
    }

    #[must_use]
    pub fn num_completed(&self) -> usize {
        self.active_mask.count_zeros()
    }

    pub fn set_thread_completed(&mut self, thread_id: usize) {
        self.active_mask.set(thread_id, false);
    }

    #[must_use]
    #[inline]
    pub fn functional_done(&self) -> bool {
        self.active_mask.not_any()
    }

    #[must_use]
    pub fn waiting(&self) -> bool {
        if self.functional_done() {
            // waiting to be initialized with a kernel
            true
        } else {
            self.num_outstanding_atomics > 0
        }
    }

    #[must_use]
    pub fn dynamic_warp_id(&self) -> usize {
        self.dynamic_warp_id
    }
}
