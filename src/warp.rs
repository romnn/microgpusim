use crate::{instruction::WarpInstruction, kernel::Kernel};
use bitvec::{array::BitArray, BitArr};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Warp size.
///
/// Number of threads per warp.
pub const WARP_SIZE: usize = 32;

/// Thread active mask.
///
/// Bitmask where a 1 at position i means that thread i is active for the current instruction.
pub type ActiveMask = BitArr!(for WARP_SIZE, in u32);

pub type Ref = Arc<Mutex<Warp>>;

#[derive(Debug)]
pub struct Warp {
    pub block_id: u64,
    pub dynamic_warp_id: usize,
    pub warp_id: usize,
    pub kernel: Option<Arc<Kernel>>,

    pub trace_pc: usize,
    pub active_mask: ActiveMask,
    pub trace_instructions: VecDeque<WarpInstruction>,

    // state
    pub done_exit: bool,
    pub num_instr_in_pipeline: usize,
    pub num_outstanding_stores: usize,
    pub num_outstanding_atomics: usize,
    pub has_imiss_pending: bool,
    pub instr_buffer: Vec<Option<WarpInstruction>>,
    pub next: usize,
}

impl PartialEq for Warp {
    fn eq(&self, other: &Self) -> bool {
        self.kernel == other.kernel
            && self.block_id == other.block_id
            && self.warp_id == other.warp_id
            && self.dynamic_warp_id == other.dynamic_warp_id
    }
}

const IBUFFER_SIZE: usize = 2;

impl Default for Warp {
    fn default() -> Self {
        let instr_buffer = vec![None; IBUFFER_SIZE];
        Self {
            block_id: 0,
            dynamic_warp_id: u32::MAX as usize,
            warp_id: u32::MAX as usize,
            kernel: None,
            trace_pc: 0,
            trace_instructions: VecDeque::new(),
            active_mask: BitArray::ZERO,
            done_exit: false,
            num_instr_in_pipeline: 0,
            num_outstanding_stores: 0,
            num_outstanding_atomics: 0,
            has_imiss_pending: false,
            instr_buffer,
            next: 0,
        }
    }
}

impl Warp {
    pub fn init(
        &mut self,
        // _start_pc: Option<usize>,
        block_id: u64,
        warp_id: usize,
        dynamic_warp_id: usize,
        active_mask: ActiveMask,
        kernel: Arc<Kernel>,
    ) {
        self.block_id = block_id;
        self.warp_id = warp_id;
        self.dynamic_warp_id = dynamic_warp_id;
        self.done_exit = false;
        self.kernel = Some(kernel);
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
        self.next = 0;
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

    pub fn ibuffer_fill(&mut self, slot: usize, instr: WarpInstruction) {
        debug_assert!(slot < self.instr_buffer.len());
        self.instr_buffer[slot] = Some(instr);
        self.next = 0;
    }

    #[must_use]
    pub fn ibuffer_size(&self) -> usize {
        self.instr_buffer.iter().filter(|x| x.is_some()).count()
    }

    pub fn ibuffer_empty(&self) -> bool {
        self.instr_buffer.iter().all(Option::is_none)
    }

    pub fn ibuffer_flush(&mut self) {
        for i in &mut self.instr_buffer {
            if i.is_some() {
                self.num_instr_in_pipeline -= 1;
            }
            *i = None;
        }
    }

    #[must_use]
    pub fn ibuffer_peek(&self) -> Option<&WarpInstruction> {
        self.instr_buffer[self.next].as_ref()
    }

    pub fn ibuffer_take(&mut self) -> Option<WarpInstruction> {
        self.instr_buffer[self.next].take()
    }

    pub fn ibuffer_step(&mut self) {
        self.next = (self.next + 1) % IBUFFER_SIZE;
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
    pub fn has_instr_in_pipeline(&self) -> bool {
        self.num_instr_in_pipeline > 0
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
    pub fn functional_done(&self) -> bool {
        self.active_mask.not_any()
    }

    #[must_use]
    pub fn waiting(&self) -> bool {
        if self.functional_done() {
            // waiting to be initialized with a kernel
            true
        // } else if core.warp_waiting_at_barrier(self.warp_id) {
        //     // waiting for other warps in block to reach barrier
        //     true
        // } else if core.warp_waiting_at_mem_barrier(self.warp_id) {
        //     // waiting for memory barrier
        //     true
        } else {
            self.num_outstanding_atomics > 0
        }
    }

    #[must_use]
    pub fn dynamic_warp_id(&self) -> usize {
        self.dynamic_warp_id
    }
}
