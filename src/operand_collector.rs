use super::{config, core::PipelineStage, instruction::WarpInstruction, register_set};
use crate::sync::Arc;
use bitvec::{array::BitArray, BitArr};
use console::style;
use itertools::Itertools;
use register_set::Access;
use smallvec::SmallVec;
use std::collections::{HashMap, VecDeque};
use trace_model::ToBitString;
use utils::box_slice;

pub const MAX_REG_OPERANDS: usize = 32;

#[inline]
fn compute_register_bank(
    reg_num: u32,
    warp_id: usize,
    num_banks: usize,
    warp_bank_shift: usize,
    sub_core_model: bool,
    banks_per_scheduler: usize,
    schededuler_id: usize,
) -> usize {
    let mut bank = reg_num as usize;
    if warp_bank_shift > 0 {
        bank += warp_id;
    }
    if sub_core_model {
        let bank_num = (schededuler_id * banks_per_scheduler) + (bank % banks_per_scheduler);
        debug_assert!(bank_num < num_banks);
        bank_num
    } else {
        bank % num_banks
    }
}

/// An instruction source operand.
///
/// For example, the instruction "add r0 r1 r2" has
/// 2 source operands: r1, r2
///
/// Source operands are read operands.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SourceOperand {
    /// The warp that this operand belongs to
    pub warp_id: usize,
    /// The logical operand
    pub operand: usize,
    /// The physical register?
    pub register: u32,
    /// The register bank this operand is assigned to
    pub bank: usize,
    /// The scheduler that issued this instruction
    pub scheduler_id: usize,
    /// The operand collector unit that tracks this operand
    pub collector_unit_id: usize,
}

/// An instruction destination operand.
///
/// For example, the instruction "add r0 r1 r2" has
/// 1 destination operand: r0
///
/// Destination operands are write operands.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DestinationOperand {
    /// The warp that this operand belongs to
    pub warp_id: usize,
    /// The physical register?
    pub register: u32,
    /// The register bank this operand is assigned to
    pub bank: usize,
    /// The scheduler that issued this instruction
    pub scheduler_id: usize,
}

#[derive(Debug, Clone)]
pub struct PendingCollectorUnitInstruction {
    // Warp id assigned to this collector unit.
    // warp_id: usize,
    /// Instruction assigned to this collector unit.
    warp_instr: WarpInstruction,

    /// The output pipeline register.
    ///
    /// This register will be issued to once all
    /// operands are ready
    output_register: PipelineStage,

    /// Source operands to collect.
    src_operands: [Option<SourceOperand>; MAX_REG_OPERANDS * 2],

    /// Bitmask of operands that are not ready yet.
    not_ready: BitArr!(for MAX_REG_OPERANDS * 2),
}

// impl PendingCollectorUnitInstruction {
//     pub fn new(warp_instr: WarpInstruction, output_register: PipelineStage) -> Self {
//         let src_operands = [(); MAX_REG_OPERANDS * 2].map(|_| None);
//         Self {
//             warp_instr,
//             src_operands,
//             output_register,
//             not_ready: BitArray::ZERO,
//         }
//     }
// }

/// A collector unit.
///
/// A collector unit is conceptionally similar to a reservation
/// station in tomasulos algorithm.
///
/// Once assigned to a warp instruction, it buffers all source operands
/// until all source operands are ready and the output register can be
/// issued to.
#[derive(Debug, Clone)]
pub struct CollectorUnit {
    id: usize,
    // is_free: bool,
    kind: Kind,

    pending: Option<PendingCollectorUnitInstruction>,
    // /// Warp id assigned to this collector unit.
    // warp_id: Option<usize>,
    // /// Instruction assigned to this collector unit.
    // warp_instr: Option<WarpInstruction>,
    //
    // /// The output pipeline register.
    // ///
    // /// This register will be issued to once all
    // /// operands are ready
    // output_register: Option<PipelineStage>,
    //
    // /// Source operands to collect.
    // src_operands: [Option<SourceOperand>; MAX_REG_OPERANDS * 2],
    //
    // /// Bitmask of operands that are not ready yet.
    // not_ready: BitArr!(for MAX_REG_OPERANDS * 2),
    num_banks: usize,
    bank_warp_shift: usize,
    sub_core_model: bool,
    /// Number of banks per scheduler
    num_banks_per_scheduler: usize,

    /// The register ID that this collector unit can read or write.
    ///
    /// If sub_core_model enabled, limit regs this cu can read or write
    reg_id: usize,
}

impl std::fmt::Display for CollectorUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CollectorUnit")
            .field("id", &self.id)
            .field("kind", &self.kind)
            .field("reg_id", &self.reg_id)
            .field("is_free", &self.is_free())
            .field(
                "output_register",
                &self.pending.as_ref().map(|pending| pending.output_register),
            )
            .finish()
    }
}

impl CollectorUnit {
    fn new(
        kind: Kind,
        id: usize,
        // num_banks: usize,
        // bank_warp_shift: usize,
        // sub_core_model: bool,
        // reg_id: usize,
        // num_banks_per_scheduler: usize,
    ) -> Self {
        // let src_operands = [(); MAX_REG_OPERANDS * 2].map(|_| None);
        Self {
            id,
            // is_free: true,
            kind,
            pending: None,
            // warp_instr: None,
            // output_register: None,
            // src_operands,
            // not_ready: BitArray::ZERO,
            // warp_id: None,
            num_banks: 0,
            bank_warp_shift: 0,
            num_banks_per_scheduler: 0,
            reg_id: 0,
            sub_core_model: false,
        }
    }

    // fn new(kind: Kind, id: usize) -> Self {
    //     let src_operands = [(); MAX_REG_OPERANDS * 2].map(|_| None);
    //     Self {
    //         id,
    //         is_free: true,
    //         kind,
    //         warp_instr: None,
    //         output_register: None,
    //         src_operands,
    //         not_ready: BitArray::ZERO,
    //         warp_id: None,
    //         num_banks: 0,
    //         bank_warp_shift: 0,
    //         num_banks_per_scheduler: 0,
    //         reg_id: 0,
    //         sub_core_model: false,
    //     }
    // }
    //
    // pub fn init(
    //     &mut self,
    //     id: usize,
    //     num_banks: usize,
    //     log2_warp_size: usize,
    //     sub_core_model: bool,
    //     reg_id: usize,
    //     banks_per_scheduler: usize,
    // ) {
    //     self.id = id;
    //     self.num_banks = num_banks;
    //     debug_assert!(self.warp_instr.is_none());
    //     self.warp_instr = None;
    //     self.bank_warp_shift = log2_warp_size;
    //     self.sub_core_model = sub_core_model;
    //     self.reg_id = reg_id;
    //     self.num_banks_per_scheduler = banks_per_scheduler;
    // }

    /// Check if the collector unit is ready to be dispatched.
    ///
    /// A collector unit is ready for dispatch once there are no pending operand
    /// reads in the collector unit and the output pipeline register is free.
    pub fn ready(&self, pipeline_stage: &[register_set::RegisterSet]) -> bool {
        let Some(ref pending) = self.pending else {
            return false;
        };
        let output_register = &pipeline_stage[pending.output_register as usize];
        let has_free_register = if self.sub_core_model {
            output_register.has_free_sub_core(self.reg_id)
        } else {
            output_register.has_free()
        };
        log::debug!(
            "is ready?: active = {} (ready={}), has free = {} output register = {:?}",
            pending.not_ready.to_bit_string(),
            pending.not_ready.not_any(),
            has_free_register,
            &output_register
        );

        pending.not_ready.not_any() && has_free_register
    }

    pub fn is_free(&self) -> bool {
        self.pending.is_none()
    }

    pub fn dispatch(&mut self, pipeline_reg: &mut [register_set::RegisterSet]) {
        let Some(pending) = self.pending.take() else {
            return;
        };
        debug_assert!(pending.not_ready.not_any());

        let output_register = pending.output_register;
        let output_register = &mut pipeline_reg[output_register as usize];
        // let warp_instr = pending.warp_instr.take();

        if self.sub_core_model {
            let free_reg = output_register.get_mut(self.reg_id).unwrap();
            assert!(free_reg.is_none());
            log::trace!("found free register at index {}", &self.reg_id);
            register_set::move_warp(Some(pending.warp_instr), free_reg);
        } else {
            let (_, free_reg) = output_register.get_free_mut().unwrap();
            register_set::move_warp(Some(pending.warp_instr), free_reg);
        }

        // self.pending = None;
        // self.is_free = true;
        // self.warp_id = None;
        // self.src_operands.fill(None);
    }

    fn allocate(
        &mut self,
        input_reg_set: &mut register_set::RegisterSet,
        output_reg_id: PipelineStage,
    ) -> bool {
        log::debug!(
            "{}",
            style(format!("operand collector::allocate({:?})", self.kind)).green(),
        );

        debug_assert!(self.is_free());
        // debug_assert!(self.not_ready.not_any());

        // self.is_free = false;
        // self.pending = Some(PendingCollectorUnitInstruction::new());
        // self.output_register = Some(output_reg_id);

        // if let Some((_, Some(ready_reg))) = input_reg_set.get_ready() {
        // if let Some((_, Some(ready_reg))) = input_reg_set
        if let Some(ready_reg) = input_reg_set
            .get_ready_mut()
            .and_then(|(_, ready_reg)| ready_reg.take())
        {
            // todo: do we need warp id??
            // self.warp_id = Some(ready_reg.warp_id);

            log::debug!(
                "operand collector::allocate({:?}) => src arch reg = {:?}",
                self.kind,
                ready_reg
                    .src_arch_reg
                    .iter()
                    .map(|r| r.map(i64::from).unwrap_or(-1))
                    .collect::<Vec<i64>>(),
            );

            // self.src_operands.fill(None);

            let mut src_operands = [(); MAX_REG_OPERANDS * 2].map(|_| None);
            let mut not_ready = BitArray::ZERO;

            for (op, reg_num) in ready_reg
                .src_arch_reg
                .iter()
                .enumerate()
                .filter_map(|(op, reg_num)| reg_num.map(|reg_num| (op, reg_num)))
                .unique_by(|(_, reg_num)| *reg_num)
            {
                let scheduler_id = ready_reg.scheduler_id.unwrap();
                let bank = compute_register_bank(
                    reg_num,
                    ready_reg.warp_id,
                    self.num_banks,
                    self.bank_warp_shift,
                    self.sub_core_model,
                    self.num_banks_per_scheduler,
                    scheduler_id,
                );

                // self.src_operands[op] = Some(SourceOperand {
                src_operands[op] = Some(SourceOperand {
                    warp_id: ready_reg.warp_id,
                    collector_unit_id: self.id,
                    operand: op,
                    register: reg_num,
                    bank,
                    scheduler_id,
                });
                not_ready.set(op, true);
                // self.not_ready.set(op, true);
            }
            log::debug!(
                "operand collector::allocate({:?}) => active: {}",
                self.kind,
                not_ready.to_bit_string(),
                // self.not_ready.to_bit_string(),
            );

            // let warp_id = ready_reg.warp_id;
            // let mut warp_instr = None;
            // input_reg_set.move_out_to(&mut warp_instr);

            // allocate here
            self.pending = Some(PendingCollectorUnitInstruction {
                // warp_id,
                // warp_instr: warp_instr.unwrap(),
                warp_instr: ready_reg,
                not_ready,
                src_operands,
                output_register: output_reg_id,
            });
            true
        } else {
            false
        }
    }

    pub fn collect_operand(&mut self, op: usize) {
        let Some(ref mut pending) = self.pending else {
            return;
        };
        log::debug!(
            "collector unit [{}] {} collecting operand for {}",
            self.id,
            pending.warp_instr,
            op,
        );
        pending.not_ready.set(op, false);
    }
}

// #[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
// pub enum AllocationKind {
//     // NO_ALLOC,
//     READ,
//     WRITE,
// }
//
// #[derive(Debug, Clone, PartialEq, Eq, Hash)]
// pub struct Allocation {
//     kind: AllocationKind,
//     op: Option<Operand>,
// }
//
// impl Default for Allocation {
//     fn default() -> Self {
//         Self {
//             kind: AllocationKind::NO_ALLOC,
//             op: None,
//         }
//     }
// }

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Allocation {
    Read(SourceOperand),
    Write(DestinationOperand),
}

impl Allocation {
    // #[must_use]
    // pub fn new(kind: AllocationKind, op: Option<Operand>) -> Self {
    //     Self { kind, op }
    // }

    #[must_use]
    pub fn is_read(&self) -> bool {
        matches!(self, Allocation::Read(_))
        // self.kind == AllocationKind::READ_ALLOC
    }

    #[must_use]
    pub fn is_write(&self) -> bool {
        matches!(self, Allocation::Write(_))
        // self.kind == AllocationKind::WRITE_ALLOC
    }

    // #[must_use]
    // pub fn is_free(&self) -> bool {
    //     self.kind == AllocationKind::NO_ALLOC
    // }

    // pub fn allocate_for_read(&mut self, op: Option<Operand>) {
    //     debug_assert!(self.is_free());
    //     self.kind = AllocationKind::READ_ALLOC;
    //     self.op = op;
    // }

    // pub fn allocate_for_write(&mut self, op: Option<Operand>) {
    //     debug_assert!(self.is_free());
    //     self.kind = AllocationKind::WRITE_ALLOC;
    //     self.op = op;
    // }

    // pub fn reset(&mut self) {
    //     self.kind = AllocationKind::NO_ALLOC;
    //     self.op = None;
    // }
}

/// Operand collector register file arbiter.
///
/// This is used to arbitrate between register reads and writes to
/// the register file.
/// As the register file is banked and only a single operand can be
/// read or written for a single ported bank, the goal is to
/// allocate to register banks as efficiently as possible.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Arbiter {
    /// Number of register file banks
    num_banks: usize,
    /// Number of collector units
    num_collectors: usize,

    bank_warp_shift: usize,
    sub_core_model: bool,

    /// Number of register file banks assigned to a single warp scheduler
    num_banks_per_scheduler: usize,

    /// bank number -> register that wins
    allocated_banks: Box<[Option<Allocation>]>,
    queue: Box<[VecDeque<SourceOperand>]>,
    // queue: Box<[VecDeque<Operand>]>,
    /// cu # -> next bank to check for request (rr-arb)
    // allocator_round_robin_head: usize,
    /// first cu to check while arb-ing banks (rr)
    last_cu: usize,
    inmatch: ndarray::Array1<Option<usize>>,
    request: ndarray::Array2<Option<usize>>,
}

impl Arbiter {
    pub fn init(
        &mut self,
        num_collectors: usize,
        num_banks: usize,
        bank_warp_shift: usize,
        sub_core_model: bool,
        num_banks_per_scheduler: usize,
    ) {
        debug_assert!(num_collectors > 0);
        debug_assert!(num_banks > 0);
        self.num_collectors = num_collectors;
        self.num_banks = num_banks;

        self.bank_warp_shift = bank_warp_shift;
        self.sub_core_model = sub_core_model;
        self.num_banks_per_scheduler = num_banks_per_scheduler;

        self.inmatch = ndarray::Array1::from_shape_simple_fn(self.num_banks, || None);
        self.request =
            ndarray::Array2::from_shape_simple_fn((self.num_banks, self.num_collectors), || None);

        self.queue = box_slice![VecDeque::new(); self.num_banks];

        self.allocated_banks = box_slice![None; self.num_banks];
        // self.allocated_banks = box_slice![Allocation::default(); self.num_banks];

        self.reset_alloction();
    }

    fn compat(matches: &[Option<usize>]) -> Vec<i64> {
        matches
            .iter()
            .map(|r| {
                r.map(i64::try_from)
                    .transpose()
                    .ok()
                    .flatten()
                    .unwrap_or(-1)
            })
            .collect()
    }

    /// Allocate operand reads.
    ///
    /// The arbiter checks the arbitration queue and allocates operand
    /// read requests to a list of registers such that operands:
    ///  1. are in different register banks,
    ///  2. do not go to the same operand collector
    ///
    pub fn allocate_reads(&mut self) -> impl Iterator<Item = SourceOperand> + Clone {
        log::trace!("allocate reads: queue={:?}", &self.queue);

        let num_inputs = self.num_banks;
        let num_outputs = self.num_collectors;

        let last_cu_before = self.last_cu;
        let mut cu_priority = self.last_cu;
        // log::debug!("last cu: {}", self.last_cu);

        let no_allocation = self.allocated_banks.iter().all(Option::is_none);
        let empty_queue = self.queue.iter().all(VecDeque::is_empty);

        // clear matching
        let mut allocated: SmallVec<[SourceOperand; 64]> = SmallVec::new();

        // fast path
        if no_allocation && empty_queue {
            self.last_cu = (self.last_cu + 1) % num_outputs;
            return allocated.into_iter();
        }

        let inmatch = &mut self.inmatch;
        let request_matrix = &mut self.request;

        inmatch.fill(None);

        // TODO: refactor to use very small u8 repr enums for this
        request_matrix.fill(Some(0));

        for bank in 0..self.num_banks {
            debug_assert!(bank < num_inputs);
            // for collector in 0..self.num_collectors {
            //     debug_assert!(collector < num_outputs);
            //     request_matrix[(bank, collector)] = Some(0);
            // }
            if let Some(op) = self.queue[bank].front() {
                // let collector_id = op.collector_unit_id.unwrap();
                debug_assert!(op.collector_unit_id < num_outputs);
                // this causes change in search
                request_matrix[(bank, op.collector_unit_id)] = Some(1);
            }
            if let Some(Allocation::Write(_)) = self.allocated_banks[bank] {
                // write gets priority
                inmatch[bank] = Some(0);
            }
            // log::trace!("request: {:?}", &Self::compat(&request[bank]));
        }

        // #[cfg(feature = "timings")]
        // {
        //     crate::TIMINGS
        //         .lock()
        //         .entry("allocate_reads_prepare")
        //         .or_default()
        //         .add(start.elapsed());
        // }

        // log::trace!("inmatch: {:?}", &Self::compat(inmatch));

        // wavefront allocator from booksim
        // loop through diagonals of request matrix

        let square = num_inputs.max(num_outputs);
        debug_assert!(square > 0);

        for p in 0..square {
            let mut output = (cu_priority + p) % num_outputs;

            // step through the current diagonal
            for input in 0..num_inputs {
                // banks at the same cycle
                if inmatch[input].is_none() && request_matrix[(input, output)] != Some(0) {
                    // Grant!
                    inmatch[input] = Some(output);

                    log::trace!("operand collector: register file granting bank {} to OC {} [scheduler id={:?}, warp={:?}]", input, output, self.queue[input].front().map(|w| w.scheduler_id), self.queue[input].front().map(|w| w.warp_id));

                    // outmatch[output] = Some(input);
                    // printf("Register File: granting bank %d to OC %d, schedid %d, warpid
                    // %d, Regid %d\n", input, output, (m_queue[input].front()).get_sid(),
                    // (m_queue[input].front()).get_wid(),
                    // (m_queue[input].front()).get_reg());
                }

                output = (output + 1) % num_outputs;
            }
        }

        log::trace!("inmatch: {:?}", &Self::compat(inmatch.as_slice().unwrap()));

        // Round-robin the priority diagonal
        cu_priority = (cu_priority + 1) % num_outputs;
        log::trace!("cu priority: {:?}", cu_priority);

        // <--- end code from booksim

        self.last_cu = cu_priority;
        log::debug!(
            "last cu: {} -> {} ({} outputs)",
            last_cu_before,
            self.last_cu,
            num_outputs
        );

        for bank in 0..self.num_banks {
            if inmatch[bank].is_some() {
                log::trace!(
                    "inmatch[bank={}] is write={}",
                    bank,
                    matches!(self.allocated_banks[bank], Some(Allocation::Write(_))),
                );
                match self.allocated_banks[bank] {
                    Some(Allocation::Write(_)) => {}
                    None | Some(Allocation::Read(_)) => {
                        if let Some(op) = self.queue[bank].pop_front() {
                            allocated.push(op);
                        }
                    }
                }
            }
        }

        // #[cfg(feature = "timings")]
        // {
        //     crate::TIMINGS
        //         .lock()
        //         .entry("allocate_reads_search_diagonal")
        //         .or_default()
        //         .add(start.elapsed());
        // }

        log::debug!(
            "arbiter allocated {} reads ({:?})",
            allocated.len(),
            &allocated
        );
        for read in &allocated {
            let reg = read.register;
            let bank = compute_register_bank(
                reg,
                read.warp_id,
                self.num_banks,
                self.bank_warp_shift,
                self.sub_core_model,
                self.num_banks_per_scheduler,
                read.scheduler_id,
            );
            self.allocate_bank_for_read(bank, read.clone());
        }

        allocated.into_iter()
        // #[cfg(feature = "timings")]
        // {
        //     crate::TIMINGS
        //         .lock()
        //         .entry("allocate_reads_register_banks")
        //         .or_default()
        //         .add(start.elapsed());
        // }
    }

    /// Adds read requests from a collector unit to the arbitration queue.
    pub fn add_read_requests(&mut self, cu: &CollectorUnit) {
        let Some(ref pending) = cu.pending else {
            return
        };
        for op in pending.src_operands.iter().flatten() {
            let bank = op.bank;
            self.queue[bank].push_back(op.clone());
        }
    }

    /// Check if a register file bank is busy.
    ///
    /// A bank is busy if it has already been allocated with an
    /// operand to be retrieved.
    pub fn bank_idle(&self, bank: usize) -> bool {
        self.allocated_banks[bank].is_none()
    }

    /// Allocates a register file bank for writing of an operand.
    pub fn allocate_bank_for_write(&mut self, bank: usize, op: DestinationOperand) {
        debug_assert!(bank < self.num_banks);
        self.allocated_banks[bank] = Some(Allocation::Write(op));
    }

    /// Allocates a register file bank for reading of an operand.
    pub fn allocate_bank_for_read(&mut self, bank: usize, op: SourceOperand) {
        debug_assert!(bank < self.num_banks);
        self.allocated_banks[bank] = Some(Allocation::Read(op));
    }

    /// Resets all allocations of register banks.
    pub fn reset_alloction(&mut self) {
        self.allocated_banks.fill(None);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DispatchUnit {
    last_collector_unit: usize,
    next_collector_unit: usize,
    sub_core_model: bool,
    num_warp_schedulers: usize,
    kind: Kind,
    id: usize,
}

impl DispatchUnit {
    #[must_use]
    pub fn new(kind: Kind, id: usize) -> Self {
        Self {
            kind,
            id,
            last_collector_unit: 0,
            next_collector_unit: 0,
            sub_core_model: false,
            num_warp_schedulers: 0,
        }
    }

    pub fn init(&mut self, sub_core_model: bool, num_warp_schedulers: usize) {
        self.sub_core_model = sub_core_model;
        self.num_warp_schedulers = num_warp_schedulers;
    }

    /// Find a ready collector unit
    pub fn find_ready<'a>(
        &mut self,
        collector_units: &'a [CollectorUnit],
        set_collector_unit_ids: &'a [usize],
        pipeline_reg: &[register_set::RegisterSet],
    ) -> Option<usize> {
        // With sub-core enabled, round robin starts with the next
        // collector unit assigned to a different sub-core than the one
        // that dispatched last
        let num_collector_units = set_collector_unit_ids.len();
        let cus_per_scheduler = num_collector_units / self.num_warp_schedulers;
        let round_robin_increment = if self.sub_core_model {
            cus_per_scheduler - (self.last_collector_unit % cus_per_scheduler)
        } else {
            1
        };

        log::debug!("dispatch unit {:?}[{}]: find ready: rr_inc = {},last cu = {},num collectors = {}, num warp schedulers = {}, cusPerSched = {}", self.kind, self.id, round_robin_increment, self.last_collector_unit, num_collector_units, self.num_warp_schedulers, cus_per_scheduler);

        debug_assert_eq!(num_collector_units, set_collector_unit_ids.len());
        for i in 0..num_collector_units {
            let i = (self.last_collector_unit + i + round_robin_increment) % num_collector_units;
            // log::trace!(
            //     "dispatch unit {:?}: checking collector unit {}",
            //     self.kind,
            //     i,
            // );

            let collector_unit_id = set_collector_unit_ids[i];
            let collector_unit = &collector_units[collector_unit_id];
            assert_eq!(collector_unit_id, collector_unit.id);

            if collector_unit.ready(pipeline_reg) {
                self.last_collector_unit = i;
                log::debug!(
                    "dispatch unit {:?}[{}]: FOUND ready: chose collector unit {} ({:?})",
                    self.kind,
                    self.id,
                    i,
                    collector_units[i].kind
                );

                // can only dispatch one collector unit per cycle?
                return Some(collector_unit_id);
            }
        }
        log::debug!(
            "dispatch unit {:?}[{}]: did NOT find ready",
            self.kind,
            self.id
        );
        None
    }
}

#[derive(Debug, Clone, Default)]
pub struct InputPort {
    in_ports: PortVec,
    out_ports: PortVec,
    collector_unit_ids: Vec<Kind>,
}

impl InputPort {
    #[must_use]
    pub fn new(in_ports: PortVec, out_ports: PortVec, collector_unit_ids: Vec<Kind>) -> Self {
        debug_assert!(in_ports.len() == out_ports.len());
        debug_assert!(!collector_unit_ids.is_empty());
        Self {
            in_ports,
            out_ports,
            collector_unit_ids,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u32)]
pub enum Kind {
    SP_CUS,
    DP_CUS,
    SFU_CUS,
    TENSOR_CORE_CUS,
    INT_CUS,
    MEM_CUS,
    GEN_CUS,
}

pub type CuSets = HashMap<Kind, Vec<usize>>;

/// Register file
#[derive(Debug, Clone)]
pub struct RegisterFileUnit {
    pub config: Arc<config::GPU>,
    pub initialized: bool,
    pub num_banks: usize,
    pub num_collectors: usize,

    pub bank_warp_shift: usize,
    pub sub_core_model: bool,
    pub num_banks_per_scheduler: usize,
    pub num_warp_schedulers: usize,

    pub arbiter: Arbiter,
    pub in_ports: Vec<InputPort>,
    pub collector_units: Vec<CollectorUnit>,
    pub collector_unit_sets: CuSets,
    pub dispatch_units: Vec<DispatchUnit>,
}

pub type PortVec = Vec<PipelineStage>;

pub trait Writeback {
    /// Writes back the result of an instruction.
    ///
    /// to its output writes back the result of an instruction to the
    fn writeback(&mut self, instr: &mut WarpInstruction) -> bool;
}

impl RegisterFileUnit {
    pub fn new(config: Arc<config::GPU>) -> Self {
        let arbiter = Arbiter::default();
        Self {
            initialized: true,
            num_banks: 0,
            config,
            num_collectors: 0,
            bank_warp_shift: 0,
            sub_core_model: false,
            num_banks_per_scheduler: 0,
            num_warp_schedulers: 0,
            arbiter,
            in_ports: Vec::new(),
            collector_units: Vec::new(),
            collector_unit_sets: CuSets::new(),
            dispatch_units: Vec::new(),
        }
    }

    pub fn init(&mut self, num_banks: usize) {
        let num_collector_units = self.collector_units.len();

        self.num_banks = num_banks;
        self.bank_warp_shift = (self.config.warp_size as f32 + 0.5).log2() as usize;
        debug_assert!(self.bank_warp_shift == 5 || self.config.warp_size != 32);

        self.sub_core_model = self.config.sub_core_model;
        self.num_warp_schedulers = self.config.num_schedulers_per_core;
        if self.sub_core_model {
            debug_assert!(self.num_banks % self.config.num_schedulers_per_core == 0);
            debug_assert!(
                self.num_warp_schedulers <= num_collector_units
                    && num_collector_units % self.num_warp_schedulers == 0
            );
        }
        self.num_banks_per_scheduler = self.num_banks / self.config.num_schedulers_per_core;

        self.arbiter.init(
            num_collector_units,
            num_banks,
            self.bank_warp_shift,
            self.sub_core_model,
            self.num_banks_per_scheduler,
        );
        // let mut reg_id = 0;
        for (cu_id, cu) in self.collector_units.iter_mut().enumerate() {
            if self.sub_core_model {
                let coll_units_per_scheduler = num_collector_units / self.num_warp_schedulers;
                let reg_id = cu_id / coll_units_per_scheduler;
                cu.reg_id = reg_id;
            }
            cu.id = cu_id;
            cu.num_banks = self.num_banks;
            cu.bank_warp_shift = self.bank_warp_shift;
            cu.sub_core_model = self.sub_core_model;
            cu.num_banks_per_scheduler = self.num_banks_per_scheduler;
            // cu.reg_id = reg_id;
            // cu.init(
            //     cu_id,
            //     self.num_banks,
            //     self.bank_warp_shift,
            //     self.sub_core_model,
            //     reg_id,
            //     self.num_banks_per_scheduler,
            // );
            // cu = CollectorUnit::new(
            //     cu_id,
            //     cu_id,
            //     self.num_banks,
            //     self.bank_warp_shift,
            //     self.sub_core_model,
            //     reg_id,
            //     self.num_banks_per_scheduler,
            // );

            debug_assert!(cu.id == cu_id);
        }
        for dispatch_unit in &mut self.dispatch_units {
            dispatch_unit.init(self.sub_core_model, self.num_warp_schedulers);
        }
        self.initialized = true;
    }

    pub fn step(&mut self, pipeline_reg: &mut [register_set::RegisterSet]) {
        log::debug!("{}", style("operand collector::step()").green());
        self.dispatch_ready_cu(pipeline_reg);
        self.allocate_reads();

        debug_assert!(!self.in_ports.is_empty());
        for input_port_num in 0..self.in_ports.len() {
            self.allocate_collector_unit(pipeline_reg, input_port_num);
        }
        self.arbiter.reset_alloction();
    }

    /// Process read requests that do not have conflicts
    pub fn allocate_reads(&mut self) {
        // process read requests that do not have conflicts
        let bank_operand_reads_iter = self.arbiter.allocate_reads();

        if log::log_enabled!(log::Level::Debug) {
            let bank_operand_reads: Vec<_> = bank_operand_reads_iter.clone().collect();
            log::debug!(
                "allocating {} reads ({:?})",
                bank_operand_reads.len(),
                bank_operand_reads
            );
        }
        for read in bank_operand_reads_iter {
            let cu_id = read.collector_unit_id;
            assert!(cu_id < self.collector_units.len());
            let cu = &mut self.collector_units[cu_id];
            cu.collect_operand(read.operand);
        }
    }

    pub fn allocate_collector_unit(
        &mut self,
        pipeline_reg: &mut [register_set::RegisterSet],
        input_port_id: usize,
    ) {
        let input_ports = &self.in_ports[input_port_id];
        log::debug!(
            "{}",
            style(format!(
                "operand collector::allocate_cu({}: {:?})",
                input_port_id, input_ports.collector_unit_ids
            ))
            .green(),
        );

        debug_assert_eq!(input_ports.in_ports.len(), input_ports.out_ports.len());

        for (input_port_id, output_port_id) in input_ports
            .in_ports
            .iter()
            .zip(input_ports.out_ports.iter())
        {
            let input_port = &mut pipeline_reg[*input_port_id as usize];

            if input_port.has_ready() {
                // find a free collector unit
                for cu_set_id in &input_ports.collector_unit_ids {
                    let cu_set: &Vec<usize> = &self.collector_unit_sets[cu_set_id];
                    let mut allocated = false;
                    let mut cu_lower_bound = 0;
                    let mut cu_upper_bound = cu_set.len();

                    // temp: so we can use them later
                    let mut scheduler_id = 0;

                    if self.sub_core_model {
                        // sub core model only allocates on the subset
                        // of CUs assigned to the scheduler that issued
                        let (reg_id, _) = input_port.get_ready().unwrap();
                        debug_assert!(
                            cu_set.len() % self.num_warp_schedulers == 0
                                && cu_set.len() >= self.num_warp_schedulers
                        );
                        let cus_per_sched = cu_set.len() / self.num_warp_schedulers;
                        scheduler_id = input_port.scheduler_id(reg_id).unwrap();
                        cu_lower_bound = scheduler_id * cus_per_sched;
                        cu_upper_bound = cu_lower_bound + cus_per_sched;
                        debug_assert!(cu_upper_bound <= cu_set.len());

                        // each scheduler manages two cu's (dual issue)
                        assert_eq!(cu_upper_bound - cu_lower_bound, 2);
                        // let cus = &cu_set[cu_lower_bound..cu_upper_bound];
                        // assert!(!cus.iter().any(|cu| self.collector_units[*cu].is_free()));
                        // println!(
                        //     "RFU for scheduler {} input port {:?} output port {:?} manages {} cus: {:?}",
                        //     scheduler_id,
                        //     input_port_id,
                        //     output_port_id,
                        //     cus.len(),
                        //     cus.iter().map(|cu| &self.collector_units[*cu]).map(ToString::to_string).collect::<Vec<_>>(),
                        // );
                    }

                    let cus = &cu_set[cu_lower_bound..cu_upper_bound];
                    for collector_unit_id in cus {
                        let collector_unit = &mut self.collector_units[*collector_unit_id];
                        if collector_unit.is_free() {
                            log::debug!(
                                "{} cu={:?}",
                                style("operand collector::allocate()".to_string()).green(),
                                collector_unit.kind
                            );

                            allocated = collector_unit.allocate(input_port, *output_port_id);
                            // todo: check here if allocated before adding
                            // read requests...
                            self.arbiter.add_read_requests(&collector_unit);
                            break;
                        }
                    }

                    if allocated {
                        if log::log_enabled!(log::Level::Trace) {
                            let num_free = cus
                                .iter()
                                .filter(|cu| self.collector_units[**cu].is_free())
                                .count();
                            if num_free == 0 {
                                log::trace!(
                                "RFU for scheduler {} input port {:?} output port {:?} allocated a collector unit (manages {} cus: {:?})",
                                scheduler_id,
                                input_port_id,
                                output_port_id,
                                cus.len(),
                                cus.iter().map(|cu| &self.collector_units[*cu]).map(ToString::to_string).collect::<Vec<_>>(),
                            );
                            }
                        }

                        // this is wrong!
                        // they could, but the dispatch units would need more
                        // than one cycle to dispatch to the same pipe reg
                        // for an execution unit.
                        // assert!(
                        //     cus.iter()
                        //         .filter_map(|cu| self.collector_units[*cu].pending.as_ref())
                        //         .map(|pending| pending.output_register)
                        //         .all_unique(),
                        //     "collector units for the same scheduler never output to the same execution unit pipeliene"
                        // );
                        assert_eq!(
                            cus.iter()
                                .map(|cu| &self.collector_units[*cu].reg_id)
                                .dedup()
                                .count(),
                            1,
                            "in sub core model, have the same reg id assigned for this scheduler to both collector units, because execution units (4sp, 4sfu, 4mem) are statically assigned to one scheduler, have 8 opcols for 4 schedulers, 2 opcolds per scheduler and each can either be of kind sp, sfu, or mem but always only dispatch into the same pipeline register in each of the execution units. that there is never an issue they never can dispatch to the same execution unit."
                        );

                        // a collector unit has been allocated,
                        // no need to search more.
                        break;
                    }
                }
            }
        }
    }

    pub fn dispatch_ready_cu(&mut self, pipeline_reg: &mut [register_set::RegisterSet]) {
        // can dispatch up to num_dispatch_units per cycle
        for dispatch_unit in &mut self.dispatch_units {
            let set = &self.collector_unit_sets[&dispatch_unit.kind];
            // log::trace!(
            //     "dispatch unit {} checking {} collector units ({} collector unit sets in total)",
            //     dispatch_unit.id,
            //     set.len(),
            //     self.collector_unit_sets.len()
            // );
            if let Some(collector_unit_id) =
                dispatch_unit.find_ready(&self.collector_units, set, pipeline_reg)
            {
                let collector_unit = &mut self.collector_units[collector_unit_id];
                collector_unit.dispatch(pipeline_reg);
            }
        }
    }

    pub fn add_cu_set(
        &mut self,
        kind: Kind,
        num_collector_units: usize,
        num_dispatch_units: usize,
    ) {
        let set = self.collector_unit_sets.entry(kind).or_default();

        for id in 0..num_collector_units {
            let unit = CollectorUnit::new(kind, id);
            set.push(id);
            self.collector_units.push(unit);
        }
        // each collector set gets dedicated dispatch units.
        for id in 0..num_dispatch_units {
            let dispatch_unit = DispatchUnit::new(kind, id);
            self.dispatch_units.push(dispatch_unit);
        }
    }

    pub fn add_port(&mut self, input: PortVec, output: PortVec, cu_sets: Vec<Kind>) {
        self.in_ports.push(InputPort::new(input, output, cu_sets));
    }
}

impl Writeback for RegisterFileUnit {
    fn writeback(&mut self, instr: &mut WarpInstruction) -> bool {
        log::trace!(
            "operand collector: writeback {} with destination registers {:?}",
            instr,
            instr
                .dest_arch_reg
                .iter()
                .filter_map(Option::as_ref)
                .collect::<Vec<_>>(),
        );

        for op in 0..MAX_REG_OPERANDS {
            let Some(reg_num) = instr.dest_arch_reg[op] else {
                continue;
            };
            let scheduler_id = instr.scheduler_id.unwrap();

            // compute the register file bank for this operand
            let bank = compute_register_bank(
                reg_num,
                instr.warp_id,
                self.num_banks,
                self.bank_warp_shift,
                self.sub_core_model,
                self.num_banks_per_scheduler,
                scheduler_id,
            );

            // check if the bank is idle
            // (not used by any other thread in this cycle)
            let bank_idle = self.arbiter.bank_idle(bank);
            log::trace!(
                "operand collector: writeback {}: destination register {:>2}: scheduler id={} bank={} (idle={})",
                instr, reg_num, scheduler_id, bank, bank_idle,
            );

            if bank_idle {
                // allocate a write on the bank
                self.arbiter.allocate_bank_for_write(
                    bank,
                    DestinationOperand {
                        warp_id: instr.warp_id,
                        register: reg_num,
                        scheduler_id,
                        bank,
                    },
                );

                // can now remove the operand
                instr.dest_arch_reg[op] = None;
            } else {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod test {
    use crate::{register_set, testing};
    use bitvec::{array::BitArray, BitArr};
    use trace_model::ToBitString;

    impl From<super::Kind> for testing::state::OperandCollectorUnitKind {
        fn from(id: super::Kind) -> Self {
            match id {
                super::Kind::SP_CUS => testing::state::OperandCollectorUnitKind::SP_CUS,
                super::Kind::DP_CUS => testing::state::OperandCollectorUnitKind::DP_CUS,
                super::Kind::SFU_CUS => testing::state::OperandCollectorUnitKind::SFU_CUS,
                super::Kind::TENSOR_CORE_CUS => {
                    testing::state::OperandCollectorUnitKind::TENSOR_CORE_CUS
                }
                super::Kind::INT_CUS => testing::state::OperandCollectorUnitKind::INT_CUS,
                super::Kind::MEM_CUS => testing::state::OperandCollectorUnitKind::MEM_CUS,
                super::Kind::GEN_CUS => testing::state::OperandCollectorUnitKind::GEN_CUS,
            }
        }
    }

    impl testing::state::Port {
        pub fn new(port: super::InputPort, pipeline_reg: &[register_set::RegisterSet]) -> Self {
            Self {
                ids: port
                    .collector_unit_ids
                    .iter()
                    .copied()
                    .map(Into::into)
                    .collect(),
                in_ports: port
                    .in_ports
                    .iter()
                    .map(|p| pipeline_reg[*p as usize].clone().into())
                    .collect(),
                out_ports: port
                    .out_ports
                    .iter()
                    .map(|p| pipeline_reg[*p as usize].clone().into())
                    .collect(),
            }
        }
    }

    impl testing::state::CollectorUnit {
        pub fn new(cu: &super::CollectorUnit, pipeline_reg: &[register_set::RegisterSet]) -> Self {
            match cu.pending {
                Some(ref pending) => Self {
                    // warp_id: if cu.is_free {
                    //     None
                    // } else {
                    //     Some(pending.warp_id)
                    // },
                    // warp_id: Some(pending.warp_id),
                    warp_id: Some(pending.warp_instr.warp_id),
                    warp_instr: Some(pending.warp_instr.clone().into()),
                    /// pipeline register to issue to when ready
                    output_register: Some(
                        pipeline_reg[pending.output_register as usize]
                            .clone()
                            .into(),
                    ),
                    not_ready: pending.not_ready.to_bit_string(),
                    reg_id: cu.reg_id,
                    kind: cu.kind.into(),
                },
                _ => Self {
                    warp_id: None,
                    warp_instr: None,
                    /// pipeline register to issue to when ready
                    output_register: None,
                    not_ready: <BitArr!(for super::MAX_REG_OPERANDS * 2)>::ZERO.to_bit_string(),
                    reg_id: cu.reg_id,
                    kind: cu.kind.into(),
                },
            }
            // Self {
            //     warp_id: cu.warp_id(),
            //     warp_instr: cu.warp_instr.clone().map(Into::into),
            //     /// pipeline register to issue to when ready
            //     output_register: cu
            //         .output_register
            //         .as_ref()
            //         .map(|r| pipeline_reg[*r as usize].clone().into()),
            //     not_ready: cu.not_ready.to_bit_string(),
            //     reg_id: cu.reg_id,
            //     kind: cu.kind.into(),
            // }
        }
    }

    impl From<&super::DispatchUnit> for testing::state::DispatchUnit {
        fn from(unit: &super::DispatchUnit) -> Self {
            Self {
                last_cu: unit.last_collector_unit,
                next_cu: unit.next_collector_unit,
                kind: unit.kind.into(),
            }
        }
    }

    impl From<&super::Arbiter> for testing::state::Arbiter {
        fn from(arbiter: &super::Arbiter) -> Self {
            Self {
                last_cu: arbiter.last_cu,
            }
        }
    }

    // impl From<&super::RegisterFileUnit> for testing::state::OperandCollector {
    //     fn from(opcoll: &super::RegisterFileUnit) -> Self {
    impl testing::state::OperandCollector {
        pub fn new(
            opcoll: &super::RegisterFileUnit,
            pipeline_reg: &[register_set::RegisterSet],
        ) -> Self {
            let dispatch_units = opcoll.dispatch_units.iter().map(Into::into).collect();
            let collector_units = opcoll
                .collector_units
                .iter()
                // .map(|cu| cu.try_lock().deref().into())
                // .map(|cu| testing::state::CollectorUnit::new(&*cu.try_lock(), pipeline_reg))
                .map(|cu| testing::state::CollectorUnit::new(cu, pipeline_reg))
                .collect();
            let ports = opcoll
                .in_ports
                .iter()
                // .map(Into::into)
                .map(|port| testing::state::Port::new(port.clone(), pipeline_reg))
                .collect();
            let arbiter = (&opcoll.arbiter).into();
            Self {
                ports,
                collector_units,
                dispatch_units,
                arbiter,
            }
        }
    }
}
