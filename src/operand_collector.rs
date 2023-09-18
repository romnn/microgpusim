use super::{config, instruction::WarpInstruction, register_set};
use bitvec::{array::BitArray, BitArr};
use console::style;
use register_set::Access;
use trace_model::ToBitString;
use utils::box_slice;

use std::collections::{HashMap, VecDeque};

use crate::sync::{Arc, Mutex};

pub const MAX_REG_OPERANDS: usize = 32;

fn register_bank(
    reg_num: u32,
    warp_id: usize,
    num_banks: usize,
    bank_warp_shift: usize,
    sub_core_model: bool,
    banks_per_sched: usize,
    sched_id: usize,
) -> usize {
    let mut bank = reg_num as usize;
    if bank_warp_shift > 0 {
        bank += warp_id;
    }
    if sub_core_model {
        let bank_num = (bank % banks_per_sched) + (sched_id * banks_per_sched);
        debug_assert!(bank_num < num_banks);
        bank_num
    } else {
        bank % num_banks
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Operand {
    pub warp_id: Option<usize>,
    pub operand: Option<usize>,
    pub register: u32,
    pub bank: usize,
    pub scheduler_id: usize,
    pub collector_unit_id: Option<usize>,
}

// impl Operand {
//     #[must_use]
//     pub fn new(
//         warp_id: Option<usize>,
//         cu_id: usize,
//         op: usize,
//         reg: u32,
//         bank: usize,
//         scheduler_id: usize,
//     ) -> Self {
//         Self {
//             bank,
//             warp_id,
//             operand: Some(op),
//             register: reg,
//             scheduler_id,
//             collector_unit_id: Some(cu_id),
//         }
//     }
//
//     #[must_use]
//     pub fn warp_id(&self) -> Option<usize> {
//         self.warp_id
//     }
// }

#[derive(Debug)]
pub struct CollectorUnit {
    free: bool,
    kind: Kind,
    /// collector unit hw id
    id: usize,
    warp_id: Option<usize>,
    warp_instr: Option<WarpInstruction>,
    /// pipeline register to issue to when ready
    output_register: Option<register_set::Ref>,
    src_operands: [Option<Operand>; MAX_REG_OPERANDS * 2],
    not_ready: BitArr!(for MAX_REG_OPERANDS * 2),
    num_banks: usize,
    bank_warp_shift: usize,
    sub_core_model: bool,
    num_banks_per_scheduler: usize,
    /// if sub_core_model enabled, limit regs this cu can r/w
    reg_id: usize,
}

impl CollectorUnit {
    fn new(kind: Kind) -> Self {
        let src_operands = [(); MAX_REG_OPERANDS * 2].map(|_| None);
        Self {
            id: 0,
            free: true,
            kind,
            warp_instr: None,
            output_register: None,
            src_operands,
            not_ready: BitArray::ZERO,
            warp_id: None,
            num_banks: 0,
            bank_warp_shift: 0,
            num_banks_per_scheduler: 0,
            reg_id: 0,
            sub_core_model: false,
        }
    }

    pub fn init(
        &mut self,
        id: usize,
        num_banks: usize,
        log2_warp_size: usize,
        sub_core_model: bool,
        reg_id: usize,
        banks_per_scheduler: usize,
    ) {
        self.id = id;
        self.num_banks = num_banks;
        debug_assert!(self.warp_instr.is_none());
        self.warp_instr = None;
        self.bank_warp_shift = log2_warp_size;
        self.sub_core_model = sub_core_model;
        self.reg_id = reg_id;
        self.num_banks_per_scheduler = banks_per_scheduler;
    }

    // looks ok
    #[must_use]
    pub fn ready(&self) -> bool {
        if self.free {
            return false;
        }
        let Some(output_register) = self.output_register.as_ref() else {
            return false;
        };
        let output_register = output_register.try_lock();
        let has_free_register = if self.sub_core_model {
            // output_register.has_free_sub_core(self.reg_id)
            unimplemented!("sub core model")
        } else {
            output_register.has_free()
        };
        log::debug!(
            "is ready?: active = {} (ready={}), has free = {} output register = {:?}",
            self.not_ready.to_bit_string(),
            self.not_ready.not_any(),
            has_free_register,
            &output_register
        );

        !self.free && self.not_ready.not_any() && has_free_register
    }

    // looks ok
    pub fn dispatch(&mut self) {
        debug_assert!(self.not_ready.not_any());
        let output_register = self.output_register.take().unwrap();
        let mut output_register = output_register.try_lock();

        let warp_instr = self.warp_instr.take();

        // TODO HOTFIX: workaround
        self.warp_id = None;
        self.reg_id = 0;

        // let mut output_register = self.output_register.as_mut().unwrap();

        if self.sub_core_model {
            // let msg = format!(
            //     "operand collector: move warp instr {:?} to output register (reg_id={})",
            //     warp_instr.as_ref().map(ToString::to_string),
            //     self.reg_id,
            // );
            // output_register.move_in_from_sub_core(self.reg_id, warp_instr);
            unimplemented!("sub core model")
        } else {
            // let msg = format!(
            //     "operand collector: move warp instr {:?} to output register",
            //     warp_instr.as_ref().map(ToString::to_string),
            // );
            output_register.move_in_from(warp_instr);
        }

        self.free = true;
        self.output_register = None;
        self.src_operands.fill(None);
    }

    fn allocate(
        &mut self,
        input_reg_set: &register_set::Ref,
        output_reg_set: &register_set::Ref,
    ) -> bool {
        log::debug!(
            "{}",
            style(format!("operand collector::allocate({:?})", self.kind)).green(),
        );

        debug_assert!(self.free);
        debug_assert!(self.not_ready.not_any());

        self.free = false;
        self.output_register = Some(Arc::clone(output_reg_set));
        let mut input_reg_set = input_reg_set.try_lock();

        if let Some((_, Some(ready_reg))) = input_reg_set.get_ready() {
            self.warp_id = Some(ready_reg.warp_id); // todo: do we need warp id??

            log::debug!(
                "operand collector::allocate({:?}) => src arch reg = {:?}",
                self.kind,
                ready_reg
                    .src_arch_reg
                    .iter()
                    .map(|r| r.map(i64::from).unwrap_or(-1))
                    .collect::<Vec<i64>>(),
            );

            // remove duplicate regs within same instr
            let mut prev_regs: Vec<u32> = Vec::new();
            for op in 0..MAX_REG_OPERANDS {
                // this math needs to match that used in function_info::ptx_decode_inst
                if let Some(reg_num) = ready_reg.src_arch_reg[op] {
                    let is_new_reg = !prev_regs.contains(&reg_num);
                    if is_new_reg {
                        // valid register
                        prev_regs.push(reg_num);
                        let scheduler_id = ready_reg.scheduler_id.unwrap();
                        let bank = register_bank(
                            reg_num,
                            ready_reg.warp_id,
                            self.num_banks,
                            self.bank_warp_shift,
                            self.sub_core_model,
                            self.num_banks_per_scheduler,
                            scheduler_id,
                        );

                        self.src_operands[op] = Some(Operand {
                            warp_id: self.warp_id,
                            collector_unit_id: Some(self.id),
                            operand: Some(op),
                            register: reg_num,
                            bank,
                            scheduler_id,
                        });
                        self.not_ready.set(op, true);
                    } else {
                        self.src_operands[op] = None;
                    }
                }
            }
            log::debug!(
                "operand collector::allocate({:?}) => active: {}",
                self.kind,
                self.not_ready.to_bit_string(),
            );

            // let msg = format!(
            //     "operand collector: move input register {} to warp instruction {:?}",
            //     &input_reg_set,
            //     self.warp_instr.as_ref().map(ToString::to_string),
            // );
            input_reg_set.move_out_to(&mut self.warp_instr);
            true
        } else {
            false
        }
    }

    pub fn collect_operand(&mut self, op: usize) {
        log::debug!(
            "collector unit [{}] {} collecting operand for {}",
            self.id,
            crate::Optional(self.warp_instr.as_ref()),
            op,
        );
        self.not_ready.set(op, false);
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum AllocationKind {
    NO_ALLOC,
    READ_ALLOC,
    WRITE_ALLOC,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Allocation {
    kind: AllocationKind,
    op: Option<Operand>,
}

impl Default for Allocation {
    fn default() -> Self {
        Self {
            kind: AllocationKind::NO_ALLOC,
            op: None,
        }
    }
}

impl Allocation {
    #[must_use]
    pub fn new(kind: AllocationKind, op: Option<Operand>) -> Self {
        Self { kind, op }
    }

    #[must_use]
    pub fn is_read(&self) -> bool {
        self.kind == AllocationKind::READ_ALLOC
    }

    #[must_use]
    pub fn is_write(&self) -> bool {
        self.kind == AllocationKind::WRITE_ALLOC
    }

    #[must_use]
    pub fn is_free(&self) -> bool {
        self.kind == AllocationKind::NO_ALLOC
    }

    pub fn allocate_for_read(&mut self, op: Option<Operand>) {
        debug_assert!(self.is_free());
        self.kind = AllocationKind::READ_ALLOC;
        self.op = op;
    }

    pub fn allocate_for_write(&mut self, op: Option<Operand>) {
        debug_assert!(self.is_free());
        self.kind = AllocationKind::WRITE_ALLOC;
        self.op = op;
    }

    pub fn reset(&mut self) {
        self.kind = AllocationKind::NO_ALLOC;
        self.op = None;
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Arbiter {
    num_banks: usize,
    num_collectors: usize,

    bank_warp_shift: usize,
    sub_core_model: bool,
    num_banks_per_scheduler: usize,

    /// bank # -> register that wins
    allocated_banks: Box<[Allocation]>,
    queue: Box<[VecDeque<Operand>]>,
    // allocated: Vec<Operand>,
    /// cu # -> next bank to check for request (rr-arb)
    // allocator_round_robin_head: usize,
    /// first cu to check while arb-ing banks (rr)
    last_cu: usize,
    inmatch: Box<[Option<usize>]>,
    outmatch: Box<[Option<usize>]>,
    request: Box<[Box<[Option<usize>]>]>,
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

        self.inmatch = box_slice![None; self.num_banks];
        self.outmatch = box_slice![None; self.num_collectors];
        self.request = box_slice![box_slice![None; self.num_collectors]; self.num_banks];

        self.queue = box_slice![VecDeque::new(); self.num_banks];
        self.allocated_banks = box_slice![Allocation::default(); self.num_banks];

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

    /// Allocate reads
    ///
    /// a list of registers that
    ///  (a) are in different register banks,
    ///  (b) do not go to the same operand collector
    ///
    /// The outcomes of this depend on the queue
    // pub fn allocate_reads(&mut self) -> &Vec<Operand> {
    pub fn allocate_reads(&mut self) -> HashMap<usize, Operand> {
        // log::trace!("queue: {:?}", &self.queue);

        let num_inputs = self.num_banks;
        let num_outputs = self.num_collectors;
        let square = if num_inputs > num_outputs {
            num_inputs
        } else {
            num_outputs
        };
        debug_assert!(square > 0);

        let last_cu_before = self.last_cu;
        let mut pri = self.last_cu;
        // log::debug!("last cu: {}", self.last_cu);

        let no_allocation = self
            .allocated_banks
            .iter()
            .all(|alloc| alloc.kind == AllocationKind::NO_ALLOC);
        let empty_queue = self.queue.iter().all(std::collections::VecDeque::is_empty);

        // fast path
        if no_allocation && empty_queue {
            self.last_cu = (self.last_cu + 1) % num_outputs;
            return HashMap::new();
        }

        // clear matching
        let mut allocated = Vec::new();
        // let result = &mut self.result;
        let inmatch = &mut self.inmatch;
        // let outmatch = &mut self.outmatch;
        let request = &mut self.request;

        // allocated.clear();
        inmatch.fill(None);
        // outmatch.fill(None);

        for bank in 0..self.num_banks {
            debug_assert!(bank < num_inputs);
            for collector in 0..self.num_collectors {
                debug_assert!(collector < num_outputs);
                request[bank][collector] = Some(0);
            }
            if let Some(op) = self.queue[bank].front() {
                let collector_id = op.collector_unit_id.unwrap();
                debug_assert!(collector_id < num_outputs);
                // this causes change in search
                request[bank][collector_id] = Some(1);
            }
            if self.allocated_banks[bank].is_write() {
                inmatch[bank] = Some(0); // write gets priority
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

        for p in 0..square {
            let mut output = (pri + p) % num_outputs;

            // step through the current diagonal
            for input in 0..num_inputs {
                debug_assert!(output < num_outputs);

                // banks at the same cycle
                assert!(output < num_outputs);
                if inmatch[input].is_none() && request[input][output] != Some(0) {
                    // Grant!
                    inmatch[input] = Some(output);
                    // outmatch[output] = Some(input);
                    // printf("Register File: granting bank %d to OC %d, schedid %d, warpid
                    // %d, Regid %d\n", input, output, (m_queue[input].front()).get_sid(),
                    // (m_queue[input].front()).get_wid(),
                    // (m_queue[input].front()).get_reg());
                }

                output = (output + 1) % num_outputs;
            }
        }

        log::trace!("inmatch: {:?}", &Self::compat(inmatch));
        // log::trace!("outmatch: {:?}", &Self::compat(outmatch));

        // Round-robin the priority diagonal
        pri = (pri + 1) % num_outputs;
        log::trace!("pri: {:?}", pri);

        // <--- end code from booksim

        self.last_cu = pri;
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
                    self.allocated_banks[bank].is_write()
                );
                if !self.allocated_banks[bank].is_write() {
                    if let Some(op) = self.queue[bank].pop_front() {
                        allocated.push(op);
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

        // allocated

        log::debug!(
            "arbiter allocated {} reads ({:?})",
            allocated.len(),
            &allocated
        );
        let mut read_ops = HashMap::new();
        for read in allocated {
            let reg = read.register;
            let warp_id = read.warp_id.unwrap();
            let bank = register_bank(
                reg,
                warp_id,
                self.num_banks,
                self.bank_warp_shift,
                self.sub_core_model,
                self.num_banks_per_scheduler,
                read.scheduler_id,
            );
            self.allocate_bank_for_read(bank, read.clone());
            read_ops.insert(bank, read);
        }
        // #[cfg(feature = "timings")]
        // {
        //     crate::TIMINGS
        //         .lock()
        //         .entry("allocate_reads_register_banks")
        //         .or_default()
        //         .add(start.elapsed());
        // }

        read_ops
    }

    pub fn add_read_requests(&mut self, cu: &CollectorUnit) {
        for src_op in cu.src_operands.iter().flatten() {
            let bank = src_op.bank;
            self.queue[bank].push_back(src_op.clone());
        }
    }

    #[must_use]
    pub fn bank_idle(&self, bank: usize) -> bool {
        self.allocated_banks[bank].is_free()
    }

    pub fn allocate_bank_for_write(&mut self, bank: usize, op: Operand) {
        debug_assert!(bank < self.num_banks);
        self.allocated_banks[bank].allocate_for_write(Some(op));
    }

    pub fn allocate_bank_for_read(&mut self, bank: usize, op: Operand) {
        debug_assert!(bank < self.num_banks);
        self.allocated_banks[bank].allocate_for_read(Some(op));
    }

    pub fn reset_alloction(&mut self) {
        for bank in &mut *self.allocated_banks {
            bank.reset();
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DispatchUnit {
    last_cu: usize,
    next_cu: usize,
    sub_core_model: bool,
    num_warp_schedulers: usize,
    kind: Kind,
}

impl DispatchUnit {
    #[must_use]
    pub fn new(kind: Kind) -> Self {
        Self {
            kind,
            last_cu: 0,
            next_cu: 0,
            sub_core_model: false,
            num_warp_schedulers: 0,
        }
    }

    pub fn init(&mut self, sub_core_model: bool, num_warp_schedulers: usize) {
        self.sub_core_model = sub_core_model;
        self.num_warp_schedulers = num_warp_schedulers;
    }

    pub fn find_ready<'a>(
        &mut self,
        collector_units: &'a Vec<Arc<Mutex<CollectorUnit>>>,
    ) -> Option<&'a Arc<Mutex<CollectorUnit>>> {
        // With sub-core enabled round robin starts with the next cu assigned to a
        // different sub-core than the one that dispatched last
        let num_collector_units = collector_units.len();
        let cus_per_scheduler = num_collector_units / self.num_warp_schedulers;
        let rr_increment = if self.sub_core_model {
            cus_per_scheduler - (self.last_cu % cus_per_scheduler)
        } else {
            1
        };

        log::debug!("dispatch unit {:?}: find ready: rr_inc = {},last cu = {},num collectors = {}, num warp schedulers = {}, cusPerSched = {}", self.kind, rr_increment, self.last_cu, num_collector_units, self.num_warp_schedulers, cus_per_scheduler);

        debug_assert_eq!(num_collector_units, collector_units.len());
        for i in 0..num_collector_units {
            let i = (self.last_cu + i + rr_increment) % num_collector_units;
            // log::trace!(
            //     "dispatch unit {:?}: checking collector unit {}",
            //     self.kind,
            //     i,
            // );

            if collector_units[i].try_lock().ready() {
                self.last_cu = i;
                log::debug!(
                    "dispatch unit {:?}: FOUND ready: chose collector unit {} ({:?})",
                    self.kind,
                    i,
                    collector_units[i].try_lock().kind
                );
                return collector_units.get(i);
            }
        }
        log::debug!("dispatch unit {:?}: did NOT find ready", self.kind);
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

pub type CuSets = HashMap<Kind, Vec<Arc<Mutex<CollectorUnit>>>>;

// operand collector based register file unit
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
    pub in_ports: VecDeque<InputPort>,
    pub collector_units: Vec<Arc<Mutex<CollectorUnit>>>,
    pub collector_unit_sets: CuSets,
    pub dispatch_units: Vec<DispatchUnit>,
}

pub type PortVec = Vec<register_set::Ref>;

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
            in_ports: VecDeque::new(),
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
        let mut reg_id = 0;
        for (cu_id, cu) in self.collector_units.iter().enumerate() {
            if self.sub_core_model {
                let coll_units_per_scheduler = num_collector_units / self.num_warp_schedulers;
                reg_id = cu_id / coll_units_per_scheduler;
            }
            let mut cu = cu.try_lock();
            cu.init(
                cu_id,
                self.num_banks,
                self.bank_warp_shift,
                self.sub_core_model,
                reg_id,
                self.num_banks_per_scheduler,
            );

            debug_assert!(cu.id == cu_id);
        }
        for dispatch_unit in &mut self.dispatch_units {
            dispatch_unit.init(self.sub_core_model, self.num_warp_schedulers);
        }
        self.initialized = true;
    }

    pub fn step(&mut self) {
        log::debug!("{}", style("operand collector::step()").green());
        self.dispatch_ready_cu();
        self.allocate_reads();

        debug_assert!(!self.in_ports.is_empty());
        for port_num in 0..self.in_ports.len() {
            self.allocate_cu(port_num);
        }
        self.process_banks();
    }

    fn process_banks(&mut self) {
        self.arbiter.reset_alloction();
    }

    /// Process read requests that do not have conflicts
    pub fn allocate_reads(&mut self) {
        // process read requests that do not have conflicts
        let read_ops = self.arbiter.allocate_reads();

        log::debug!("allocating {} reads ({:?})", read_ops.len(), &read_ops);
        for read in read_ops.values() {
            let cu_id = read.collector_unit_id.unwrap();
            assert!(cu_id < self.collector_units.len());
            let mut cu = self.collector_units[cu_id].try_lock();
            if let Some(operand) = read.operand {
                cu.collect_operand(operand);
            }

            // if self.config.clock_gated_reg_file {
            //     let mut active_count = 0;
            //     let mut thread_id = 0;
            //     while thread_id < self.config.warp_size {
            //         for i in 0..self.config.n_regfile_gating_group {
            //             if read.active_mask[thread_id + i] {
            //                 active_count += self.config.n_regfile_gating_group;
            //                 break;
            //             }
            //         }
            //
            //         thread_id += self.config.n_regfile_gating_group;
            //     }
            //     // self.stats.incregfile_reads(active_count);
            // } else {
            //     // self.stats.incregfile_reads(self.config.warp_size);
            // }
        }
    }

    pub fn allocate_cu(&mut self, port_num: usize) {
        let port = &self.in_ports[port_num];
        log::debug!(
            "{}",
            style(format!(
                "operand collector::allocate_cu({:?}: {:?})",
                port_num, port.collector_unit_ids
            ))
            .green()
        );

        debug_assert_eq!(port.in_ports.len(), port.out_ports.len());

        for (input_port, output_port) in port.in_ports.iter().zip(port.out_ports.iter()) {
            if input_port.try_lock().has_ready() {
                // find a free collector unit
                for cu_set_id in &port.collector_unit_ids {
                    let cu_set: &Vec<_> = &self.collector_unit_sets[cu_set_id];
                    let mut allocated = false;
                    let cu_lower_bound = 0;
                    let cu_upper_bound = cu_set.len();

                    if self.sub_core_model {
                        // sub core model only allocates on the subset of CUs assigned
                        // to the scheduler that issued
                        // let (reg_id, _) = input_port.try_lock().get_ready().unwrap();
                        // debug_assert!(
                        //     cu_set.len() % self.num_warp_schedulers == 0
                        //         && cu_set.len() >= self.num_warp_schedulers
                        // );
                        // let cus_per_sched = cu_set.len() / self.num_warp_schedulers;
                        // let schd_id = input_port.try_lock().scheduler_id(reg_id).unwrap();
                        // cu_lower_bound = schd_id * cus_per_sched;
                        // cu_upper_bound = cu_lower_bound + cus_per_sched;
                        // debug_assert!(cu_upper_bound <= cu_set.len());
                        unimplemented!("sub core model")
                    }

                    for collector_unit in &cu_set[cu_lower_bound..cu_upper_bound] {
                        let mut collector_unit = collector_unit.try_lock();

                        if collector_unit.free {
                            log::debug!(
                                "{} cu={:?}",
                                style("operand collector::allocate()".to_string()).green(),
                                collector_unit.kind
                            );

                            allocated = collector_unit.allocate(input_port, output_port);
                            self.arbiter.add_read_requests(&collector_unit);
                            break;
                        }
                    }

                    if allocated {
                        break; // cu has been allocated, no need to search more.
                    }
                }
            }
        }
    }

    pub fn dispatch_ready_cu(&mut self) {
        for dispatch_unit in &mut self.dispatch_units {
            let set = &self.collector_unit_sets[&dispatch_unit.kind];
            if let Some(collector_unit) = dispatch_unit.find_ready(set) {
                collector_unit.try_lock().dispatch();
            }
        }
    }

    pub fn writeback(&mut self, instr: &mut WarpInstruction) -> bool {
        let regs: Vec<u32> = instr.dest_arch_reg.iter().filter_map(|reg| *reg).collect();
        log::trace!(
            "operand collector: writeback {} with destination registers {:?}",
            instr,
            regs
        );

        for op in 0..MAX_REG_OPERANDS {
            // this math needs to match that used in function_info::ptx_decode_inst
            if let Some(reg_num) = instr.dest_arch_reg[op] {
                let scheduler_id = instr.scheduler_id.unwrap();
                let bank = register_bank(
                    reg_num,
                    instr.warp_id,
                    self.num_banks,
                    self.bank_warp_shift,
                    self.sub_core_model,
                    self.num_banks_per_scheduler,
                    scheduler_id,
                );
                let bank_idle = self.arbiter.bank_idle(bank);
                log::trace!(
                  "operand collector: writeback {}: destination register {:>2}: scheduler id={} bank={} (idle={})",
                  instr, reg_num, scheduler_id, bank, bank_idle);

                if bank_idle {
                    self.arbiter.allocate_bank_for_write(
                        bank,
                        Operand {
                            warp_id: Some(instr.warp_id),
                            register: reg_num,
                            scheduler_id,
                            operand: None,
                            bank,
                            collector_unit_id: None,
                        },
                    );
                    instr.dest_arch_reg[op] = None;
                } else {
                    return false;
                }
            }
        }
        true
    }

    pub fn add_cu_set(
        &mut self,
        kind: Kind,
        num_collector_units: usize,
        num_dispatch_units: usize,
    ) {
        let set = self.collector_unit_sets.entry(kind).or_default();

        for _ in 0..num_collector_units {
            let unit = Arc::new(Mutex::new(CollectorUnit::new(kind)));
            set.push(Arc::clone(&unit));
            self.collector_units.push(unit);
        }
        // each collector set gets dedicated dispatch units.
        for _ in 0..num_dispatch_units {
            let dispatch_unit = DispatchUnit::new(kind);
            self.dispatch_units.push(dispatch_unit);
        }
    }

    pub fn add_port(&mut self, input: PortVec, output: PortVec, cu_sets: Vec<Kind>) {
        self.in_ports
            .push_back(InputPort::new(input, output, cu_sets));
    }
}

#[cfg(test)]
mod test {
    use crate::{testing};
    use std::ops::Deref;
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

    impl From<&super::InputPort> for testing::state::Port {
        fn from(port: &super::InputPort) -> Self {
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
                    .map(|p| p.try_lock().clone().into())
                    .collect(),
                out_ports: port
                    .out_ports
                    .iter()
                    .map(|p| p.try_lock().clone().into())
                    .collect(),
            }
        }
    }

    impl From<&super::CollectorUnit> for testing::state::CollectorUnit {
        fn from(cu: &super::CollectorUnit) -> Self {
            Self {
                warp_id: cu.warp_id,
                warp_instr: cu.warp_instr.clone().map(Into::into),
                /// pipeline register to issue to when ready
                output_register: cu
                    .output_register
                    .as_ref()
                    .map(|r| r.try_lock().deref().clone().into()),
                // src_operands: [Option<Operand>; MAX_REG_OPERANDS * 2],
                not_ready: cu.not_ready.to_bit_string(),
                reg_id: if cu.warp_id.is_some() {
                    Some(cu.reg_id)
                } else {
                    None
                },
                kind: cu.kind.into(),
            }
        }
    }

    impl From<&super::DispatchUnit> for testing::state::DispatchUnit {
        fn from(unit: &super::DispatchUnit) -> Self {
            Self {
                last_cu: unit.last_cu,
                next_cu: unit.next_cu,
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

    impl From<&super::RegisterFileUnit> for testing::state::OperandCollector {
        fn from(opcoll: &super::RegisterFileUnit) -> Self {
            let dispatch_units = opcoll.dispatch_units.iter().map(Into::into).collect();
            let collector_units = opcoll
                .collector_units
                .iter()
                .map(|cu| cu.try_lock().deref().into())
                .collect();
            let ports = opcoll.in_ports.iter().map(Into::into).collect();
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
