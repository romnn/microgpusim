use super::{instruction::WarpInstruction, register_set::RegisterSet, scheduler as sched};
use crate::config;
use crate::ported::mem_fetch::BitString;
use bitvec::{array::BitArray, BitArr};
use console::style;
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use std::sync::Arc;

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
    // valid: bool,
    // collector_unit: Option<CollectorUnit>,
    // TODO: removing warp instr
    warp_id: Option<usize>,
    // warp_instr: Option<WarpInstruction>,
    /// operand offset in instruction. e.g., add r1,r2,r3;
    /// r2 is oprd 0, r3 is 1 (r1 is dst)
    operand: Option<usize>,

    register: u32,
    bank: usize,
    /// scheduler id that has issued this inst
    // core_id: usize,
    scheduler_id: usize,
    collector_unit_id: usize,
    // collector_unit: Option<Weak<RefCell<CollectorUnit>>>,
}

impl Operand {
    pub fn new(
        // cu: Weak<RefCell<CollectorUnit>>,
        warp_id: Option<usize>,
        cu_id: usize,
        op: usize,
        reg: u32,
        bank: usize,
        scheduler_id: usize,
    ) -> Self {
        Self {
            bank,
            warp_id,
            // warp_instr: None,
            operand: Some(op),
            register: reg,
            scheduler_id,
            // collector_unit: Some(cu),
            collector_unit_id: cu_id,
        }
    }

    // pub fn collector_unit(&self) -> Option<Rc<RefCell<CollectorUnit>>> {
    //     let collector_unit = self.collector_unit.as_ref()?;
    //     let collector_unit = Weak::upgrade(collector_unit)?;
    //     Some(collector_unit)
    // }

    pub fn warp_id(&self) -> Option<usize> {
        self.warp_id
    }

    // pub fn warp_id(&self) -> Option<usize> {
    //     self.warp_instr.as_ref().map(|warp| warp.warp_id)
    // }

    // TODO: do we need warp instruction?
    // pub fn warp_id(&self) -> Option<usize> {
    //     if let Some(ref warp) = self.warp_instr {
    //         Some(warp.warp_id)
    //     } else if let Some(cu) = self.collector_unit() {
    //         cu.borrow().warp_id
    //     } else {
    //         None
    //     }
    // }
}

#[derive(Debug, PartialEq, Eq)]
pub struct CollectorUnit {
    free: bool,
    kind: OperandCollectorUnitKind,
    /// collector unit hw id
    id: usize,
    warp_id: Option<usize>,
    warp_instr: Option<WarpInstruction>,
    /// pipeline register to issue to when ready
    output_register: Option<Rc<RefCell<RegisterSet>>>,
    src_operands: [Option<Operand>; MAX_REG_OPERANDS * 2],
    not_ready: BitArr!(for MAX_REG_OPERANDS * 2),
    num_banks: usize,
    bank_warp_shift: usize,
    // opndcoll_rfu_t *m_rfu;
    num_banks_per_scheduler: usize,
    sub_core_model: bool,
    /// if sub_core_model enabled, limit regs this cu can r/w
    reg_id: usize,
}

// impl Default for CollectorUnit {
//     fn default() -> Self {
//         Self {
//             id: 0,
//             free: true,
//             warp_instr: None,
//             // output_register: None,
//             // src_op = new op_t[MAX_REG_OPERANDS * 2];
//             not_ready: BitArray::ZERO,
//             warp_id: None,
//             num_banks: 0,
//             bank_warp_shift: 0,
//             num_banks_per_sched: 0,
//             reg_id: 0,
//             sub_core_model: false,
//         }
//     }
// }

impl CollectorUnit {
    fn new(kind: OperandCollectorUnitKind) -> Self {
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
        config: Arc<config::GPUConfig>,
        sub_core_model: bool,
        reg_id: usize,
        banks_per_scheduler: usize,
    ) {
        self.id = id;
        self.num_banks = num_banks;
        debug_assert!(self.warp_instr.is_none());
        // self.warp_instr = Some(WarpInstruction::new_empty(&*config));
        self.warp_instr = None;
        self.bank_warp_shift = log2_warp_size;
        self.sub_core_model = sub_core_model;
        self.reg_id = reg_id;
        self.num_banks_per_scheduler = banks_per_scheduler;
    }

    // looks ok
    pub fn ready(&self) -> bool {
        if self.free {
            return false;
        }
        let Some(output_register) = self.output_register.as_ref() else {
            return false;
        };
        let output_register = output_register.borrow();
        let has_free_register = if self.sub_core_model {
            output_register.has_free_sub_core(self.reg_id)
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
        let mut output_register = output_register.borrow_mut();

        let warp_instr = self.warp_instr.take();

        // TODO HOTFIX: workaround
        self.warp_id = None;
        self.reg_id = 0;

        // let mut output_register = self.output_register.as_mut().unwrap();

        if self.sub_core_model {
            let msg = format!(
                "operand collector: move warp instr {:?} to output register (reg_id={})",
                warp_instr.as_ref().map(ToString::to_string),
                self.reg_id,
            );

            output_register.move_in_from_sub_core(self.reg_id, warp_instr, msg);
        } else {
            let msg = format!(
                "operand collector: move warp instr {:?} to output register",
                warp_instr.as_ref().map(ToString::to_string),
            );
            output_register.move_in_from(warp_instr, msg);
        }

        self.free = true;
        self.output_register = None;
        self.src_operands.fill(None);
    }

    // looks ok
    fn allocate(
        &mut self,
        // cu is ourselves??
        // cu: Weak<RefCell<CollectorUnit>>,
        input_reg_set: &Rc<RefCell<RegisterSet>>,
        output_reg_set: &Rc<RefCell<RegisterSet>>,
    ) -> bool {
        log::debug!(
            "{}",
            style(format!("operand collector::allocate({:?})", self.kind)).green(),
        );

        debug_assert!(self.free);
        debug_assert!(self.not_ready.not_any());

        self.free = false;
        self.output_register = Some(Rc::clone(output_reg_set));
        let mut input_reg_set = input_reg_set.borrow_mut();

        if let Some((_, Some(ready_reg))) = input_reg_set.get_ready() {
            // if ((ready_reg) and !((*pipeline_reg)->empty())) {
            // debug_assert!(!ready_reg.empty());

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
                    // let mut is_new_reg = true;
                    // is_new_reg &= !prev_regs.contains(&reg_num);
                    let is_new_reg = !prev_regs.contains(&reg_num);
                    // for &r in &prev_regs {
                    //     if r == reg_num {
                    //         new_reg = false;
                    //     }
                    // }
                    if reg_num >= 0 && is_new_reg {
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

                        self.src_operands[op] = Some(Operand::new(
                            // Weak::clone(&cu),
                            self.warp_id,
                            self.id, // cu id
                            op,
                            reg_num,
                            bank,
                            scheduler_id,
                        ));
                        // panic!("setting op as not ready");
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

            let msg = format!(
                "operand collector: move input register {} to warp instruction {:?}",
                &input_reg_set,
                self.warp_instr.as_ref().map(ToString::to_string),
            );
            input_reg_set.move_out_to(&mut self.warp_instr, msg);
            true
        } else {
            false
        }
        // todo!("collector unit: allocate");
    }

    // pub fn operands(&self) {
    //     self.src_operand
    // }

    // pub fn active_mask(&self) -> Option<&sched::ThreadActiveMask> {
    //     self.warp_instr.map(|i| &i.active_mask)
    // }

    pub fn collect_operand(&mut self, op: usize) {
        log::debug!(
            "collector unit [{}] {:?} collecting operand for {}",
            self.id,
            self.warp_instr.as_ref().map(ToString::to_string),
            op,
        );
        self.not_ready.set(op, false);
    }

    // pub fn num_operands(&self) -> usize {
    //     self.warp_instr.map(|i| i.num_operands()).unwrap_or(0)
    // }
    //
    // pub fn num_regs(&self) {
    //     self.warp_instr.map(|i| i.num_regs()).unwrap_or(0)
    // }
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
    pub fn new(kind: AllocationKind, op: Option<Operand>) -> Self {
        Self { kind, op }
    }

    // pub fn write(op: Option<Operand>) -> Self {
    //     Self {
    //         kind: AllocationKind::WRITE_ALLOC,
    //         op,
    //     }
    // }
    //
    // pub fn read(op: Option<Operand>) -> Self {
    //     Self {
    //         kind: AllocationKind::READ_ALLOC,
    //         op,
    //     }
    // }
    //
    pub fn is_read(&self) -> bool {
        self.kind == AllocationKind::READ_ALLOC
    }

    pub fn is_write(&self) -> bool {
        self.kind == AllocationKind::WRITE_ALLOC
    }

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

    /// bank # -> register that wins
    allocated_banks: Box<[Allocation]>,
    queue: Box<[VecDeque<Operand>]>,
    result: Vec<Operand>,

    /// cu # -> next bank to check for request (rr-arb)
    // allocator_round_robin_head: usize,
    /// first cu to check while arb-ing banks (rr)
    last_cu: usize,
    inmatch: Box<[Option<usize>]>,
    outmatch: Box<[Option<usize>]>,
    request: Box<[Box<[Option<usize>]>]>,
}

impl Arbiter {
    pub fn init(&mut self, num_collectors: usize, num_banks: usize) {
        debug_assert!(num_collectors > 0);
        debug_assert!(num_banks > 0);
        self.num_collectors = num_collectors;
        self.num_banks = num_banks;

        self.result = Vec::new();
        self.inmatch = vec![None; self.num_banks].into_boxed_slice();
        self.outmatch = vec![None; self.num_collectors].into_boxed_slice();
        self.request = vec![vec![None; self.num_collectors].into_boxed_slice(); self.num_banks]
            .into_boxed_slice();

        self.queue = vec![VecDeque::new(); self.num_banks].into_boxed_slice();
        self.allocated_banks = vec![Allocation::default(); self.num_banks].into_boxed_slice();

        //   m_allocator_rr_head = new unsigned[num_cu];
        //   for (unsigned n = 0; n < num_cu; n++)
        //     m_allocator_rr_head[n] = n % num_banks;
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

    pub fn allocate_reads(&mut self) -> &Vec<Operand> {
        /// a list of registers that (a) are in different register banks,
        /// (b) do not go to the same operand collector
        // let mut result = Vec::new();

        log::trace!("queue: {:?}", &self.queue);

        let _inputs = self.num_banks;
        let _outputs = self.num_collectors;
        let _square = if _inputs > _outputs {
            _inputs
        } else {
            _outputs
        };
        // debug_assert!(_square > 0);
        let last_cu_before = self.last_cu;
        let mut _pri = self.last_cu;
        log::debug!("last cu: {}", self.last_cu);

        // clear matching
        // let mut inmatch = vec![None; self.num_banks];
        // let mut outmatch = vec![None; self.num_collectors];
        // let mut request = vec![vec![None; self.num_collectors]; self.num_banks];
        let result = &mut self.result;
        let inmatch = &mut self.inmatch;
        let outmatch = &mut self.outmatch;
        let request = &mut self.request;

        result.clear();
        inmatch.fill(None);
        outmatch.fill(None);

        for bank in 0..self.num_banks {
            debug_assert!(bank < _inputs);
            for collector in 0..self.num_collectors {
                debug_assert!(collector < _outputs);
                request[bank][collector] = Some(0);
            }
            if let Some(op) = self.queue[bank].front() {
                // todo: this is bad: store the cu hardware id in the operand?
                // let cu = op.collector_unit.as_ref().unwrap().upgrade().unwrap();
                // let collector_id = cu.borrow().id;
                let collector_id = op.collector_unit_id;
                debug_assert!(collector_id < _outputs);
                request[bank][collector_id] = Some(1);
            }
            if self.allocated_banks[bank].is_write() {
                inmatch[bank] = Some(0); // write gets priority
            }
            log::trace!("request: {:?}", &Self::compat(&request[bank]));
        }

        log::trace!("inmatch: {:?}", &Self::compat(&inmatch));

        // wavefront allocator from booksim
        // loop through diagonals of request matrix

        let mut output = 0;
        for p in 0.._square {
            output = (_pri + p) % _outputs;

            // step through the current diagonal
            for input in 0.._inputs {
                debug_assert!(output < _outputs);

                // banks at the same cycle
                if output < _outputs
                    && inmatch[input].is_none()
                    && request[input][output] != Some(0)
                {
                    // Grant!
                    inmatch[input] = Some(output);
                    outmatch[output] = Some(input);
                    // printf("Register File: granting bank %d to OC %d, schedid %d, warpid
                    // %d, Regid %d\n", input, output, (m_queue[input].front()).get_sid(),
                    // (m_queue[input].front()).get_wid(),
                    // (m_queue[input].front()).get_reg());
                }

                output = (output + 1) % _outputs;
            }
        }

        log::trace!("inmatch: {:?}", &Self::compat(&inmatch));
        log::trace!("outmatch: {:?}", &Self::compat(&outmatch));

        // Round-robin the priority diagonal
        _pri = (_pri + 1) % _outputs;
        log::trace!("pri: {:?}", _pri);

        // <--- end code from booksim

        self.last_cu = _pri;
        log::debug!(
            "last cu: {} -> {} ({} outputs)",
            last_cu_before,
            self.last_cu,
            _outputs
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
                        result.push(op);
                    }
                }
            }
        }

        result
    }

    pub fn add_read_requests(&mut self, cu: &CollectorUnit) {
        for src_op in &cu.src_operands {
            if let Some(src_op) = src_op {
                // if src_op.valid() {
                let bank = src_op.bank;
                self.queue[bank].push_back(src_op.clone());
            }
        }
    }

    pub fn bank_idle(&self, bank: usize) -> bool {
        self.allocated_banks[bank].is_free()
    }

    pub fn allocate_bank_for_write(&mut self, bank: usize, op: &Operand) {
        debug_assert!(bank < self.num_banks);
        self.allocated_banks[bank].allocate_for_write(Some(op.clone()));
    }

    pub fn allocate_bank_for_read(&mut self, bank: usize, op: &Operand) {
        debug_assert!(bank < self.num_banks);
        self.allocated_banks[bank].allocate_for_read(Some(op.clone()));
    }

    pub fn reset_alloction(&mut self) {
        for bank in self.allocated_banks.iter_mut() {
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
    kind: OperandCollectorUnitKind,
}

impl DispatchUnit {
    pub fn new(kind: OperandCollectorUnitKind) -> Self {
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
        collector_units: &'a Vec<Rc<RefCell<CollectorUnit>>>,
    ) -> Option<&'a Rc<RefCell<CollectorUnit>>> {
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

            if collector_units[i].borrow().ready() {
                self.last_cu = i;
                log::debug!(
                    "dispatch unit {:?}: FOUND ready: chose collector unit {} ({:?})",
                    self.kind,
                    i,
                    collector_units[i].borrow().kind
                );
                return collector_units.get(i);
            }
        }
        log::debug!("dispatch unit {:?}: did NOT find ready", self.kind);
        None
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InputPort {
    in_ports: PortVec,
    out_ports: PortVec,
    collector_unit_ids: Vec<OperandCollectorUnitKind>,
}

impl InputPort {
    pub fn new(
        in_ports: PortVec,
        out_ports: PortVec,
        collector_unit_ids: Vec<OperandCollectorUnitKind>,
    ) -> Self {
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
pub enum OperandCollectorUnitKind {
    SP_CUS,
    DP_CUS,
    SFU_CUS,
    TENSOR_CORE_CUS,
    INT_CUS,
    MEM_CUS,
    GEN_CUS,
}

pub type CuSets = HashMap<OperandCollectorUnitKind, Vec<Rc<RefCell<CollectorUnit>>>>;

// operand collector based register file unit
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperandCollectorRegisterFileUnit {
    pub config: Arc<config::GPUConfig>,

    pub initialized: bool,
    pub num_banks: usize,
    pub num_collectors: usize,

    pub bank_warp_shift: usize,
    pub sub_core_model: bool,
    pub num_banks_per_scheduler: usize,
    pub num_warp_schedulers: usize,

    pub arbiter: Arbiter,
    pub in_ports: VecDeque<InputPort>,
    pub collector_units: Vec<Rc<RefCell<CollectorUnit>>>,
    pub collector_unit_sets: CuSets,
    pub dispatch_units: Vec<DispatchUnit>,
    // dispatch_units: Vec<Rc<RefCell<DispatchUnit>>>,
    // dispatch_units: VecDeque<DispatchUnit>,
    // allocation_t *m_allocated_bank;  // bank # -> register that wins
    // std::list<op_t> *m_queue;
    //
    // unsigned *
    //     m_allocator_rr_head;  // cu # -> next bank to check for request (rr-arb)
    // unsigned m_last_cu;       // first cu to check while arb-ing banks (rr)
    //
    // int *_inmatch;
    // int *_outmatch;
    // int **_request;
}

pub type PortVec = Vec<Rc<RefCell<RegisterSet>>>;

impl OperandCollectorRegisterFileUnit {
    pub fn new(config: Arc<config::GPUConfig>) -> Self {
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
        self.arbiter.init(num_collector_units, num_banks);
        self.num_banks = num_banks;
        self.bank_warp_shift = (self.config.warp_size as f32 + 0.5).log2() as usize;
        // (unsigned)(int)(std::log(m_warp_size + 0.5) / std::log(2.0));
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

        let mut reg_id = 0;
        for (cu_id, cu) in self.collector_units.iter().enumerate() {
            if self.sub_core_model {
                let coll_units_per_scheduler = num_collector_units / self.num_warp_schedulers;
                reg_id = cu_id / coll_units_per_scheduler;
            }
            let mut cu = cu.try_borrow_mut().unwrap();
            cu.init(
                cu_id,
                self.num_banks,
                self.bank_warp_shift,
                self.config.clone(),
                self.sub_core_model,
                reg_id,
                self.num_banks_per_scheduler,
            );

            debug_assert!(cu.id == cu_id);
        }
        for dispatch_unit in self.dispatch_units.iter_mut() {
            dispatch_unit.init(self.sub_core_model, self.num_warp_schedulers);
        }
        self.initialized = true;
    }

    // /// NOTE: using an iterator does not work because hash map ordering is non-deterministic
    // pub fn collector_units(&mut self) -> impl Iterator<Item = &mut CollectorUnit> {
    //     self.collector_unit_sets
    //         .values_mut()
    //         .flat_map(|units| units)
    // }

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
        // todo!("operand collector: process banks");
        self.arbiter.reset_alloction();
    }

    /// Process read requests that do not have conflicts
    pub fn allocate_reads(&mut self) {
        // process read requests that do not have conflicts
        let allocated: Vec<Operand> = self.arbiter.allocate_reads().clone();
        // TODO: move this into the arbiter??
        log::debug!(
            "arbiter allocated {} reads ({:?})",
            allocated.len(),
            &allocated
        );
        let mut read_ops = HashMap::new();
        for read in &allocated {
            let reg = read.register;
            let warp_id = read.warp_id().unwrap();
            let bank = register_bank(
                reg,
                warp_id,
                self.num_banks,
                self.bank_warp_shift,
                self.sub_core_model,
                self.num_banks_per_scheduler,
                read.scheduler_id,
            );
            self.arbiter.allocate_bank_for_read(bank, read);
            read_ops.insert(bank, read);
        }

        log::debug!("allocating {} reads ({:?})", read_ops.len(), &read_ops);
        for (bank, read) in &read_ops {
            // todo: use the cu id here
            // todo!("use cu id here for collecting operand");

            // debug_assert!(read.collector_unit_id < self.collector_units.len());
            let mut cu = self.collector_units[read.collector_unit_id].borrow_mut();
            if let Some(operand) = read.operand {
                cu.collect_operand(operand);
            }

            // let cu = read.collector_unit();
            // let mut cu = cu.as_ref().unwrap().borrow_mut();
            // if let Some(operand) = read.operand {
            //     cu.collect_operand(operand);
            // }

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
            // todo!("operand coll unit: allocate reads");
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
            if input_port.borrow().has_ready() {
                // find a free collector unit
                for cu_set_id in &port.collector_unit_ids {
                    let cu_set: &Vec<_> = &self.collector_unit_sets[cu_set_id];
                    let mut allocated = false;
                    let mut cu_lower_bound = 0;
                    let mut cu_upper_bound = cu_set.len();

                    if self.sub_core_model {
                        // sub core model only allocates on the subset of CUs assigned
                        // to the scheduler that issued
                        let (reg_id, _) = input_port.borrow().get_ready().unwrap();
                        debug_assert!(
                            cu_set.len() % self.num_warp_schedulers == 0
                                && cu_set.len() >= self.num_warp_schedulers
                        );
                        let cus_per_sched = cu_set.len() / self.num_warp_schedulers;
                        let schd_id = input_port.borrow().scheduler_id(reg_id).unwrap();
                        cu_lower_bound = schd_id * cus_per_sched;
                        cu_upper_bound = cu_lower_bound + cus_per_sched;
                        debug_assert!(0 <= cu_lower_bound && cu_upper_bound <= cu_set.len());
                    }

                    for k in cu_lower_bound..cu_upper_bound {
                        // let cu = &cu_set[k];
                        // let cu_backref = Rc::downgrade(&cu_set[k]);
                        // let mut cu = cu_set[k].try_borrow_mut().unwrap();
                        let mut collector_unit = cu_set[k].try_borrow_mut().unwrap();

                        if collector_unit.free {
                            log::debug!(
                                "{} cu={:?}",
                                style(format!("operand collector::allocate()")).green(),
                                collector_unit.kind
                            );

                            allocated = collector_unit.allocate(&input_port, &output_port);
                            // allocated = collector_unit.allocate(cu_backref, &input_port, &output_port);
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
        // todo!("operand collector: allocate cu");
    }

    pub fn dispatch_ready_cu(&mut self) {
        // todo!("operand collector: dispatch ready cu");

        for dispatch_unit in &mut self.dispatch_units {
            // for (unsigned p = 0; p < m_dispatch_units.size(); ++p) {
            // dispatch_unit_t &du = m_dispatch_units[p];

            let set = &self.collector_unit_sets[&dispatch_unit.kind];
            if let Some(collector_unit) = dispatch_unit.find_ready(set) {
                // if (cu) {
                // for (unsigned i = 0; i < (cu->get_num_operands() - cu->get_num_regs());
                //      i++) {
                //   if (m_shader->get_config()->gpgpu_clock_gated_reg_file) {
                //     unsigned active_count = 0;
                //     for (unsigned i = 0; i < m_shader->get_config()->warp_size;
                //          i = i + m_shader->get_config()->n_regfile_gating_group) {
                //       for (unsigned j = 0;
                //            j < m_shader->get_config()->n_regfile_gating_group; j++) {
                //         if (cu->get_active_mask().test(i + j)) {
                //           active_count += m_shader->get_config()->n_regfile_gating_group;
                //           break;
                //         }
                //       }
                //     }
                //     m_shader->incnon_rf_operands(active_count);
                //   } else {
                //     m_shader->incnon_rf_operands(
                //         m_shader->get_config()->warp_size);  // cu->get_active_count());
                //   }
                // }
                collector_unit.borrow_mut().dispatch();
            }
        }
    }

    pub fn writeback(&mut self, instr: &mut WarpInstruction) -> bool {
        // std::list<unsigned> regs = m_shader->get_regs_written(inst);
        let regs: Vec<u32> = instr.dest_arch_reg.iter().filter_map(|reg| *reg).collect();
        log::trace!(
            "operand collector: writeback {} with destination registers {:?}",
            instr,
            regs
        );

        //   std::list<unsigned> result;
        // for (unsigned op = 0; op < MAX_REG_OPERANDS; op++) {
        //   int reg_num = fvt.arch_reg.dst[op];  // this math needs to match that used
        //                                        // in function_info::ptx_decode_inst
        //   if (reg_num >= 0)                    // valid register
        //     result.push_back(reg_num);
        // }
        // return result;

        // instr.dest_arch_reg[m] == instr.outputs[m](private)

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
                        &Operand {
                            // warp_instr: Some(instr.clone()),
                            warp_id: Some(instr.warp_id),
                            register: reg_num,
                            scheduler_id,
                            operand: None,
                            bank,
                            collector_unit_id: 0, // TODO: that fine?
                                                  // collector_unit: None,
                        }, // op_t(&inst, reg_num, m_num_banks, m_bank_warp_shift, sub_core_model,
                           //      m_num_banks_per_sched, inst.get_schd_id()));
                    );
                    instr.dest_arch_reg[op] = None;
                } else {
                    return false;
                }
            }
        }

        // for (unsigned op = 0; op < MAX_REG_OPERANDS; op++) {
        //   int reg_num = inst.arch_reg.dst[op];  // this math needs to match that used
        //                                         // in function_info::ptx_decode_inst
        //   if (reg_num >= 0) {                   // valid register
        //     unsigned bank = register_bank(reg_num, inst.warp_id(), m_num_banks,
        //                                   m_bank_warp_shift, sub_core_model,
        //                                   m_num_banks_per_sched, inst.get_schd_id());
        //     if (m_arbiter.bank_idle(bank)) {
        //       m_arbiter.allocate_bank_for_write(
        //           bank,
        //           op_t(&inst, reg_num, m_num_banks, m_bank_warp_shift, sub_core_model,
        //                m_num_banks_per_sched, inst.get_schd_id()));
        //       inst.arch_reg.dst[op] = -1;
        //     } else {
        //       return false;
        //     }
        //   }
        // }
        // for (unsigned i = 0; i < (unsigned)regs.size(); i++) {
        //   if (m_shader->get_config()->gpgpu_clock_gated_reg_file) {
        //     unsigned active_count = 0;
        //     for (unsigned i = 0; i < m_shader->get_config()->warp_size;
        //          i = i + m_shader->get_config()->n_regfile_gating_group) {
        //       for (unsigned j = 0; j < m_shader->get_config()->n_regfile_gating_group;
        //            j++) {
        //         if (inst.get_active_mask().test(i + j)) {
        //           active_count += m_shader->get_config()->n_regfile_gating_group;
        //           break;
        //         }
        //       }
        //     }
        //     m_shader->incregfile_writes(active_count);
        //   } else {
        //     m_shader->incregfile_writes(
        //         m_shader->get_config()->warp_size);  // inst.active_count());
        //   }
        // }
        true
    }

    pub fn add_cu_set(
        &mut self,
        kind: OperandCollectorUnitKind,
        num_collector_units: usize,
        num_dispatch_units: usize,
    ) {
        // this is necessary to stop pointers in m_cu from being invalid
        // to do a resize;
        // let set = self.collector_unit_sets.get_mut(&set_id).unwrap();
        // set.reserve_exact(num_cu);

        let set = self.collector_unit_sets.entry(kind).or_default();

        for coll_unit_id in 0..num_collector_units {
            let unit = Rc::new(RefCell::new(CollectorUnit::new(kind)));
            set.push(Rc::clone(&unit));
            self.collector_units.push(unit);
        }
        // for now each collector set gets dedicated dispatch units.
        for dispatch_unit_id in 0..num_dispatch_units {
            let dispatch_unit = DispatchUnit::new(kind);
            self.dispatch_units.push(dispatch_unit);
        }
        // todo!("operand collector: add cu set");
    }

    pub fn add_port(
        &mut self,
        input: PortVec,
        output: PortVec,
        cu_sets: Vec<OperandCollectorUnitKind>,
    ) {
        self.in_ports
            .push_back(InputPort::new(input, output, cu_sets));
    }
}

#[cfg(test)]
mod test {
    use crate::ported::mem_fetch::BitString;
    use crate::ported::testing;
    use std::ops::Deref;

    // #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    // pub enum {
    //     SP_CUS,
    //     DP_CUS,
    //     SFU_CUS,
    //     TENSOR_CORE_CUS,
    //     INT_CUS,
    //     MEM_CUS,
    //     GEN_CUS,
    // }

    impl From<super::OperandCollectorUnitKind> for testing::state::OperandCollectorUnitKind {
        fn from(id: super::OperandCollectorUnitKind) -> Self {
            match id {
                super::OperandCollectorUnitKind::SP_CUS => {
                    testing::state::OperandCollectorUnitKind::SP_CUS
                }
                super::OperandCollectorUnitKind::DP_CUS => {
                    testing::state::OperandCollectorUnitKind::DP_CUS
                }
                super::OperandCollectorUnitKind::SFU_CUS => {
                    testing::state::OperandCollectorUnitKind::SFU_CUS
                }
                super::OperandCollectorUnitKind::TENSOR_CORE_CUS => {
                    testing::state::OperandCollectorUnitKind::TENSOR_CORE_CUS
                }
                super::OperandCollectorUnitKind::INT_CUS => {
                    testing::state::OperandCollectorUnitKind::INT_CUS
                }
                super::OperandCollectorUnitKind::MEM_CUS => {
                    testing::state::OperandCollectorUnitKind::MEM_CUS
                }
                super::OperandCollectorUnitKind::GEN_CUS => {
                    testing::state::OperandCollectorUnitKind::GEN_CUS
                }
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
                    .map(|p| p.borrow().clone().into())
                    .collect(),
                out_ports: port
                    .out_ports
                    .iter()
                    .map(|p| p.borrow().clone().into())
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
                    .map(|r| r.borrow().deref().clone().into()),
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

    impl From<&super::OperandCollectorRegisterFileUnit> for testing::state::OperandCollector {
        fn from(opcoll: &super::OperandCollectorRegisterFileUnit) -> Self {
            let dispatch_units = opcoll.dispatch_units.iter().map(Into::into).collect();
            let collector_units = opcoll
                .collector_units
                .iter()
                .map(|cu| cu.borrow().deref().into())
                .collect();
            let ports = opcoll.in_ports.iter().map(Into::into).collect();
            let arbiter = (&opcoll.arbiter).into();
            Self {
                ports,
                dispatch_units,
                collector_units,
                arbiter,
            }
        }
    }
}
