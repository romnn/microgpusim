use super::{instruction::WarpInstruction, register_set::RegisterSet};
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Default)]
pub struct CollectorUnit {}

#[derive(Debug, Default)]
pub struct DispatchUnit {}

#[derive(Debug, Default)]
pub struct InputPort {
    // port_vector_t m_in, m_out;
    // uint_vector_t m_cu_sets;
}

impl InputPort {
    pub fn new(input: PortVec, output: PortVec, cu_sets: Vec<u32>) -> Self {
        debug_assert!(input.len() == output.len());
        debug_assert!(!cu_sets.is_empty());
        Self {}
    }
}

pub type CuSets = HashMap<usize, Vec<CollectorUnit>>;

// operand collector based register file unit
#[derive(Debug)]
pub struct OperandCollectorRegisterFileUnit {
    num_banks: usize,
    num_collectors: usize,

    in_ports: VecDeque<InputPort>,
    cus: CuSets,
    dispatch_units: VecDeque<DispatchUnit>,
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

pub type PortVec = Vec<RegisterSet>;
// pub type uint_vector = usize;

impl OperandCollectorRegisterFileUnit {
    pub fn new(num_banks: usize) -> Self {
        Self {
            num_banks,
            num_collectors: 0,
            in_ports: VecDeque::new(),
            cus: CuSets::new(),
            dispatch_units: VecDeque::new(),
        }
    }

    pub fn step(&mut self) {
        self.dispatch_ready_cu();
        self.allocate_reads();
        for port_num in 0..self.in_ports.len() {
            self.allocate_cu(port_num);
        }
        self.process_banks();
    }

    fn process_banks(&mut self) {
        todo!("operand collector: process banks");
        // self.arbiter.reset_alloction();
    }

    /// Process read requests that do not have conflicts
    pub fn allocate_reads(&mut self) {
        todo!("operand collector: allocate reads");
        // std::list<op_t> allocated = m_arbiter.allocate_reads();
        // std::map<unsigned, op_t> read_ops;
        // for (std::list<op_t>::iterator r = allocated.begin(); r != allocated.end();
        //      r++) {
        //   const op_t &rr = *r;
        //   unsigned reg = rr.get_reg();
        //   unsigned wid = rr.get_wid();
        //   unsigned bank =
        //       register_bank(reg, wid, m_num_banks, m_bank_warp_shift, sub_core_model,
        //                     m_num_banks_per_sched, rr.get_sid());
        //   m_arbiter.allocate_for_read(bank, rr);
        //   read_ops[bank] = rr;
        // }
        // std::map<unsigned, op_t>::iterator r;
        // for (r = read_ops.begin(); r != read_ops.end(); ++r) {
        //   op_t &op = r->second;
        //   unsigned cu = op.get_oc_id();
        //   unsigned operand = op.get_operand();
        //   m_cu[cu]->collect_operand(operand);
        //   if (m_shader->get_config()->gpgpu_clock_gated_reg_file) {
        //     unsigned active_count = 0;
        //     for (unsigned i = 0; i < m_shader->get_config()->warp_size;
        //          i = i + m_shader->get_config()->n_regfile_gating_group) {
        //       for (unsigned j = 0; j < m_shader->get_config()->n_regfile_gating_group;
        //            j++) {
        //         if (op.get_active_mask().test(i + j)) {
        //           active_count += m_shader->get_config()->n_regfile_gating_group;
        //           break;
        //         }
        //       }
        //     }
        //     m_shader->incregfile_reads(active_count);
        //   } else {
        //     m_shader->incregfile_reads(
        //         m_shader->get_config()->warp_size);  // op.get_active_count());
        //   }
        // }
    }

    pub fn allocate_cu(&mut self, port_num: usize) {
        todo!("operand collector: allocate cu");
        // input_port_t &inp = m_in_ports[port_num];
        // for (unsigned i = 0; i < inp.m_in.size(); i++) {
        //   if ((*inp.m_in[i]).has_ready()) {
        //     // find a free cu
        //     for (unsigned j = 0; j < inp.m_cu_sets.size(); j++) {
        //       std::vector<collector_unit_t> &cu_set = m_cus[inp.m_cu_sets[j]];
        //       bool allocated = false;
        //       unsigned cuLowerBound = 0;
        //       unsigned cuUpperBound = cu_set.size();
        //       unsigned schd_id;
        //       if (sub_core_model) {
        //         // Sub core model only allocates on the subset of CUs assigned to the
        //         // scheduler that issued
        //         unsigned reg_id = (*inp.m_in[i]).get_ready_reg_id();
        //         schd_id = (*inp.m_in[i]).get_schd_id(reg_id);
        //         assert(cu_set.size() % m_num_warp_scheds == 0 &&
        //                cu_set.size() >= m_num_warp_scheds);
        //         unsigned cusPerSched = cu_set.size() / m_num_warp_scheds;
        //         cuLowerBound = schd_id * cusPerSched;
        //         cuUpperBound = cuLowerBound + cusPerSched;
        //         assert(0 <= cuLowerBound && cuUpperBound <= cu_set.size());
        //       }
        //       for (unsigned k = cuLowerBound; k < cuUpperBound; k++) {
        //         if (cu_set[k].is_free()) {
        //           collector_unit_t *cu = &cu_set[k];
        //           allocated = cu->allocate(inp.m_in[i], inp.m_out[i]);
        //           m_arbiter.add_read_requests(cu);
        //           break;
        //         }
        //       }
        //       if (allocated) break;  // cu has been allocated, no need to search more.
        //     }
        //     // break;  // can only service a single input, if it failed it will fail
        //     // for
        //     // others.
        //   }
        // }
    }

    pub fn dispatch_ready_cu(&mut self) {
        todo!("operand collector: dispatch ready cu");

        //   for (unsigned p = 0; p < m_dispatch_units.size(); ++p) {
        //   dispatch_unit_t &du = m_dispatch_units[p];
        //   collector_unit_t *cu = du.find_ready();
        //   if (cu) {
        //     for (unsigned i = 0; i < (cu->get_num_operands() - cu->get_num_regs());
        //          i++) {
        //       if (m_shader->get_config()->gpgpu_clock_gated_reg_file) {
        //         unsigned active_count = 0;
        //         for (unsigned i = 0; i < m_shader->get_config()->warp_size;
        //              i = i + m_shader->get_config()->n_regfile_gating_group) {
        //           for (unsigned j = 0;
        //                j < m_shader->get_config()->n_regfile_gating_group; j++) {
        //             if (cu->get_active_mask().test(i + j)) {
        //               active_count += m_shader->get_config()->n_regfile_gating_group;
        //               break;
        //             }
        //           }
        //         }
        //         m_shader->incnon_rf_operands(active_count);
        //       } else {
        //         m_shader->incnon_rf_operands(
        //             m_shader->get_config()->warp_size);  // cu->get_active_count());
        //       }
        //     }
        //     cu->dispatch();
        //   }
        // }
    }

    pub fn writeback(&mut self, instr: &WarpInstruction) -> bool {
        debug_assert!(!instr.empty());
        todo!("operand collector: writeback");
        // std::list<unsigned> regs = m_shader->get_regs_written(inst);
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

    pub fn add_cu_set(&mut self, set_id: usize, num_cu: usize, num_dispatch: usize) {
        // this is necessary to stop pointers in m_cu from being invalid
        // to do a resize;
        let set = self.cus.get_mut(&set_id).unwrap();
        set.reserve_exact(num_cu);

        // for (unsigned i = 0; i < num_cu; i++) {
        //   m_cus[set_id].push_back(collector_unit_t());
        //   m_cu.push_back(&m_cus[set_id].back());
        // }
        // // for now each collector set gets dedicated dispatch units.
        // for (unsigned i = 0; i < num_dispatch; i++) {
        //   m_dispatch_units.push_back(dispatch_unit_t(&m_cus[set_id]));
        // }
        todo!("operand collector: add cu set");
    }

    pub fn add_port(&mut self, input: PortVec, output: PortVec, cu_sets: Vec<u32>) {
        self.in_ports
            .push_back(InputPort::new(input, output, cu_sets));
        todo!("operand collector: add port");
    }
}
