use super::{
    instruction::WarpInstruction, interconn as ic, opcodes, register_set::RegisterSet,
    simd_function_unit as fu,
};
use crate::config::GPUConfig;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

#[derive()]
// pub struct SPUnit<I> {
pub struct SPUnit {
    // core_id: usize,
    // cluster_id: usize,
    config: Arc<GPUConfig>,
    pipelined_simd_unit: fu::PipelinedSimdUnitImpl,
}

// impl<I> SPUnit<I> {
impl SPUnit {
    pub fn new(
        id: usize,
        result_port: Rc<RefCell<RegisterSet>>,
        config: Arc<GPUConfig>,
        stats: Arc<Mutex<stats::Stats>>,
        cycle: super::Cycle,
        issue_reg_id: usize,
    ) -> Self {
        let pipeline_depth = config.shared_memory_latency;
        let pipelined_simd_unit = fu::PipelinedSimdUnitImpl::new(
            id,
            "SPUnit".to_string(),
            Some(result_port),
            pipeline_depth,
            config.clone(),
            cycle,
            issue_reg_id,
        );

        Self {
            config,
            pipelined_simd_unit,
        }
    }
}

impl std::fmt::Display for SPUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SPUnit")
    }
}

impl std::fmt::Debug for SPUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SPUnit").finish()
    }
}

// impl<I> fu::SimdFunctionUnit for SPUnit<I>
impl fu::SimdFunctionUnit for SPUnit
// where
//     I: ic::Interconnect<super::core::Packet>,
{
    fn can_issue(&self, instr: &WarpInstruction) -> bool {
        match instr.opcode.category {
            opcodes::ArchOp::SFU_OP => false,
            opcodes::ArchOp::LOAD_OP => false,
            opcodes::ArchOp::TENSOR_CORE_LOAD_OP => false,
            opcodes::ArchOp::STORE_OP => false,
            opcodes::ArchOp::TENSOR_CORE_STORE_OP => false,
            opcodes::ArchOp::MEMORY_BARRIER_OP => false,
            opcodes::ArchOp::DP_OP => false,
            _ => self.pipelined_simd_unit.can_issue(instr),
        }
        // todo!("load store unit: can issue");
    }

    fn is_issue_partitioned(&self) -> bool {
        true
    }

    fn active_lanes_in_pipeline(&self) -> usize {
        let active = self.pipelined_simd_unit.active_lanes_in_pipeline();
        debug_assert!(active <= self.config.warp_size);
        // m_core->incspactivelanes_stat(active_count);
        // m_core->incfuactivelanes_stat(active_count);
        // m_core->incfumemactivelanes_stat(active_count);
        active
    }

    // fn issue(&mut self, source_reg: &mut RegisterSet) {
    fn issue(&mut self, source_reg: WarpInstruction) {
        // let ready_reg = source_reg.get_ready(self.config.sub_core_model, self.issue_reg_id);
        // m_core->incexecstat((*ready_reg));
        // ready_reg.op_pipe = SP__OP;
        // m_core->incsp_stat(m_core->get_config()->warp_size, (*ready_reg)->latency);

        self.pipelined_simd_unit.issue(source_reg);
        // todo!("sp unit: issue");
    }

    fn cycle(&mut self) {
        self.pipelined_simd_unit.cycle();
    }

    fn issue_reg_id(&self) -> usize {
        self.pipelined_simd_unit.issue_reg_id()
    }

    fn stallable(&self) -> bool {
        self.pipelined_simd_unit.stallable()
    }

    fn clock_multiplier(&self) -> usize {
        self.pipelined_simd_unit.clock_multiplier()
    }
}
