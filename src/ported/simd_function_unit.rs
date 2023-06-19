use super::{
    instruction::WarpInstruction,
    register_set::{self, RegisterSet},
    scheduler as sched,
};
use crate::config;
use bitvec::{array::BitArray, BitArr};
use std::sync::Arc;

pub trait SimdFunctionUnit: std::fmt::Debug {
    // modifiers
    fn cycle(&mut self);
    fn issue(&mut self, source_reg: &mut RegisterSet);
    // fn compute_active_lanes_in_pipeline(&mut self);

    // accessors
    fn clock_multiplier(&self) -> usize {
        1
    }
    fn can_issue(&self, instr: &WarpInstruction) -> bool;
    fn active_lanes_in_pipeline(&self) -> usize;
    // {
    //     return m_dispatch_reg->empty() && !occupied.test(inst.latency);
    //   }
    fn is_issue_partitioned(&self) -> bool;
    fn issue_reg_id(&self) -> usize;
    fn stallable(&self) -> bool;

    // void simd_function_unit::issue(register_set &source_reg) {
    //   bool partition_issue =
    //       m_config->sub_core_model && this->is_issue_partitioned();
    //   source_reg.move_out_to(partition_issue, this->get_issue_reg_id(),
    //                          m_dispatch_reg);
    //   occupied.set(m_dispatch_reg->latency);
    // }
}

pub const MAX_ALU_LATENCY: usize = 512;

#[derive()]
pub struct PipelinedSimdUnitImpl {
    pub result_port: Option<RegisterSet>,
    pub pipeline_depth: usize,
    // pub pipeline_reg: Vec<WarpInstruction>,
    pub pipeline_reg: Vec<Option<WarpInstruction>>,
    pub issue_reg_id: usize,
    pub active_insts_in_pipeline: usize,
    pub dispatch_reg: Option<WarpInstruction>,
    pub occupied: BitArr!(for MAX_ALU_LATENCY),
    pub config: Arc<config::GPUConfig>,
}

impl std::fmt::Debug for PipelinedSimdUnitImpl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelinedSimdUnitImpl").finish()
    }
}

impl PipelinedSimdUnitImpl {
    pub fn new(
        result_port: Option<RegisterSet>,
        depth: usize,
        config: Arc<config::GPUConfig>,
        issue_reg_id: usize,
    ) -> Self {
        let pipeline_reg = (0..depth)
            // .map(|_| WarpInstruction::new_empty(&*config))
            .map(|_| None)
            .collect();
        Self {
            result_port,
            pipeline_depth: depth,
            pipeline_reg,
            issue_reg_id,
            active_insts_in_pipeline: 0,
            dispatch_reg: None,
            occupied: BitArray::ZERO,
            config,
        }
    }
}

impl SimdFunctionUnit for PipelinedSimdUnitImpl {
    fn active_lanes_in_pipeline(&self) -> usize {
        let mut active_lanes: sched::ThreadActiveMask = BitArray::ZERO;
        // if self.config.
        for stage in &self.pipeline_reg {
            if let Some(stage) = stage {
                active_lanes |= stage.active_mask;
            }
        }
        // for (unsigned stage = 0; (stage + 1) < m_pipeline_depth; stage++) {
        //   if (!m_pipeline_reg[stage]->empty())
        //     active_lanes |= m_pipeline_reg[stage]->get_active_mask();
        // }
        active_lanes.count_ones()
    }

    fn cycle(&mut self) {
        // if !self.pipeline_reg[0].empty() {
        if let Some(port) = &mut self.result_port {
            if let Some(stage) = self.pipeline_reg[0].take() {
                port.move_in_from(Some(stage));
            }
            debug_assert!(self.active_insts_in_pipeline > 0);
            self.active_insts_in_pipeline -= 1;
        }
        if self.active_insts_in_pipeline > 0 {
            for stage in 0..self.pipeline_reg.len() - 1 {
                let current = self.pipeline_reg[stage].take();
                let next = &mut self.pipeline_reg[stage + 1];
                register_set::move_warp(current, next);
            }
        }
        if let Some(dispatch) = &self.dispatch_reg {
            // if !dispatch.empty() && !dispatch.dispatch_delay() {
            if !dispatch.empty() {
                // let start_stage = dispatch.latency - dispatch.initiation_interval;
                // move_warp(m_pipeline_reg[start_stage], m_dispatch_reg);
                self.active_insts_in_pipeline += 1;
            }
        }
        // occupied latencies are shifted each cycle
        self.occupied.shift_right(1);

        // todo!("pipelined simd unit: cycle");
    }

    fn issue(&mut self, src_reg: &mut RegisterSet) {
        let partition_issue = self.config.sub_core_model && self.is_issue_partitioned();
        // let ready_reg = src_reg.get_ready(partition_issue, self.issue_reg_id());
        // // self.core.incexecstat((*ready_reg));
        //
        // // from simd function unit
        if partition_issue {
            src_reg.move_out_to_sub_core(self.issue_reg_id(), &mut self.dispatch_reg);
        } else {
            src_reg.move_out_to(&mut self.dispatch_reg);
        }

        let dispatched = self.dispatch_reg.as_ref().unwrap();
        self.occupied.set(dispatched.latency, true);
        // if let Some(dispatch) = &self.dispatch_reg {
        //     self.occupied.set(dispatch.latency, true);
        // }
        // todo!("pipelined simd unit: issue");
    }

    fn clock_multiplier(&self) -> usize {
        1
    }

    fn can_issue(&self, instr: &WarpInstruction) -> bool {
        self.dispatch_reg.is_none() && !self.occupied[instr.latency]
        // todo!("pipelined simd unit: can issue");
    }

    // fn active_lanes_in_pipeline(&self) -> usize;
    // {
    //     return m_dispatch_reg->empty() && !occupied.test(inst.latency);
    //   }

    fn is_issue_partitioned(&self) -> bool {
        todo!("pipelined simd unit: is issue partitioned");
    }

    fn issue_reg_id(&self) -> usize {
        todo!("pipelined simd unit: issue reg id");
    }

    fn stallable(&self) -> bool {
        false
        // todo!("pipelined simd unit: stallable");
    }
}
