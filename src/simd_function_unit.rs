use super::{instruction::WarpInstruction, register_set, warp};
use crate::config;
use bitvec::{array::BitArray, BitArr};
use std::sync::Arc;

pub trait SimdFunctionUnit: Send + Sync + std::fmt::Display + 'static {
    fn id(&self) -> &str;
    fn cycle(&mut self);
    fn issue(&mut self, source_reg: WarpInstruction);

    // accessors
    fn clock_multiplier(&self) -> usize {
        1
    }
    fn can_issue(&self, instr: &WarpInstruction) -> bool;
    fn pipeline(&self) -> &Vec<Option<WarpInstruction>>;
    fn active_lanes_in_pipeline(&self) -> usize;
    fn is_issue_partitioned(&self) -> bool;
    fn issue_reg_id(&self) -> usize;
    fn stallable(&self) -> bool;
}

pub const MAX_ALU_LATENCY: usize = 512;

#[derive()]
pub struct PipelinedSimdUnitImpl {
    pub cycle: super::Cycle,
    pub result_port: Option<register_set::Ref>,
    pub id: usize,
    pub name: String,
    pub pipeline_depth: usize,
    pub pipeline_reg: Vec<Option<WarpInstruction>>,
    pub issue_reg_id: usize,
    pub active_insts_in_pipeline: usize,
    pub dispatch_reg: Option<WarpInstruction>,
    pub occupied: BitArr!(for MAX_ALU_LATENCY),
    pub config: Arc<config::GPUConfig>,
}

impl std::fmt::Display for PipelinedSimdUnitImpl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PipelinedSimdUnitImpl")
    }
}

impl std::fmt::Debug for PipelinedSimdUnitImpl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelinedSimdUnitImpl").finish()
    }
}

impl PipelinedSimdUnitImpl {
    pub fn new(
        id: usize,
        name: String,
        // result_port: Option<Rc<RefCell<RegisterSet>>>,
        result_port: Option<register_set::Ref>,
        depth: usize,
        config: Arc<config::GPUConfig>,
        cycle: super::Cycle,
        issue_reg_id: usize,
    ) -> Self {
        let pipeline_reg = (0..depth).map(|_| None).collect();
        Self {
            cycle,
            id,
            name,
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

    pub fn num_active_instr_in_pipeline(&self) -> usize {
        self.pipeline_reg
            .iter()
            .map(Option::as_ref)
            .filter(Option::is_some)
            .count()
    }
}

impl SimdFunctionUnit for PipelinedSimdUnitImpl {
    fn active_lanes_in_pipeline(&self) -> usize {
        let mut active_lanes: warp::ActiveMask = BitArray::ZERO;
        for stage in self.pipeline_reg.iter().flatten() {
            active_lanes |= stage.active_mask;
        }
        active_lanes.count_ones()
    }

    fn id(&self) -> &str {
        &self.name
    }

    fn pipeline(&self) -> &Vec<Option<WarpInstruction>> {
        &self.pipeline_reg
    }

    fn cycle(&mut self) {
        log::debug!(
            "fu[{:03}] {:<10} cycle={:03}: \tpipeline={:?} ({}/{} active)",
            self.id,
            self.name,
            self.cycle.get(),
            self.pipeline_reg
                .iter()
                .map(|reg| reg.as_ref().map(std::string::ToString::to_string))
                .collect::<Vec<_>>(),
            self.num_active_instr_in_pipeline(),
            self.pipeline_reg.len(),
        );

        if let Some(result_port) = &mut self.result_port {
            if let Some(pipe_reg) = self.pipeline_reg[0].take() {
                // move to EX_WB result port
                // let mut result_port = result_port.borrow_mut();
                let mut result_port = result_port.try_lock().unwrap();
                // let msg = format!(
                //     "{}: move pipeline[0] to result port {:?}",
                //     self.name, result_port.stage
                // );
                result_port.move_in_from(Some(pipe_reg));

                debug_assert!(self.active_insts_in_pipeline > 0);
                self.active_insts_in_pipeline -= 1;
            }
        }
        debug_assert_eq!(
            self.num_active_instr_in_pipeline(),
            self.active_insts_in_pipeline
        );
        if self.active_insts_in_pipeline > 0 {
            for stage in 0..(self.pipeline_reg.len() - 1) {
                let current = self.pipeline_reg[stage + 1].take();
                let next = &mut self.pipeline_reg[stage];
                // let msg = format!("{} moving to next slot in pipeline register", self.name);
                register_set::move_warp(current, next);
            }
        }
        if let Some(dispatch) = self.dispatch_reg.take() {
            // if !dispatch.empty() && !dispatch.dispatch_delay() {
            let start_stage_idx = dispatch.latency - dispatch.initiation_interval;
            if self.pipeline_reg[start_stage_idx].is_none() {
                register_set::move_warp(
                    Some(dispatch),
                    &mut self.pipeline_reg[start_stage_idx],
                    // format!(
                    //     "{} moving dispatch register to free pipeline_register[start_stage={}]",
                    //     self.name, start_stage_idx
                    // ),
                );
                self.active_insts_in_pipeline += 1;
            }
        }

        // occupied latencies are shifted each cycle
        self.occupied.shift_right(1);
    }

    fn issue(&mut self, src_reg: WarpInstruction) {
        register_set::move_warp(
            Some(src_reg),
            &mut self.dispatch_reg,
            // format!(
            //     "{} moving register to dispatch register for issue",
            //     self.name,
            // ),
        );
    }

    fn clock_multiplier(&self) -> usize {
        1
    }

    fn can_issue(&self, instr: &WarpInstruction) -> bool {
        self.dispatch_reg.is_none() && !self.occupied[instr.latency]
    }

    fn is_issue_partitioned(&self) -> bool {
        todo!("pipelined simd unit: is issue partitioned");
    }

    fn issue_reg_id(&self) -> usize {
        todo!("pipelined simd unit: issue reg id");
    }

    fn stallable(&self) -> bool {
        false
    }
}
