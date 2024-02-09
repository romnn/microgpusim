pub mod dp;
pub mod int;
pub mod load_store;
pub mod sfu;
pub mod sp;

pub use dp::DPUnit;
pub use int::IntUnit;
pub use load_store::LoadStoreUnit;
pub use sfu::SFU;
pub use sp::SPUnit;

use crate::{
    config, core::PipelineStage, instruction::WarpInstruction, interconn as ic, mem_fetch,
    register_set, scoreboard, warp,
};
use bitvec::{array::BitArray, BitArr};
use register_set::Access;
use std::sync::Arc;

pub const MAX_ALU_LATENCY: usize = 512;
pub type OccupiedSlots = BitArr!(for MAX_ALU_LATENCY);

// crate::engine::cycle::Component + Send + Sync + std::fmt::Display + 'static
pub trait SimdFunctionUnit: Send + Sync + std::fmt::Display + 'static {
    fn id(&self) -> &str;
    fn issue_port(&self) -> PipelineStage;
    fn result_port(&self) -> Option<PipelineStage>;

    fn issue(&mut self, instr: WarpInstruction, stats: &mut stats::PerKernel);

    // accessors
    fn clock_multiplier(&self) -> usize {
        1
    }
    fn can_issue(&self, instr: &WarpInstruction) -> bool;
    fn pipeline(&self) -> &Vec<Option<WarpInstruction>>;
    fn occupied(&self) -> &OccupiedSlots;
    fn active_lanes_in_pipeline(&self) -> usize;
    fn is_issue_partitioned(&self) -> bool;
    fn issue_reg_id(&self) -> usize;
    fn stallable(&self) -> bool;

    fn cycle(
        &mut self,
        operand_collector: &mut dyn crate::operand_collector::Writeback,
        scoreboard: &mut dyn scoreboard::Access,
        warps: &mut [warp::Warp],
        stats: &mut stats::PerKernel,
        mem_port: &mut dyn ic::Connection<ic::Packet<mem_fetch::MemFetch>>,
        result_port: Option<&mut register_set::RegisterSet>,
        cycle: u64,
    );
}

#[derive()]
pub struct PipelinedSimdUnit {
    pub id: usize,
    pub name: String,
    pub pipeline_depth: usize,
    pub pipeline_reg: Vec<Option<WarpInstruction>>,
    pub issue_reg_id: usize,
    pub active_insts_in_pipeline: usize,
    pub dispatch_reg: Option<WarpInstruction>,
    pub occupied: OccupiedSlots,
    pub config: Arc<config::GPU>,
}

impl std::fmt::Display for PipelinedSimdUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PipelinedSimdUnit")
    }
}

impl std::fmt::Debug for PipelinedSimdUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelinedSimdUnit").finish()
    }
}

impl PipelinedSimdUnit {
    pub fn new(
        id: usize,
        name: String,
        // result_port: Option<register_set::Ref>,
        depth: usize,
        config: Arc<config::GPU>,
        issue_reg_id: usize,
    ) -> Self {
        let pipeline_reg = (0..depth).map(|_| None).collect();
        Self {
            id,
            name,
            // result_port,
            pipeline_depth: depth,
            pipeline_reg,
            issue_reg_id,
            active_insts_in_pipeline: 0,
            dispatch_reg: None,
            occupied: BitArray::ZERO,
            config,
        }
    }

    // #[inline]
    pub fn num_active_instr_in_pipeline(&self) -> usize {
        self.pipeline_reg
            .iter()
            .map(Option::as_ref)
            .filter(Option::is_some)
            .count()
    }

    // #[inline]
    #[must_use]
    pub fn active_lanes_in_pipeline(&self) -> usize {
        let mut active_lanes = warp::ActiveMaskInner::ZERO;
        for instr in self.pipeline_reg.iter().flatten() {
            active_lanes |= *instr.active_mask;
        }
        active_lanes.count_ones()
    }

    // #[inline]
    pub fn issue(&mut self, src_reg: WarpInstruction) {
        // let mut active = 0;
        // match src_reg.opcode.op {
        //     crate::opcodes::Op::EXIT | crate::opcodes::Op::NOP => {}
        //     // crate::opcodes::Op::NOP => {}
        //     _ => {
        //         active += src_reg.active_mask.count_ones();
        //     }
        // }
        // crate::WIP_STATS.lock().executed_instructions += active as u64;

        // let opcode_count = self.stats.entry(src_reg.opcode.to_string()).or_insert();
        register_set::move_warp(
            Some(src_reg),
            &mut self.dispatch_reg,
            // format!(
            //     "{} moving register to dispatch register for issue",
            //     self.name,
            // ),
        );
        if let Some(ref dispatch_reg) = self.dispatch_reg {
            self.occupied.set(dispatch_reg.latency, true);
        }
    }

    // #[inline]
    #[must_use]
    pub fn can_issue(&self, instr: &WarpInstruction) -> bool {
        self.dispatch_reg.is_none() && !self.occupied[instr.latency]
    }
    // }

    pub fn cycle(&mut self, result_port: Option<&mut register_set::RegisterSet>, cycle: u64) {
        // fast path
        let no_result = self.pipeline_reg[0].is_none();
        let empty_pipeline = self.active_insts_in_pipeline == 0;
        // let nothing_occupied = self.occupied.not_any();
        // assert_eq!(nothing_occupied, empty_pipeline);
        let no_dispatch = self.dispatch_reg.is_none();

        if no_dispatch && no_result && empty_pipeline {
            self.occupied.shift_left(1);
            return;
        }

        log::debug!(
            "fu[{:03}] {:<10} cycle={:03}: \tpipeline={:?} ({}/{} active)",
            self.id,
            self.name,
            cycle,
            self.pipeline_reg
                .iter()
                .map(|reg| reg.as_ref().map(std::string::ToString::to_string))
                .collect::<Vec<_>>(),
            self.num_active_instr_in_pipeline(),
            self.pipeline_reg.len(),
        );

        // write results in the first pipeline register to the result port
        if let Some(result_port) = result_port {
            if let Some(result_reg) = self.pipeline_reg[0].take() {
                // move to EX_WB result port
                // let mut result_port = result_port.borrow_mut();
                // let mut result_port = result_port.try_lock();
                // let msg = format!(
                //     "{}: move pipeline[0] to result port {:?}",
                //     self.name, result_port.stage
                // );
                result_port.move_in_from(Some(result_reg));

                debug_assert!(self.active_insts_in_pipeline > 0);
                self.active_insts_in_pipeline -= 1;
            }
        }
        debug_assert_eq!(
            self.num_active_instr_in_pipeline(),
            self.active_insts_in_pipeline
        );

        // advance the instruction pipeline
        if self.active_insts_in_pipeline > 0 {
            for stage in 0..(self.pipeline_reg.len() - 1) {
                let current = self.pipeline_reg[stage + 1].take();
                let next = &mut self.pipeline_reg[stage];
                // let msg = format!("{} moving to next slot in pipeline register", self.name);
                register_set::move_warp(current, next);
            }
        }

        // dispatch register dispatches IN this pipeline
        if let Some(ref mut dispatch) = self.dispatch_reg {
            dispatch.dispatch_delay_cycles = dispatch.dispatch_delay_cycles.saturating_sub(1);
            if dispatch.dispatch_delay_cycles == 0 {
                // ready for dispatch
                let start_stage_idx = dispatch.latency - dispatch.initiation_interval;
                let dispatch = self.dispatch_reg.take().unwrap();
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
        }

        // occupied latencies are shifted each cycle
        // note: in rust, shift left is semantically equal to "towards the zero index"
        self.occupied.shift_left(1);
    }
}

#[cfg(test)]
mod tests {
    use utils::diff;

    #[allow(clippy::similar_names)]
    #[test]
    fn test_bitset_shift_right() {
        use bitvec::{
            array::BitArray,
            order::{Lsb0, Msb0},
            BitArr,
        };
        use playground::bitset::Bitset;
        use trace_model::ToBitString;

        let mut cpp = Bitset::default();
        let mut rust_lsb: BitArr!(for 32, in u32, Lsb0) = BitArray::ZERO;
        let mut rust_msb: BitArr!(for 32, in u32, Msb0) = BitArray::ZERO;
        assert_eq!(cpp.size(), rust_lsb.len());
        assert_eq!(cpp.size(), rust_msb.len());
        diff::assert_eq!(cpp: cpp.to_string(), rust_lsb: rust_lsb.to_bit_string());
        diff::assert_eq!(cpp: cpp.to_string(), rust_msb: rust_msb.to_bit_string());

        // set the value 5 (101)
        cpp.set(0, true);
        cpp.set(2, true);

        rust_lsb.set(0, true);
        rust_msb.set(0, true);

        rust_lsb.set(2, true);
        rust_msb.set(2, true);

        diff::assert_eq!(cpp: cpp.to_string(), rust_lsb: rust_lsb.to_bit_string());
        diff::assert_eq!(cpp: cpp.to_string(), rust_msb: rust_msb.to_bit_string());

        cpp.shift_right(1);
        rust_lsb.shift_left(1);
        rust_msb.shift_left(1);
        diff::assert_eq!(cpp: cpp.to_string(), rust_lsb: rust_lsb.to_bit_string());
        diff::assert_eq!(cpp: cpp.to_string(), rust_msb: rust_msb.to_bit_string());

        cpp.shift_right(3);
        rust_lsb.shift_left(3);
        rust_msb.shift_left(3);
        diff::assert_eq!(cpp: cpp.to_string(), rust_lsb: rust_lsb.to_bit_string());
        diff::assert_eq!(cpp: cpp.to_string(), rust_msb: rust_msb.to_bit_string());
    }
}
