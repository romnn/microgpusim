use super::{
    config, instruction::WarpInstruction, opcodes, register_set, simd_function_unit as fu,
};
use std::sync::{Arc, Mutex};

#[derive()]
pub struct SFU {
    config: Arc<config::GPU>,
    pipelined_simd_unit: fu::PipelinedSimdUnitImpl,
}

impl SFU {
    pub fn new(
        id: usize,
        result_port: register_set::Ref,
        config: Arc<config::GPU>,
        _stats: &Arc<Mutex<stats::Stats>>,
        cycle: super::Cycle,
        issue_reg_id: usize,
    ) -> Self {
        let pipeline_depth = config.max_sfu_latency;
        let pipelined_simd_unit = fu::PipelinedSimdUnitImpl::new(
            id,
            "SFU".to_string(),
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

impl std::fmt::Display for SFU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SFU")
    }
}

impl std::fmt::Debug for SFU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SFU").finish()
    }
}

impl fu::SimdFunctionUnit for SFU {
    fn can_issue(&self, instr: &WarpInstruction) -> bool {
        use opcodes::ArchOp;
        match instr.opcode.category {
            ArchOp::SFU_OP | ArchOp::ALU_SFU_OP | ArchOp::DP_OP => {
                self.pipelined_simd_unit.can_issue(instr)
            }
            _ => false,
        }
    }

    fn pipeline(&self) -> &Vec<Option<WarpInstruction>> {
        &self.pipelined_simd_unit.pipeline_reg
    }

    fn occupied(&self) -> &fu::OccupiedSlots {
        &self.pipelined_simd_unit.occupied()
    }

    fn id(&self) -> &str {
        &self.pipelined_simd_unit.name
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

    fn issue(&mut self, source_reg: WarpInstruction) {
        // let ready_reg = source_reg.get_ready(self.config.sub_core_model, self.issue_reg_id);
        // m_core->incexecstat((*ready_reg));
        // ready_reg.op_pipe = SP__OP;
        // m_core->incsp_stat(m_core->get_config()->warp_size, (*ready_reg)->latency);

        self.pipelined_simd_unit.issue(source_reg);
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
