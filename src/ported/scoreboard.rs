use super::instruction::WarpInstruction;
use std::collections::HashSet;

/// Scoreboard implementation
///
/// This should however not be needed in trace driven mode..
#[derive(Debug, Default)]
pub struct Scoreboard {
    core_id: usize,
    cluster_id: usize,
    max_warps: usize,

    register_table: Vec<HashSet<u32>>,
    /// Register that depend on a long operation (global, local or tex memory)
    long_op_registers: Vec<HashSet<u32>>,
}

impl Scoreboard {
    pub fn new(core_id: usize, cluster_id: usize, max_warps: usize) -> Self {
        let register_table: Vec<_> = (0..max_warps).map(|_| HashSet::new()).collect();
        let long_op_registers = register_table.clone();
        Self {
            core_id,
            cluster_id,
            max_warps,
            register_table,
            long_op_registers,
        }
    }

    /// Checks to see if registers used by an instruction are reserved in the scoreboard
    ///
    /// # Returns
    /// true if WAW or RAW hazard (no WAR since in-order issue)
    ///
    pub fn has_collision(&self, warp_id: usize, instr: &WarpInstruction) -> bool {
        // Get list of all input and output registers
        let mut instr_registers: HashSet<u32> = HashSet::new();
        instr_registers.extend(instr.outputs());
        instr_registers.extend(instr.inputs());
        // ar1 = 0;
        // ar2 = 0;

        // predicate register number
        // if instr.pred > 0
        //   inst_regs.insert(inst->pred);
        // if (inst->ar1 > 0)
        //   inst_regs.insert(inst->ar1);
        // if (inst->ar2 > 0)
        //   inst_regs.insert(inst->ar2);

        // check for collision, get the intersection of reserved registers and instruction registers
        let mut intersection = instr_registers.intersection(&self.register_table[warp_id]);
        !intersection.next().is_some()
        // todo!("scoreboard: check collision");
    }

    pub fn pending_writes(&self, warp_id: usize) -> bool {
        !self.register_table[warp_id].is_empty()
        // todo!("scoreboard: pending writes");
    }
}
