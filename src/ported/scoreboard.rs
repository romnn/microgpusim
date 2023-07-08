use super::instruction::{MemorySpace, WarpInstruction};
use std::collections::HashSet;

/// Scoreboard implementation
///
/// This should however not be needed in trace driven mode..
#[derive(Debug, Default)]
pub struct Scoreboard {
    core_id: usize,
    cluster_id: usize,
    max_warps: usize,

    pub register_table: Vec<HashSet<u32>>,
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

        // get the intersection of reserved registers and instruction registers
        let Some(reserved) = self.register_table.get(warp_id) else {
            return false;
        };
        let mut intersection = instr_registers.intersection(&reserved);
        intersection.next().is_some()
        // todo!("scoreboard: check collision");
    }

    pub fn pending_writes(&self, warp_id: usize) -> &HashSet<u32> {
        &self.register_table[warp_id] // .is_empty()
                                      // todo!("scoreboard: pending writes");
    }

    pub fn release_register(&mut self, warp_id: usize, reg_num: u32) {
        // if (!(reg_table[wid].find(regnum) != reg_table[wid].end()))
        //     return;
        //   SHADER_DPRINTF(SCOREBOARD, "Release register - warp:%d, reg: %d\n", wid,
        //                  regnum);
        self.register_table[warp_id].remove(&reg_num);
    }

    pub fn release_registers(&mut self, instr: &WarpInstruction) {
        for &out_reg in instr.outputs() {
            self.release_register(instr.warp_id, out_reg);
            self.long_op_registers[instr.warp_id].remove(&out_reg);
        }
    }

    pub fn reserve_register(&mut self, warp_id: usize, reg_num: u32) {
        let warp_registers = &mut self.register_table[warp_id];
        if warp_registers.contains(&reg_num) {
            panic!("trying to reserve an already reserved register (core_id={}, warp_id={}, reg_num={})",
           self.core_id, warp_id, reg_num);
        }
        self.register_table[warp_id].insert(reg_num);
    }

    pub fn reserve_registers(&mut self, instr: &WarpInstruction) {
        for &out_reg in instr.outputs() {
            self.reserve_register(instr.warp_id, out_reg);
        }

        // Keep track of long operations
        if instr.is_load()
            && matches!(
                instr.memory_space,
                Some(MemorySpace::Global | MemorySpace::Local | MemorySpace::Texture)
            )
        {
            // inst->space.get_type() == local_space ||
            // inst->space.get_type() == param_space_kernel ||
            // inst->space.get_type() == param_space_local ||
            // inst->space.get_type() == param_space_unclassified ||
            for &out_reg in instr.outputs() {
                self.long_op_registers[instr.warp_id].insert(out_reg);
            }
        }
    }
}
