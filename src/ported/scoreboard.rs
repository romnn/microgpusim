use super::instruction::WarpInstruction;
use std::collections::HashSet;

/// Scoreboard implementation
///
/// This should however not be needed in trace driven mode..
#[derive(Debug, Default)]
pub struct Scoreboard {
    pub core_id: usize,
    pub cluster_id: usize,

    pub register_table: Vec<HashSet<u32>>,
}

impl Scoreboard {
    #[must_use] pub fn new(core_id: usize, cluster_id: usize, max_warps: usize) -> Self {
        let register_table: Vec<_> = (0..max_warps).map(|_| HashSet::new()).collect();
        Self {
            core_id,
            cluster_id,
            register_table,
        }
    }

    /// Checks to see if registers used by an instruction are reserved in the scoreboard
    ///
    /// # Returns
    /// true if WAW or RAW hazard (no WAR since in-order issue)
    ///
    #[must_use] pub fn has_collision(&self, warp_id: usize, instr: &WarpInstruction) -> bool {
        use itertools::Itertools;

        // Get list of all input and output registers
        let mut instr_registers: HashSet<u32> = HashSet::new();
        instr_registers.extend(instr.outputs());
        instr_registers.extend(instr.inputs());

        log::trace!(
            "scoreboard: {} uses registers {:?} (in) + {:?} (out) = {:?}",
            instr,
            instr.inputs().sorted().collect::<Vec<_>>(),
            instr.outputs().sorted().collect::<Vec<_>>(),
            instr_registers.iter().sorted().collect::<Vec<_>>(),
        );

        // get the intersection of reserved registers and instruction registers
        let Some(reserved) = self.register_table.get(warp_id) else {
            return false;
        };
        log::trace!(
            "scoreboard: warp {} has reserved registers: {:?}",
            warp_id,
            reserved.iter().sorted().collect::<Vec<_>>(),
        );
        let mut intersection = instr_registers.intersection(reserved);
        intersection.next().is_some()
    }

    #[must_use] pub fn pending_writes(&self, warp_id: usize) -> &HashSet<u32> {
        &self.register_table[warp_id]
    }

    pub fn release_register(&mut self, warp_id: usize, reg_num: u32) {
        let removed = self.register_table[warp_id].remove(&reg_num);
        if removed {
            log::trace!(
                "scoreboard: warp {} releases register: {}",
                warp_id,
                reg_num
            );
        }
    }

    pub fn release_registers(&mut self, instr: &WarpInstruction) {
        for &out_reg in instr.outputs() {
            self.release_register(instr.warp_id, out_reg);
        }
    }

    pub fn reserve_register(&mut self, warp_id: usize, reg_num: u32) {
        let warp_registers = &mut self.register_table[warp_id];
        assert!(!warp_registers.contains(&reg_num), "trying to reserve an already reserved register (core_id={}, warp_id={}, reg_num={})",
           self.core_id, warp_id, reg_num);
        log::trace!(
            "scoreboard: warp {} reserves register: {}",
            warp_id,
            reg_num
        );
        self.register_table[warp_id].insert(reg_num);
    }

    pub fn reserve_registers(&mut self, instr: &WarpInstruction) {
        for &out_reg in instr.outputs() {
            self.reserve_register(instr.warp_id, out_reg);
        }
    }
}
