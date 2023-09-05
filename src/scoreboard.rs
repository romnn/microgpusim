use super::instruction::WarpInstruction;
use std::collections::HashSet;

/// Scoreboard access.
///
/// The scoreboard keeps track of registers used by warps inside an SM.
pub trait Access<I>: Sync + Send + 'static {
    /// Checks to see if registers used by an instruction are reserved in the scoreboard
    ///
    /// # Returns
    /// `true` if WAW or RAW hazard (no WAR since in-order issue)
    #[must_use]
    fn has_collision(&self, warp_id: usize, instr: &I) -> bool;

    /// Get all pending writes for a warp.
    #[must_use]
    fn pending_writes(&self, warp_id: usize) -> &HashSet<u32>;

    /// Release register for a warp.
    fn release(&mut self, warp_id: usize, reg_num: u32);

    /// Release all output registers for an instruction.
    fn release_all(&mut self, instr: &I);

    /// Reserve a register for a warp.
    fn reserve(&mut self, warp_id: usize, reg_num: u32);

    /// Reserve all output registers for an instruction.
    fn reserve_all(&mut self, instr: &I);
}

/// Scoreboard configuration.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct Config {
    pub core_id: usize,
    pub cluster_id: usize,
    pub max_warps: usize,
}

/// Scoreboard implementation
///
/// This should however not be needed in trace driven mode..
#[derive(Debug, Default)]
pub struct Scoreboard {
    pub core_id: usize,
    pub cluster_id: usize,

    pub warp_registers: Box<[HashSet<u32>]>,
}

impl Scoreboard {
    #[must_use]
    pub fn new(config: Config) -> Self {
        let Config {
            max_warps,
            core_id,
            cluster_id,
        } = config;
        let warp_registers = utils::box_slice![HashSet::new(); max_warps];
        Self {
            core_id,
            cluster_id,
            warp_registers,
        }
    }
}

impl Access<WarpInstruction> for Scoreboard {
    #[inline]
    fn has_collision(&self, warp_id: usize, instr: &WarpInstruction) -> bool {
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
        let Some(reserved) = self.warp_registers.get(warp_id) else {
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

    #[inline]
    fn pending_writes(&self, warp_id: usize) -> &HashSet<u32> {
        &self.warp_registers[warp_id]
    }

    #[inline]
    fn release(&mut self, warp_id: usize, reg_num: u32) {
        let removed = self.warp_registers[warp_id].remove(&reg_num);
        if removed {
            log::trace!(
                "scoreboard: warp {} releases register: {}",
                warp_id,
                reg_num
            );
        }
    }

    #[inline]
    fn release_all(&mut self, instr: &WarpInstruction) {
        for &out_reg in instr.outputs() {
            self.release(instr.warp_id, out_reg);
        }
    }

    #[inline]
    fn reserve(&mut self, warp_id: usize, reg_num: u32) {
        let warp_registers = &mut self.warp_registers[warp_id];
        assert!(
            !warp_registers.contains(&reg_num),
            "trying to reserve an already reserved register (core_id={}, warp_id={}, reg_num={})",
            self.core_id,
            warp_id,
            reg_num
        );
        log::trace!(
            "scoreboard: warp {} reserves register: {}",
            warp_id,
            reg_num
        );
        self.warp_registers[warp_id].insert(reg_num);
    }

    #[inline]
    fn reserve_all(&mut self, instr: &WarpInstruction) {
        for &out_reg in instr.outputs() {
            self.reserve(instr.warp_id, out_reg);
        }
    }
}
