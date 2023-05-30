use super::instruction::WarpInstruction;

#[derive(Debug, Default)]
pub struct Scoreboard {
    core_id: usize,
    cluster_id: usize,
    max_warps: usize,
}

impl Scoreboard {
    pub fn new(core_id: usize, cluster_id: usize, max_warps: usize) -> Self {
        Self {
            core_id,
            cluster_id,
            max_warps,
        }
    }

    pub fn check_collision(&self, warp_id: usize, instr: &WarpInstruction) -> bool {
        todo!("scoreboard: check collision");
    }

    pub fn pending_writes(&self, warp_id: usize) -> bool {
        todo!("scoreboard: pending writes");
    }
}
