use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sim {
    pub cycles: u64,
    pub instructions: u64,
    pub num_blocks: u64,
}

impl std::ops::AddAssign for Sim {
    fn add_assign(&mut self, other: Self) {
        self.cycles += other.cycles;
        self.instructions += other.instructions;
        self.num_blocks += other.num_blocks;
    }
}
