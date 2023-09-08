use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sim {
    pub cycles: u64,
    pub instructions: u64,
    pub num_blocks: u64,
}
