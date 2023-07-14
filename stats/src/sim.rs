use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sim {
    pub cycles: usize,
    pub instructions: usize,
}
