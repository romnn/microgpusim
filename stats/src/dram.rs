use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DRAM {
    /// bank writes [shader id][dram chip id][bank id]
    pub bank_writes: Vec<Vec<Vec<u64>>>,
    /// bank reads [shader id][dram chip id][bank id]
    pub bank_reads: Vec<Vec<Vec<u64>>>,
    /// bank writes [dram chip id][bank id]
    pub total_bank_writes: Vec<Vec<u64>>,
    /// bank reads [dram chip id][bank id]
    pub total_bank_reads: Vec<Vec<u64>>,
}

impl DRAM {
    #[must_use]
    pub fn new(num_total_cores: usize, num_mem_units: usize, num_banks: usize) -> Self {
        let total_bank_writes = vec![vec![0; num_banks]; num_mem_units];
        let total_bank_reads = total_bank_writes.clone();
        let bank_reads = vec![total_bank_reads.clone(); num_total_cores];
        let bank_writes = bank_reads.clone();
        Self {
            bank_writes,
            bank_reads,
            total_bank_writes,
            total_bank_reads,
        }
    }

    #[must_use]
    pub fn total_reads(&self) -> u64 {
        self.total_bank_reads.iter().flatten().sum()
    }

    #[must_use]
    pub fn total_writes(&self) -> u64 {
        self.total_bank_writes.iter().flatten().sum()
    }
}