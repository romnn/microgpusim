use serde::{Deserialize, Serialize};

// use indexmap::IndexMap;
// #[derive(Clone, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
// pub struct JSONDRAM {
//     /// bank writes [shader id][dram chip id][bank id]
//     pub bank_writes: IndexMap<usize, IndexMap<usize, IndexMap<usize, u64>>>,
//     /// bank reads [shader id][dram chip id][bank id]
//     pub bank_reads: IndexMap<usize, IndexMap<usize, IndexMap<usize, u64>>>,
//     /// bank writes [dram chip id][bank id]
//     pub total_bank_writes: IndexMap<usize, IndexMap<usize, u64>>,
//     /// bank reads [dram chip id][bank id]
//     pub total_bank_reads: IndexMap<usize, IndexMap<usize, u64>>,
// }

#[derive(Clone, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerCoreDRAM {
    /// bank writes [shader id][dram chip id][bank id]
    pub bank_writes: Vec<Vec<Vec<u64>>>,
    /// bank reads [shader id][dram chip id][bank id]
    pub bank_reads: Vec<Vec<Vec<u64>>>,
}

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
    pub fn flatten(self) -> Self {
        todo!("flatten dram stats");
    }

    #[must_use]
    pub fn total_reads(&self) -> u64 {
        self.total_bank_reads.iter().flatten().sum()
    }

    #[must_use]
    pub fn total_writes(&self) -> u64 {
        self.total_bank_writes.iter().flatten().sum()
    }

    // #[must_use]
    // pub fn to_json(self) -> u64 {
    //     let bank_writes = IndexMap::new();
    //     for shader_id, per_shader in self.bank_writes.iter().enumerate() {
    //         for chip_id, per_chip in per_shader.iter().enumerate() {
    //             for bank_id, per_bank in per_shader.iter().enumerate() {
    //                 // bank_writes.entry(
    //
    //             }
    //         }
    //     }
    //
    // pub bank_writes: IndexMap<usize, IndexMap<usize, IndexMap<usize, u64>>>,
    //     JSONDRAM {
    // /// bank writes [shader id][dram chip id][bank id]
    // pub bank_writes: IndexMap<usize, IndexMap<usize, IndexMap<usize, u64>>>,
    // /// bank reads [shader id][dram chip id][bank id]
    // pub bank_reads: IndexMap<usize, IndexMap<usize, IndexMap<usize, u64>>>,
    // /// bank writes [dram chip id][bank id]
    // pub total_bank_writes: IndexMap<usize, IndexMap<usize, u64>>,
    // /// bank reads [dram chip id][bank id]
    // pub total_bank_reads: IndexMap<usize, IndexMap<usize, u64>>,
    //
    //     }
    // }
}
