use serde::{Deserialize, Serialize};
use utils::box_slice;

#[derive(Clone, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BankAccessesCsvRow {
    pub kernel_name: String,
    pub kernel_name_mangled: String,
    pub kernel_launch_id: usize,
    /// Core ID
    pub core_id: usize,
    /// DRAM chip ID
    pub chip_id: usize,
    /// Bank ID
    pub bank_id: usize,
    /// Number of reads
    pub reads: u64,
    /// Number of writes
    pub writes: u64,
}

#[derive(Clone, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccessesCsvRow {
    pub kernel_name: String,
    pub kernel_name_mangled: String,
    pub kernel_launch_id: usize,
    /// DRAM chip ID
    pub chip_id: usize,
    /// Bank ID
    pub bank_id: usize,
    /// Number of reads
    pub reads: u64,
    /// Number of writes
    pub writes: u64,
}

pub type PerChipBankStats = Box<[Box<[Box<[u64]>]>]>;
pub type BankStats = Box<[Box<[u64]>]>;

#[derive(Clone, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DRAM {
    /// Kernel info
    pub kernel_info: super::KernelInfo,
    /// Number of bank writes [shader id][dram chip id][bank id]
    pub bank_writes: PerChipBankStats,
    /// Number of bank reads [shader id][dram chip id][bank id]
    pub bank_reads: PerChipBankStats,
    /// Number of bank writes [dram chip id][bank id]
    pub total_bank_writes: BankStats,
    /// Number of bank reads [dram chip id][bank id]
    pub total_bank_reads: BankStats,

    /// Number of cores
    pub num_cores: usize,
    /// Number of DRAM chips
    pub num_chips: usize,
    /// Number of banks
    pub num_banks: usize,
}

impl std::ops::AddAssign for DRAM {
    fn add_assign(&mut self, other: Self) {
        for core_id in 0..self.num_cores {
            for chip_id in 0..self.num_chips {
                for bank_id in 0..self.num_banks {
                    self.bank_reads[core_id][chip_id][bank_id] +=
                        other.bank_reads[core_id][chip_id][bank_id];
                    self.bank_writes[core_id][chip_id][bank_id] +=
                        other.bank_writes[core_id][chip_id][bank_id];
                }
            }
        }
        for chip_id in 0..self.num_chips {
            for bank_id in 0..self.num_banks {
                self.total_bank_writes[chip_id][bank_id] +=
                    other.total_bank_writes[chip_id][bank_id];
                self.total_bank_reads[chip_id][bank_id] += other.total_bank_reads[chip_id][bank_id];
            }
        }
    }
}

impl DRAM {
    #[must_use]
    pub fn new(num_total_cores: usize, num_mem_units: usize, num_banks: usize) -> Self {
        let total_bank_writes = box_slice![box_slice![0; num_banks]; num_mem_units];
        let total_bank_reads = total_bank_writes.clone();
        let bank_reads = box_slice![total_bank_reads.clone(); num_total_cores];
        let bank_writes = bank_reads.clone();
        Self {
            kernel_info: super::KernelInfo::default(),
            bank_writes,
            bank_reads,
            total_bank_writes,
            total_bank_reads,
            num_banks,
            num_cores: num_total_cores,
            num_chips: num_mem_units,
        }
    }

    #[must_use]
    pub fn bank_accesses_csv(&self) -> Vec<BankAccessesCsvRow> {
        let mut out = Vec::new();
        for core_id in 0..self.num_cores {
            for chip_id in 0..self.num_chips {
                for bank_id in 0..self.num_banks {
                    let reads = self.bank_reads[core_id][chip_id][bank_id];
                    let writes = self.bank_writes[core_id][chip_id][bank_id];
                    out.push(BankAccessesCsvRow {
                        kernel_name: self.kernel_info.name.clone(),
                        kernel_name_mangled: self.kernel_info.mangled_name.clone(),
                        kernel_launch_id: self.kernel_info.launch_id,
                        core_id,
                        chip_id,
                        bank_id,
                        reads,
                        writes,
                    });
                }
            }
        }
        out
    }

    #[must_use]
    pub fn accesses_csv(&self) -> Vec<AccessesCsvRow> {
        let mut out = Vec::new();
        for chip_id in 0..self.num_chips {
            for bank_id in 0..self.num_banks {
                let reads = self.total_bank_reads[chip_id][bank_id];
                let writes = self.total_bank_writes[chip_id][bank_id];
                out.push(AccessesCsvRow {
                    kernel_name: self.kernel_info.name.clone(),
                    kernel_name_mangled: self.kernel_info.mangled_name.clone(),
                    kernel_launch_id: self.kernel_info.launch_id,
                    chip_id,
                    bank_id,
                    reads,
                    writes,
                });
            }
        }
        out
    }

    #[must_use]
    pub fn total_reads(&self) -> u64 {
        self.total_bank_reads.iter().flat_map(AsRef::as_ref).sum()
    }

    #[must_use]
    pub fn total_writes(&self) -> u64 {
        self.total_bank_writes.iter().flat_map(AsRef::as_ref).sum()
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
