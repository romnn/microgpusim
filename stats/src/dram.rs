use super::mem::AccessKind;
use ndarray::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
    /// Access kind
    pub access_kind: AccessKind,
    /// Number of accesses
    pub num_accesses: u64,
}

#[derive(Clone, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DRAM {
    /// Kernel info
    pub kernel_info: super::KernelInfo,
    /// Bank accesses
    pub bank_accesses: ndarray::Array4<u64>,
    /// Number of cores
    pub num_cores: usize,
    /// Number of DRAM chips
    pub num_chips: usize,
    /// Number of banks
    pub num_banks: usize,
}

impl std::ops::AddAssign for DRAM {
    fn add_assign(&mut self, other: Self) {
        assert_eq!(self.num_cores, other.num_cores);
        assert_eq!(self.num_chips, other.num_chips);
        assert_eq!(self.num_banks, other.num_banks);

        self.bank_accesses = other.bank_accesses + self.bank_accesses.view_mut();
    }
}

impl DRAM {
    #[must_use]
    pub fn new(num_total_cores: usize, num_mem_units: usize, num_banks: usize) -> Self {
        Self {
            kernel_info: super::KernelInfo::default(),
            bank_accesses: ndarray::Array4::zeros((
                num_total_cores,
                num_mem_units,
                num_banks,
                AccessKind::count(),
            )),
            num_banks,
            num_cores: num_total_cores,
            num_chips: num_mem_units,
        }
    }

    #[must_use]
    pub fn bank_accesses_csv(&self, full: bool) -> Vec<BankAccessesCsvRow> {
        let mut out = Vec::new();

        for ((core_id, chip_id, bank_id, access_kind), num_accesses) in
            self.bank_accesses.indexed_iter()
        {
            // add single row to prevent empty data frame
            let need_row = out.is_empty();
            if !full && !need_row && *num_accesses < 1 {
                continue;
            }
            out.push(BankAccessesCsvRow {
                kernel_name: self.kernel_info.name.clone(),
                kernel_name_mangled: self.kernel_info.mangled_name.clone(),
                kernel_launch_id: self.kernel_info.launch_id,
                core_id,
                chip_id,
                bank_id,
                access_kind: AccessKind::from_repr(access_kind).unwrap(),
                num_accesses: *num_accesses,
            });
        }
        out
    }

    #[must_use]
    pub fn reduce(&self) -> HashMap<AccessKind, u64> {
        AccessKind::iter()
            .map(|access_kind| {
                (
                    access_kind,
                    self.bank_accesses
                        .slice(s![.., .., .., access_kind as usize])
                        .sum(),
                )
            })
            .collect()
    }

    #[must_use]
    pub fn total_reads(&self) -> u64 {
        AccessKind::reads()
            .map(|access_kind| {
                self.bank_accesses
                    .slice(s![.., .., .., access_kind as usize])
                    .sum()
            })
            .sum()
    }

    #[must_use]
    pub fn total_writes(&self) -> u64 {
        AccessKind::writes()
            .map(|access_kind| {
                self.bank_accesses
                    .slice(s![.., .., .., access_kind as usize])
                    .sum()
            })
            .sum()
    }
}
