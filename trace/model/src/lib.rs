pub mod dim;
pub use dim::{Dim, Point};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct MemAccessTraceEntry {
    pub cuda_ctx: u64,
    pub sm_id: u32,
    pub kernel_id: u64,
    pub block_id: Dim,
    pub warp_id_in_sm: u32,
    pub warp_id_in_block: u32,
    pub warp_size: u32,
    pub line_num: u32,
    pub instr_data_width: u32,
    pub instr_opcode: String,
    pub instr_offset: u32,
    pub instr_idx: u32,
    pub instr_predicate: nvbit_model::Predicate,
    pub instr_mem_space: nvbit_model::MemorySpace,
    pub instr_is_mem: bool,
    pub instr_is_load: bool,
    pub instr_is_store: bool,
    pub instr_is_extended: bool,
    pub dest_regs: [u32; 1],
    pub num_dest_regs: u32,
    pub src_regs: [u32; 5],
    pub num_src_regs: u32,
    pub active_mask: u32,
    /// Accessed address per thread of a warp
    pub addrs: [u64; 32],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[repr(transparent)]
pub struct MemAccessTrace(pub Vec<MemAccessTraceEntry>);

impl MemAccessTrace {
    #[must_use]
    pub fn is_valid(&self) -> bool {
        is_valid_trace(&self.0)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[must_use]
    pub fn iter(&self) -> std::slice::Iter<MemAccessTraceEntry> {
        self.0.iter()
    }

    #[must_use]
    pub fn to_warp_traces(self) -> HashMap<(Dim, u32), Vec<MemAccessTraceEntry>> {
        let mut warp_traces: HashMap<(Dim, u32), Vec<MemAccessTraceEntry>> = HashMap::new();
        for entry in self.0 {
            warp_traces
                .entry((entry.block_id.clone(), entry.warp_id_in_block))
                .or_default()
                .push(entry);
        }
        warp_traces
    }
}

impl std::ops::Deref for MemAccessTrace {
    type Target = Vec<MemAccessTraceEntry>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Sanity checks a trace for _any_ valid sorting.
///
/// In order to pass, a trace must not contain any non-consecutive duplicate block or warp ids.
#[must_use]
#[inline]
pub fn is_valid_trace(trace: &[MemAccessTraceEntry]) -> bool {
    use itertools::Itertools;

    let duplicate_blocks = trace
        .iter()
        .map(|t| &t.block_id)
        .dedup()
        .duplicates()
        .count();
    let duplicate_warp_ids = trace
        .iter()
        .map(|t| (&t.block_id, &t.warp_id_in_block))
        .dedup()
        .duplicates()
        .count();

    // assert_eq!(duplicate_blocks, 0);
    // assert_eq!(duplicate_warp_ids, 0);
    duplicate_blocks == 0 && duplicate_warp_ids == 0
}

/// A memory allocation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemAllocation {
    pub device_ptr: u64,
    pub num_bytes: u64,
}

/// Information about a kernel launch.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KernelLaunch {
    pub name: String,
    pub trace_file: String,
    pub id: u64,
    pub grid: Dim,
    pub block: Dim,
    pub shared_mem_bytes: u32,
    pub num_registers: u32,
    pub binary_version: i32,
    pub stream_id: u64,
    pub shared_mem_base_addr: u64,
    pub local_mem_base_addr: u64,
    pub nvbit_version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemcpyHtoD {
    pub allocation_name: Option<String>,
    pub dest_device_addr: u64,
    pub num_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemAlloc {
    pub allocation_name: Option<String>,
    pub device_ptr: u64,
    pub num_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Command {
    MemcpyHtoD(MemcpyHtoD),
    MemAlloc(MemAlloc),
    KernelLaunch(KernelLaunch),
}
