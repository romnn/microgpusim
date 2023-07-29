use serde::{Deserialize, Serialize};
use std::ffi::OsStr;
use std::path::PathBuf;

/// Extract app arguments.
///
/// This skips the binary (`argv[0]`) itself.
#[must_use]
pub fn app_args(bin_name: Option<&str>) -> Vec<String> {
    std::env::args()
        .skip_while(|arg| {
            let arg = PathBuf::from(arg);
            let Some(arg_name) = arg.file_name().and_then(OsStr::to_str) else {
                return false;
            };
            match bin_name {
                Some(b) => b == arg_name,
                None => false,
            }
        })
        .collect()
}

/// Buidld unique app prefix.
///
/// Used for storing traces related to a specific invocation
/// of an application with arguments.
pub fn app_prefix(bin_name: Option<&str>) -> String {
    let mut args: Vec<_> = app_args(bin_name);
    if let Some(executable) = args.get_mut(0) {
        *executable = PathBuf::from(&*executable)
            .file_name()
            .and_then(OsStr::to_str)
            .map(String::from)
            .unwrap_or(String::new());
    }
    args.join("-")
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct MemAccessTraceEntry {
    pub cuda_ctx: u64,
    pub kernel_id: u64,
    pub block_id: nvbit_model::Dim,
    pub thread_id: nvbit_model::Dim,
    pub unique_thread_id: u32,
    pub global_warp_id: u32,
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
    // pub dest_reg: Option<u32>,
    pub dest_regs: [u32; 1],
    pub num_dest_regs: u32,
    pub src_regs: [u32; 5],
    pub num_src_regs: u32,
    pub active_mask: u32,
    /// Accessed address per thread of a warp
    pub addrs: [u64; 32],
}

impl MemAccessTraceEntry {
    #[must_use]
    #[inline]
    pub fn sort_key(&self) -> (&u64, &nvbit_model::Dim, &u32) {
        (&self.kernel_id, &self.block_id, &self.warp_id_in_block)
    }
}

impl Ord for MemAccessTraceEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.sort_key().cmp(&other.sort_key())
    }
}

impl PartialOrd for MemAccessTraceEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
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
    pub grid: nvbit_model::Dim,
    pub block: nvbit_model::Dim,
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
