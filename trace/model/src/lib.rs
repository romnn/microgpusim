use serde::{Deserialize, Serialize};
use std::ffi::OsStr;
use std::path::PathBuf;

/// Extract app arguments.
///
/// This skips the binary (`argv[0]`) itself.
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemAccessTraceEntry {
    pub cuda_ctx: u64,
    pub kernel_id: u64,
    pub block_id: nvbit_model::Dim,
    pub warp_id: u32,
    pub line_num: u32,
    pub instr_opcode: String,
    pub instr_offset: u32,
    pub instr_idx: u32,
    pub instr_predicate: nvbit_model::Predicate,
    pub instr_mem_space: nvbit_model::MemorySpace,
    pub instr_is_load: bool,
    pub instr_is_store: bool,
    pub instr_is_extended: bool,
    pub active_mask: u32,
    /// Accessed address per thread of a warp
    pub addrs: [u64; 32],
}

/// A memory allocation.
#[derive(Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
pub struct MemAllocation {
    pub device_ptr: u64,
    pub num_bytes: u64,
}

/// Information about a kernel launch.
#[derive(Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
pub struct KernelLaunch {
    pub name: String,
    pub trace_file: PathBuf,
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

#[derive(Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
pub enum Command {
    MemcpyHtoD {
        dest_device_addr: u64,
        num_bytes: u64,
    },
    KernelLaunch(KernelLaunch),
}
