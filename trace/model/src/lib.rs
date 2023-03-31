use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub fn app_prefix(bin_name: Option<&str>) -> String {
    let mut args: Vec<_> = std::env::args()
        .into_iter()
        .skip_while(|arg| {
            let arg = PathBuf::from(arg);
            let Some(arg_name) = arg.file_name().and_then(std::ffi::OsStr::to_str) else {
                return false;
            };
            match bin_name {
                Some(b) => b == arg_name,
                None => false,
            }
        })
        .collect();
    if let Some(executable) = args.get_mut(0) {
        *executable = PathBuf::from(&*executable)
            .file_name()
            .and_then(std::ffi::OsStr::to_str)
            .map(String::from)
            .unwrap_or(String::new());
    }
    args.join("-")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemAccessTraceEntry {
    pub cuda_ctx: u64,
    pub grid_launch_id: u64,
    pub cta_id: nvbit_model::Dim,
    pub warp_id: u32,
    pub instr_opcode: String,
    pub instr_offset: u32,
    pub instr_idx: u32,
    pub instr_predicate: nvbit_model::Predicate,
    pub instr_mem_space: nvbit_model::MemorySpace,
    pub instr_is_load: bool,
    pub instr_is_store: bool,
    pub instr_is_extended: bool,
    /// Accessed address per thread of a warp
    pub addrs: [u64; 32],
}

#[derive(Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
pub struct MemAllocation {
    pub device_ptr: u64,
    pub bytes: usize,
}
