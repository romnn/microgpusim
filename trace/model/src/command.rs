use super::dim::Dim;
use serde::{Deserialize, Serialize};

/// Information about a kernel launch.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelLaunch {
    pub mangled_name: String,
    pub unmangled_name: String,
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

impl std::cmp::Ord for KernelLaunch {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

impl std::cmp::PartialOrd for KernelLaunch {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MemcpyHtoD {
    pub allocation_name: Option<String>,
    pub dest_device_addr: u64,
    pub num_bytes: u64,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MemAlloc {
    pub allocation_name: Option<String>,
    pub device_ptr: u64,
    pub num_bytes: u64,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Command {
    MemcpyHtoD(MemcpyHtoD),
    MemAlloc(MemAlloc),
    KernelLaunch(KernelLaunch),
}
