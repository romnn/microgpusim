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

impl std::fmt::Display for KernelLaunch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = [self.unmangled_name.as_str(), self.mangled_name.as_str()]
            .into_iter()
            .filter(|name| !name.is_empty())
            .next()
            .unwrap_or("unknown");
        f.debug_struct("KernelLaunch")
            .field("name", &name)
            .field("id", &self.id)
            .field("block", &self.block)
            .finish()
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MemcpyHtoD {
    pub allocation_name: Option<String>,
    pub dest_device_addr: u64,
    pub num_bytes: u64,
}

impl std::fmt::Display for MemcpyHtoD {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemcpyHtoD")
            .field("name", &self.allocation_name)
            .field("dest_addr", &self.dest_device_addr)
            .field("size", &human_bytes::human_bytes(self.num_bytes as f64))
            .finish()
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MemAlloc {
    pub allocation_name: Option<String>,
    pub device_ptr: u64,
    pub num_bytes: u64,
}

impl std::fmt::Display for MemAlloc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemAlloc")
            .field("name", &self.allocation_name)
            .field("addr", &self.device_ptr)
            .field("size", &human_bytes::human_bytes(self.num_bytes as f64))
            .finish()
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Command {
    MemcpyHtoD(MemcpyHtoD),
    MemAlloc(MemAlloc),
    KernelLaunch(KernelLaunch),
}

impl std::fmt::Display for Command {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MemAlloc(inner) => inner.fmt(f),
            Self::MemcpyHtoD(inner) => inner.fmt(f),
            Self::KernelLaunch(inner) => inner.fmt(f),
        }
    }
}
