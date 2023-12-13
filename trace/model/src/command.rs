use super::{dim::Dim, DeviceProperties};
use serde::{Deserialize, Serialize};

/// Information about a kernel launch.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelLaunch {
    /// Mangled kernel name.
    pub mangled_name: String,
    /// Unmangled kernel name.
    pub unmangled_name: String,
    /// Path to trace file for the kernel.
    pub trace_file: String,
    /// Unique kernel launch id.
    pub id: u64,
    /// The grid dimensions.
    pub grid: Dim,
    /// The block dimensions.
    pub block: Dim,
    /// The number of shared memory bytes used by the kernel.
    pub shared_mem_bytes: u32,
    /// The number of registers used by the kernel.
    pub num_registers: u32,
    /// Binary version of the compute capability of the device that collected the trace.
    pub binary_version: i32,
    /// CUDA stream ID the kernel is launch on
    pub stream_id: u64,
    /// Base address of the shared memory space
    pub shared_mem_base_addr: u64,
    /// Address limit of the shared memory space
    pub shared_mem_addr_limit: u64,
    /// Base address of the local memory space
    pub local_mem_base_addr: u64,
    /// Address limit of the local memory space
    pub local_mem_addr_limit: u64,
    /// The nvbit version of the tracer
    pub nvbit_version: String,
    /// Properties of the device that traced this kernel launch
    pub device_properties: DeviceProperties,
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

impl KernelLaunch {
    pub fn threads_per_block(&self) -> usize {
        let block = &self.block;
        block.x as usize * block.y as usize * block.z as usize
    }

    pub fn num_blocks(&self) -> usize {
        let grid = &self.grid;
        grid.x as usize * grid.y as usize * grid.z as usize
    }

    pub fn name(&self) -> &str {
        &self.unmangled_name
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
    pub fill_l2: bool,
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
