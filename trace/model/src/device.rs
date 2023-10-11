use serde::{Deserialize, Serialize};

/// Device properties.
#[derive(Debug, Default, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Properties {
    /// Unified adressing.
    ///
    /// See `cudaDevAttrUnifiedAddressing`
    pub unified_addressing: Option<bool>,

    /// Maximum number of registers per SM.
    ///
    /// See `cudaDevAttrMaxRegistersPerMultiprocessor`
    pub max_registers_per_sm: Option<bool>,

    /// Warp size.
    ///
    /// See `cudaDevAttrWarpSize`
    pub warp_size: Option<usize>,

    /// Maximum constant memory in bytes.
    ///
    /// See `cudaDevAttrTotalConstantMemory`
    pub constant_memory_bytes: Option<usize>,

    /// Maximum number of threads per block
    ///
    /// See `cudaDevAttrMaxThreadsPerBlock`
    pub max_threads_per_block: Option<usize>,

    /// Maximum number of blocks per SM
    ///
    /// See `cudaDevAttrMaxBlocksPerMultiprocessor`
    pub max_blocks_per_sm: Option<usize>,

    /// Maximum shared memory size per block in bytes.
    ///
    /// See `cudaDevAttrMaxSharedMemoryPerBlock`
    pub max_shared_memory_per_block_bytes: Option<usize>,

    /// Maximum shared memory size per SM in bytes.
    ///
    /// See `cudaDevAttrMaxSharedMemoryPerMultiprocessor`
    pub max_shared_memory_per_sm_bytes: Option<usize>,
}
