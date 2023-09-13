use serde::{Deserialize, Serialize};

/// A memory allocation.
#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MemAllocation {
    pub device_ptr: u64,
    pub num_bytes: u64,
}
