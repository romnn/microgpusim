// use super::model;

/// Thread index.
#[derive(
    Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct ThreadIndex {
    pub(crate) block_id: trace_model::Point,
    pub(crate) warp_id_in_block: usize,
    pub(crate) thread_id_in_warp: usize,
    // pub(crate) unqiquethread_id_in_warp: u64,
    // pub(crate) idx: u64,
    pub kernel_launch_id: u64,
    pub block_idx: trace_model::Dim,
    pub block_dim: trace_model::Dim,
    pub thread_idx: trace_model::Dim,
}

/// A kernel implementation.
pub trait Kernel {
    type Error: std::error::Error;

    /// Run an instance of the kernel on a thread identified by its index
    fn run(&mut self, idx: &ThreadIndex) -> Result<(), Self::Error>;

    fn name(&self) -> Option<&str> {
        None
    }
}
