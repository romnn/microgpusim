// use super::model;

/// Thread index.
#[derive(
    Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct ThreadIndex {
    pub block_idx: trace_model::Dim,
    pub block_dim: trace_model::Dim,
    pub thread_idx: trace_model::Dim,
}

/// A kernel implementation.
pub trait Kernel {
    type Error: std::error::Error;

    /// Run an instance of the kernel on a thread identified by its index
    fn run(&mut self, idx: &ThreadIndex) -> Result<(), Self::Error>;

    fn name(&self) -> &str;
}

/// A kernel implementation.
#[async_trait::async_trait]
pub trait AsyncKernel {
    type Error: std::error::Error;

    /// Run an instance of the kernel on a thread identified by its index
    async fn run(&mut self, idx: &ThreadIndex) -> Result<(), Self::Error>;

    fn name(&self) -> &str;
}
