use super::{model, tracegen};
use std::sync::Arc;

pub struct ThreadBlock {
    pub(crate) barrier: Arc<tokio::sync::Barrier>,
    pub memory: Arc<dyn tracegen::MemoryAccess + Send + Sync>,
    pub thread_id: ThreadIndex,
}

impl ThreadBlock {
    pub async fn synchronize_threads(&self) {
        let inst = model::ThreadInstruction::Barrier;
        self.memory.push_thread_instruction(&self.thread_id, inst);
        self.barrier.wait().await;
    }

    pub fn reconverge_branch(&self, branch_id: usize) {
        let inst = model::ThreadInstruction::Reconverge(branch_id);
        self.memory.push_thread_instruction(&self.thread_id, inst);
    }

    pub fn took_branch(&self, branch_id: usize) {
        let inst = model::ThreadInstruction::TookBranch(branch_id);
        self.memory.push_thread_instruction(&self.thread_id, inst);
    }

    pub fn start_branch(&self, branch_id: usize) {
        let inst = model::ThreadInstruction::Branch(branch_id);
        self.memory.push_thread_instruction(&self.thread_id, inst);
    }
}

#[derive(
    Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct ThreadIndex {
    pub(crate) block_id: trace_model::Point,
    pub(crate) warp_id_in_block: usize,
    pub(crate) thread_id_in_warp: usize,
    pub kernel_launch_id: u64,
    pub block_idx: trace_model::Dim,
    pub block_dim: trace_model::Dim,
    pub thread_idx: trace_model::Dim,
}

/// A kernel implementation.
#[async_trait::async_trait]
pub trait Kernel {
    type Error: std::error::Error;

    /// Run an instance of the kernel on a thread identified by its index
    async fn run(&self, block: &ThreadBlock, idx: &ThreadIndex) -> Result<(), Self::Error>;

    fn name(&self) -> Option<&str> {
        None
    }
}
