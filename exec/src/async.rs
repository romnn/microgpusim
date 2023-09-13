use super::kernel::{Kernel, ThreadIndex};
use super::model;
use super::nop::ArithmeticNop;
use super::tracegen::Error;
use super::DevicePtr;
use bitvec::field::BitField;
use itertools::Itertools;
use std::sync::{atomic, Arc, Mutex};

// use tokio::runtime::Runtime;

#[async_trait::async_trait]
pub trait TraceGenerator {
    type Error;

    /// Trace kernel.
    fn trace_kernel<G, B, K>(
        // &mut self,
        &self,
        grid: G,
        block_size: B,
        kernel: K,
    ) -> Result<(), Error<K::Error, Self::Error>>
    where
        G: Into<trace_model::Dim>,
        B: Into<trace_model::Dim>,
        K: Kernel;

    /// Allocate a variable.
    // fn allocate<'s, 'a, T>(
    // fn allocate<'s, T, O>(
    fn allocate<'s, C, T>(
        &'s self,
        var: C,
        // var: &'a mut T,
        size: u64,
        mem_space: model::MemorySpace,
        // ) -> DevicePtr<'s, 'a, T>;
        // ) -> DevicePtr<'s, T, O>
    ) -> DevicePtr<'s, C, T>;
    // where
    //     T: Container,
    //     <T as Container>::Elem: Zero;
}

// // Create the runtime
// let rt = Runtime::new().unwrap();
//
// // Spawn a blocking function onto the runtime
// rt.spawn_blocking(|| {
//     println!("now running on a worker thread");
// });

// pub trait TraceGenerator {
//
// #[async_trait::async_trait]

#[cfg(test)]
mod tests {
    use color_eyre::eyre;
    // use utils::diff;
    // use std::path::PathBuf;
    // use trace_model as model;
    // use gpucachesim::exec::{self, MemorySpace, TraceGenerator, Tracer};

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_it_works() -> eyre::Result<()> {
        Ok(())
    }
}
