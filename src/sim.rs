use itertools::Itertools;
use num_traits::{AsPrimitive, Float, NumCast, Zero};
use nvbit_model as model;
use std::sync::Arc;

#[derive(Debug)]
pub struct DevicePtr<'s, 'a, T> {
    inner: &'a mut T,
    sim: &'s Simulation,
}

/// Convert multi-dimensional index into flat linear index.
pub trait ToFlatIndex {
    fn flatten(&self) -> usize;
}

impl ToFlatIndex for usize {
    fn flatten(&self) -> usize {
        *self
    }
}

impl<T, O, I> std::ops::Index<I> for DevicePtr<'_, '_, T>
where
    T: std::ops::Index<I, Output = O> + std::fmt::Debug,
    I: ToFlatIndex + std::fmt::Debug,
{
    type Output = O;

    fn index(&self, idx: I) -> &Self::Output {
        let elem_size = std::mem::size_of::<O>();
        let flat_idx = idx.flatten();
        let addr = elem_size * flat_idx;
        // println!("{:?}[{:?}] => {}", &self, &idx, &addr);
        &self.inner[idx]
    }
}

impl<T, O, I> std::ops::IndexMut<I> for DevicePtr<'_, '_, T>
where
    T: std::ops::IndexMut<I, Output = O> + std::fmt::Debug,
    I: ToFlatIndex + std::fmt::Debug,
{
    fn index_mut(&mut self, idx: I) -> &mut Self::Output {
        let elem_size = std::mem::size_of::<O>();
        let flat_idx = idx.flatten();
        let addr = elem_size * flat_idx;
        // println!("{:?}[{:?}] => {}", &self, &idx, &addr);
        &mut self.inner[idx]
    }
}

/// Thread index.
#[derive(Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ThreadIndex {
    pub block_idx: model::Dim,
    pub block_dim: model::Dim,
    pub thread_idx: model::Dim,
}

/// A kernel implementation.
pub trait Kernel {
    type Error: std::error::Error;

    /// Run an instance of the kernel on a thread identified by its index
    #[allow(clippy::missing_errors_doc)]
    fn run(&mut self, idx: &ThreadIndex) -> Result<(), Self::Error>;
}

/// Simulation
#[derive(Debug, Default)]
pub struct Simulation {}

impl Simulation {
    // pub fn new(first_level: Arc<dyn CacheLevel>, main_mem: MainMemory) -> Self {
    //     Self {}
    // }

    #[must_use]
    pub fn new() -> Self {
        Self {}
    }

    /// Allocate a variable.
    pub fn allocate<'s, 'a, T>(&'s self, var: &'a mut T) -> DevicePtr<'s, 'a, T> {
        DevicePtr {
            inner: var,
            sim: self,
        }
    }

    /// Launches a kernel.
    ///
    /// # Errors
    /// When the kernel fails.
    pub fn launch_kernel<G, B, K>(
        &self,
        grid: G,
        block_size: B,
        mut kernel: K,
    ) -> Result<(), K::Error>
    where
        G: Into<model::Dim>,
        B: Into<model::Dim>,
        K: Kernel,
    {
        let grid: model::Dim = grid.into();
        let block_size: model::Dim = block_size.into();
        dbg!(&grid);
        dbg!(&block_size);

        // loop over the grid
        for block_idx in grid {
            let mut thread_idx = ThreadIndex {
                block_idx,
                block_dim: block_size,
                thread_idx: block_size,
            };

            // loop over the block size (must run on same sms)
            // and form warps
            let mut threads = block_size.into_iter();
            for warp in &threads.chunks(32) {
                for warp_thread_idx in warp {
                    thread_idx.thread_idx = warp_thread_idx;
                    // println!("calling thread {thread_idx:?}");
                    kernel.run(&thread_idx)?;
                }
            }
        }
        Ok(())
    }
}
