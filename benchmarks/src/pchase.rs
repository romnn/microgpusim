#![allow(warnings)]

use gpucachesim::exec::tracegen::{TraceGenerator, Tracer};
use gpucachesim::exec::{DevicePtr, Kernel, MemorySpace, ThreadBlock, ThreadIndex};
// use num_traits::{Float, Zero};

use tokio::sync::Mutex;

#[derive(
    Debug, Clone, Copy, strum::EnumIter, strum::EnumString, Hash, PartialEq, Eq, PartialOrd, Ord,
)]
#[strum(ascii_case_insensitive)]
pub enum Memory {
    L1Data,
    L1ReadOnly,
    L1Texture,
    L2,
}

/// Fine-grain p-chase kernel.
pub struct FineGrainPChase<'a> {
    dev_array: Mutex<DevicePtr<&'a mut Vec<u32>>>,
    size: usize,
    stride: usize,
    warmup_iterations: usize,
    iter_size: usize,
}

#[async_trait::async_trait]
impl<'a> Kernel for FineGrainPChase<'a>
// impl<'a, T> Kernel for FineGrainPChase<'a, T>
// where
//     T: Float + Send + Sync,
{
    type Error = std::convert::Infallible;

    #[gpucachesim::exec::inject_reconvergence_points]
    async fn run(&self, block: &ThreadBlock, tid: &ThreadIndex) -> Result<(), Self::Error> {
        let mut j = 0u32;

        let mut dev_array = self.dev_array.lock().await;

        let iter_start = (self.warmup_iterations as i64 * -(self.iter_size as i64));
        let iter_end = self.iter_size as i64;
        log::debug!(
            "p-chase: {} .. {} [n={}, stride={}]",
            iter_start,
            iter_end,
            self.size,
            self.stride
        );
        for k in iter_start..iter_end {
            if (k >= 0) {
                j = dev_array[(tid, j as usize)];
            } else {
                j = dev_array[(tid, j as usize)];
            }
        }

        dev_array[(tid, self.size)] = j;
        dev_array[(tid, self.size + 1)] = dev_array[(tid, j as usize)];
        Ok(())
    }

    fn name(&self) -> Option<&str> {
        Some("fine_grain_p_chase")
    }
}

/// Fine-grain p-chase application.
pub async fn pchase(
    memory: Memory,
    size_bytes: usize,
    stride_bytes: usize,
    warmup_iterations: usize,
    iter_size: usize,
) -> super::Result {
    let mut traces = vec![];
    let tracer = Tracer::new();

    let size = size_bytes / std::mem::size_of::<u32>();
    let stride = stride_bytes / std::mem::size_of::<u32>();

    // initialize host array with pointers into dev_array
    let mut array: Vec<u32> = vec![0; size as usize + 2];
    for i in 0..size {
        array[i] = ((i + stride) % size) as u32;
    }

    array[size] = 0;
    array[size + 1] = 0;

    // allocate memory for each vector on simulated GPU device
    let mut dev_array = tracer
        .allocate(&mut array, MemorySpace::Global, Some("array"))
        .await;

    if memory == Memory::L2 {
        dev_array.bypass_l1 = true;
    }

    // number of thread blocks in grid
    let kernel = FineGrainPChase {
        dev_array: Mutex::new(dev_array),
        size,
        stride,
        warmup_iterations,
        iter_size,
    };
    let grid_size = 1;
    let block_size = 1;
    let trace = tracer.trace_kernel(grid_size, block_size, kernel).await?;
    traces.push(trace);
    Ok((tracer.commands().await, traces))
}

#[cfg(test)]
mod tests {
    use color_eyre::eyre;
    use gpucachesim::exec::tracegen::fmt::{self, SimplifiedTraceInstruction};
    use ndarray::Array1;
    use utils::diff;

    const KB: usize = 1024;
    const MB: usize = 1024 * 1024;
    const EPSILON: f32 = 0.0001;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_generate_trace() -> eyre::Result<()> {
        crate::tests::init_test();

        let iter_size = ((48 * KB) / 2) / std::mem::size_of::<u32>();
        let size_bytes = 16 * KB;
        let stride_bytes = 4;
        let warmup_iterations = 1;

        assert_eq!(stride_bytes, std::mem::size_of::<u32>());

        let (_commands, kernel_traces) = super::pchase(
            super::Memory::L1Data,
            size_bytes,
            // size_bytes,
            // 1,
            stride_bytes,
            warmup_iterations,
            iter_size,
        )
        .await?;
        assert_eq!(kernel_traces.len(), 1);
        let (_launch_config, trace) = kernel_traces.into_iter().next().unwrap();
        let warp_traces = trace.clone().to_warp_traces();
        let first_warp = &warp_traces[&(trace_model::Dim::ZERO, 0)];

        let simplified_trace = fmt::simplify_warp_trace(&first_warp, true).collect::<Vec<_>>();
        for inst in &simplified_trace {
            println!("{}", inst);
        }
        Ok(())
    }
}
