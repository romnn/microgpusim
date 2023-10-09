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
        // unsigned int start_time, end_time;
        let mut j = 0u32;
        // __shared__ uint32_t s_tvalue[ITER_SIZE];
        // __shared__ uint32_t s_index[ITER_SIZE];

        // for (size_t k = 0; k < ITER_SIZE; k++) {
        //   s_index[k] = 0;
        //   s_tvalue[k] = 0;
        // }

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
                // start_time = clock();
                // log::trace!("p-chase: measure j={}", j);
                j = dev_array[(tid, j as usize)];
                // s_index[k] = j;
                // end_time = clock();

                // s_tvalue[k] = end_time - start_time;
            } else {
                // log::trace!("p-chase: warmup j={}", j);
                j = dev_array[(tid, j as usize)];
            }
        }

        // let mut dev_array = self.dev_array.lock().await;
        dev_array[(tid, self.size)] = j;
        dev_array[(tid, self.size + 1)] = dev_array[(tid, j as usize)];

        // for (size_t k = 0; k < NUM_LOADS; k++) {
        //   index[k] = s_index[k];
        //   duration[k] = s_tvalue[k];
        // }

        Ok(())
    }

    fn name(&self) -> Option<&str> {
        Some("fine_grain_p_chase")
    }
}

// pub async fn pchase<T>(a: &Vec<T>, b: &Vec<T>, result: &mut Vec<T>) -> super::Result
// where
//     T: Float + Zero + Send + Sync,

/// Fine-grain p-chase application.
pub async fn pchase(
    memory: Memory,
    size_bytes: usize,
    stride_bytes: usize,
    warmup_iterations: usize,
    iter_size: usize,
) -> super::Result {
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
    let dev_array = tracer
        .allocate(&mut array, MemorySpace::Global, Some("array"))
        .await;

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
    Ok((tracer.commands().await, vec![trace]))
}

#[cfg(test)]
mod tests {
    use color_eyre::eyre;
    use gpucachesim::exec::tracegen::testing::{self, SimplifiedTraceInstruction};
    use ndarray::Array1;
    use utils::diff;

    const KB: usize = 1024;
    const MB: usize = 1024 * 1024;
    const EPSILON: f32 = 0.0001;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_correctness() -> eyre::Result<()> {
        crate::tests::init_test();

        let iter_size = ((48 * KB) / 2) / std::mem::size_of::<u32>();
        let size_bytes = 16 * KB;
        let stride_bytes = 4;
        let warmup_iterations = 1;

        assert_eq!(stride_bytes, std::mem::size_of::<u32>());

        let (_commands, kernel_traces) = super::pchase(
            super::Memory::L1Data,
            size_bytes,
            stride_bytes,
            warmup_iterations,
            iter_size,
        )
        .await?;
        assert_eq!(kernel_traces.len(), 1);
        let (_launch_config, trace) = kernel_traces.into_iter().next().unwrap();
        let warp_traces = trace.clone().to_warp_traces();
        let first_warp = &warp_traces[&(trace_model::Dim::ZERO, 0)];

        let simplified_trace = testing::simplify_warp_trace(&first_warp).collect::<Vec<_>>();
        for inst in &simplified_trace {
            println!("{}", inst);
        }

        // create host vectors
        // let n = 100;
        // let mut a: Vec<f32> = vec![0.0; n];
        // let mut b: Vec<f32> = vec![0.0; n];
        // let mut result: Vec<f32> = vec![0.0; n];
        // let mut ref_result: Vec<f32> = vec![0.0; n];
        //
        // // initialize vectors
        // for i in 0..n {
        //     let angle = i as f32;
        //     a[i] = angle.sin() * angle.sin();
        //     b[i] = angle.cos() * angle.cos();
        // }
        //
        // let ndarray_result = {
        //     let ref_a = Array1::from_shape_vec(n, a.clone())?;
        //     let ref_b = Array1::from_shape_vec(n, b.clone())?;
        //     ref_a + ref_b
        // };
        // let (_commands, kernel_traces) = super::vectoradd(&a, &b, &mut result).await?;
        // assert_eq!(kernel_traces.len(), 1);
        // let (_launch_config, trace) = kernel_traces.into_iter().next().unwrap();
        // super::reference(&a, &b, &mut ref_result);
        //
        // let ref_result = Array1::from_shape_vec(n, ref_result)?;
        // let result = Array1::from_shape_vec(n, result)?;
        // dbg!(&ref_result);
        // dbg!(&result);
        //
        // if !approx::abs_diff_eq!(ref_result, ndarray_result, epsilon = EPSILON) {
        //     diff::assert_eq!(have: ref_result, want: ndarray_result);
        // }
        // if !approx::abs_diff_eq!(result, ndarray_result, epsilon = EPSILON) {
        //     diff::assert_eq!(have: result, want: ndarray_result);
        // }
        //
        // let warp_traces = trace.clone().to_warp_traces();
        // let first_warp = &warp_traces[&(trace_model::Dim::ZERO, 0)];
        //
        // let simplified_trace = testing::simplify_warp_trace(&first_warp).collect::<Vec<_>>();
        // for inst in &simplified_trace {
        //     println!("{}", inst);
        // }
        // diff::assert_eq!(
        //     have: simplified_trace,
        //     want: [
        //         ("LDG", 0, "11111111111111111111111111111111", 0),
        //         ("LDG", 512, "11111111111111111111111111111111", 0),
        //         ("STG", 1024, "11111111111111111111111111111111", 0),
        //         ("EXIT", 0, "11111111111111111111111111111111", 0),
        //     ].into_iter().enumerate().map(SimplifiedTraceInstruction::from).collect::<Vec<_>>()
        // );
        Ok(())
    }
}
