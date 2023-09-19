use color_eyre::eyre;
use gpucachesim::exec::tracegen::{TraceGenerator, Tracer};
use gpucachesim::exec::{DevicePtr, Kernel, MemorySpace, ThreadBlock, ThreadIndex};
use num_traits::{Float, Zero};

use tokio::sync::Mutex;

struct VecAdd<'a, T> {
    dev_a: Mutex<DevicePtr<&'a Vec<T>>>,
    dev_b: Mutex<DevicePtr<&'a Vec<T>>>,
    dev_result: Mutex<DevicePtr<&'a mut Vec<T>>>,
    n: usize,
}

#[async_trait::async_trait]
impl<'a, T> Kernel for VecAdd<'a, T>
where
    T: Float + Send + Sync,
{
    type Error = std::convert::Infallible;

    #[gpucachesim::exec::inject_reconvergence_points]
    async fn run(&self, block: &ThreadBlock, tid: &ThreadIndex) -> Result<(), Self::Error> {
        let idx = (tid.block_idx.x * tid.block_dim.x + tid.thread_idx.x) as usize;

        let dev_a = self.dev_a.lock().await;
        let dev_b = self.dev_b.lock().await;
        let mut dev_result = self.dev_result.lock().await;

        if idx < self.n {
            dev_result[(tid, idx)] = dev_a[(tid, idx)] + dev_b[(tid, idx)];
        } else {
            // this is no longer required because we inject reconvergence points.
            // dev_result[tid] = dev_a[tid] + dev_b[tid];
        }
        Ok(())
    }

    fn name(&self) -> Option<&str> {
        Some("VecAdd")
    }
}

// Number of threads in each thread block
pub const BLOCK_SIZE: u32 = 1024;

pub fn reference<T>(a: &[T], b: &[T], result: &mut [T])
where
    T: Float,
{
    for (i, sum) in result.iter_mut().enumerate() {
        *sum = a[i] + b[i];
    }
}

/// Vectoradd benchmark application.
pub async fn benchmark<T>(
    n: usize,
) -> eyre::Result<(
    trace_model::command::KernelLaunch,
    trace_model::MemAccessTrace,
)>
where
    T: Float + Zero + Send + Sync,
{
    // create host vectors
    let mut a: Vec<T> = vec![T::zero(); n];
    let mut b: Vec<T> = vec![T::zero(); n];
    let mut result: Vec<T> = vec![T::zero(); n];

    // initialize vectors
    for i in 0..n {
        let angle = T::from(i).unwrap();
        a[i] = angle.sin() * angle.sin();
        b[i] = angle.cos() * angle.cos();
        result[i] = T::zero();
    }

    vectoradd(&a, &b, &mut result).await
}

pub async fn vectoradd<T>(
    a: &Vec<T>,
    b: &Vec<T>,
    result: &mut Vec<T>,
) -> eyre::Result<(
    trace_model::command::KernelLaunch,
    trace_model::MemAccessTrace,
)>
where
    T: Float + Zero + Send + Sync,
{
    let tracer = Tracer::new();

    assert_eq!(a.len(), b.len());
    assert_eq!(b.len(), result.len());
    let n = a.len();

    // allocate memory for each vector on simulated GPU device
    let dev_a = tracer.allocate(a, MemorySpace::Global).await;
    let dev_b = tracer.allocate(b, MemorySpace::Global).await;
    let dev_result = tracer.allocate(result, MemorySpace::Global).await;

    // number of thread blocks in grid
    let grid_size = (n as f64 / <f64 as From<_>>::from(BLOCK_SIZE)).ceil() as u32;
    let kernel: VecAdd<T> = VecAdd {
        dev_a: Mutex::new(dev_a),
        dev_b: Mutex::new(dev_b),
        dev_result: Mutex::new(dev_result),
        n,
    };
    let trace = tracer.trace_kernel(grid_size, BLOCK_SIZE, kernel).await?;
    Ok(trace)
    
}

#[cfg(test)]
mod tests {
    use color_eyre::eyre;
    use gpucachesim::exec::tracegen::testing::{self, SimplifiedTraceInstruction};
    use ndarray::Array1;
    use utils::diff;

    const EPSILON: f32 = 0.0001;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_correctness() -> eyre::Result<()> {
        // create host vectors
        let n = 100;
        let mut a: Vec<f32> = vec![0.0; n];
        let mut b: Vec<f32> = vec![0.0; n];
        let mut result: Vec<f32> = vec![0.0; n];
        let mut ref_result: Vec<f32> = vec![0.0; n];

        // initialize vectors
        for i in 0..n {
            let angle = i as f32;
            a[i] = angle.sin() * angle.sin();
            b[i] = angle.cos() * angle.cos();
        }

        let ndarray_result = {
            let ref_a = Array1::from_shape_vec(n, a.clone())?;
            let ref_b = Array1::from_shape_vec(n, b.clone())?;
            ref_a + ref_b
        };
        let (_launch_config, trace) = super::vectoradd(&a, &b, &mut result).await?;
        super::reference(&a, &b, &mut ref_result);

        let ref_result = Array1::from_shape_vec(n, ref_result)?;
        let result = Array1::from_shape_vec(n, result)?;
        dbg!(&ref_result);
        dbg!(&result);

        if !approx::abs_diff_eq!(ref_result, ndarray_result, epsilon = EPSILON) {
            diff::assert_eq!(have: ref_result, want: ndarray_result);
        }
        if !approx::abs_diff_eq!(result, ndarray_result, epsilon = EPSILON) {
            diff::assert_eq!(have: result, want: ndarray_result);
        }

        let warp_traces = trace.clone().to_warp_traces();
        let first_warp = &warp_traces[&(trace_model::Dim::ZERO, 0)];

        testing::print_warp_trace(first_warp);
        diff::assert_eq!(
            have: testing::simplify_warp_trace(first_warp).collect::<Vec<_>>(),
            want: [
                ("LDG", 0, "11111111111111111111111111111111", 0),
                ("LDG", 400, "11111111111111111111111111111111", 0),
                ("STG", 800, "11111111111111111111111111111111", 0),
                ("EXIT", 0, "11111111111111111111111111111111", 0),
            ].into_iter().enumerate().map(SimplifiedTraceInstruction::from).collect::<Vec<_>>()
        );

        Ok(())
    }

    // #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    // async fn generate_trace() -> eyre::Result<()> {
    //     let n = 100;
    //     let tracer = Tracer::new();
    //
    //     let mut a: Vec<f32> = vec![0.0; n];
    //     let mut b: Vec<f32> = vec![0.0; n];
    //     let mut result: Vec<f32> = vec![0.0; n];
    //
    //     // initialize vectors
    //     for i in 0..n {
    //         let angle = i as f32;
    //         a[i] = angle.sin() * angle.sin();
    //         b[i] = angle.cos() * angle.cos();
    //     }
    //
    //     // allocate memory for each vector on simulated GPU device
    //     let dev_a = tracer.allocate(&a, MemorySpace::Global).await;
    //     let dev_b = tracer.allocate(&b, MemorySpace::Global).await;
    //     let dev_result = tracer.allocate(&mut result, MemorySpace::Global).await;
    //
    //     // number of thread blocks in grid
    //     let grid_size = (n as f64 / <f64 as From<_>>::from(super::BLOCK_SIZE)).ceil() as u32;
    //
    //     let kernel: super::VecAdd<f32> = super::VecAdd {
    //         dev_a: Mutex::new(dev_a),
    //         dev_b: Mutex::new(dev_b),
    //         dev_result: Mutex::new(dev_result),
    //         n,
    //     };
    //     tracer
    //         .trace_kernel(grid_size, super::BLOCK_SIZE, kernel)
    //         .await?;
    //     // let stats = sim.run_to_completion()?;
    //
    //     // sum up vector c and print result divided by n.
    //     // this should equal 1 within
    //     // let total_sum: f32 = result.iter().copied().sum();
    //     // println!(
    //     //     "Final sum = {total_sum}; sum/n = {:.2} (should be ~1)\n",
    //     //     total_sum / n as f32
    //     // );
    //
    //     // assert!(false);
    //     Ok(())
    // }

    // #[test]
    // pub fn trace_instructions() -> eyre::Result<()> {
    //     let traces_dir = PathBuf::from(file!())
    //         .parent()
    //         .unwrap()
    //         .join("../results/vectorAdd/vectorAdd-dtype-32-length-100/trace");
    //     dbg!(&traces_dir);
    //     let rmp_trace_file_path = traces_dir.join("kernel-0.msgpack");
    //     dbg!(&rmp_trace_file_path);
    //
    //     let mut reader = utils::fs::open_readable(rmp_trace_file_path)?;
    //     let full_trace: model::MemAccessTrace = rmp_serde::from_read(&mut reader)?;
    //     let warp_traces = full_trace.to_warp_traces();
    //     dbg!(&warp_traces[&(model::Dim::ZERO, 0)]
    //         .iter()
    //         .map(|entry| (&entry.instr_opcode, &entry.active_mask))
    //         .collect::<Vec<_>>());
    //
    //     assert!(false);
    //     Ok(())
    // }
}
