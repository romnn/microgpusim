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
pub async fn benchmark<T>(n: usize) -> super::Result
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

pub async fn vectoradd<T>(a: &Vec<T>, b: &Vec<T>, result: &mut Vec<T>) -> super::Result
where
    T: Float + Zero + Send + Sync,
{
    let tracer = Tracer::new();

    assert_eq!(a.len(), b.len());
    assert_eq!(b.len(), result.len());
    let n = a.len();

    // allocate memory for each vector on simulated GPU device
    let dev_a = tracer.allocate(a, MemorySpace::Global, Some("a")).await;
    let dev_b = tracer.allocate(b, MemorySpace::Global, Some("b")).await;
    let dev_result = tracer
        .allocate(result, MemorySpace::Global, Some("result"))
        .await;

    // number of thread blocks in grid
    let grid_size = (n as f64 / <f64 as From<_>>::from(BLOCK_SIZE)).ceil() as u32;
    let kernel: VecAdd<T> = VecAdd {
        dev_a: Mutex::new(dev_a),
        dev_b: Mutex::new(dev_b),
        dev_result: Mutex::new(dev_result),
        n,
    };
    let trace = tracer.trace_kernel(grid_size, BLOCK_SIZE, kernel).await?;
    Ok((tracer.commands().await, vec![trace]))
}

#[cfg(test)]
mod tests {
    use color_eyre::eyre;
    use gpucachesim::exec::tracegen::fmt::{self, SimplifiedTraceInstruction};
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
        let (_commands, kernel_traces) = super::vectoradd(&a, &b, &mut result).await?;
        assert_eq!(kernel_traces.len(), 1);
        let (_launch_config, trace) = kernel_traces.into_iter().next().unwrap();
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

        let simplified_trace = fmt::simplify_warp_trace(&first_warp).collect::<Vec<_>>();
        for inst in &simplified_trace {
            println!("{}", inst);
        }
        diff::assert_eq!(
            have: simplified_trace,
            want: [
                ("LDG", 0, "11111111111111111111111111111111", 0),
                ("LDG", 512, "11111111111111111111111111111111", 0),
                ("STG", 1024, "11111111111111111111111111111111", 0),
                ("EXIT", 0, "11111111111111111111111111111111", 0),
            ].into_iter().enumerate().map(SimplifiedTraceInstruction::from).collect::<Vec<_>>()
        );

        Ok(())
    }
}
