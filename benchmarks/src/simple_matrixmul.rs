use gpucachesim::exec::tracegen::{TraceGenerator, Tracer};
use gpucachesim::exec::{
    model::{Dim, MemorySpace},
    DevicePtr, Kernel, ThreadBlock, ThreadIndex,
};
use num_traits::{Float, NumCast, Zero};
use rand::{
    distributions::{self, Distribution},
    Rng,
};
use tokio::sync::Mutex;

// Number of threads in each thread block
pub const BLOCK_DIM: usize = 32;

#[derive(Debug)]
struct SimpleMatrixmul<'a, T> {
    dev_a: Mutex<DevicePtr<&'a Vec<T>>>,
    dev_b: Mutex<DevicePtr<&'a Vec<T>>>,
    dev_result: Mutex<DevicePtr<&'a mut Vec<T>>>,
    m: usize,
    n: usize,
    p: usize,
}

#[async_trait::async_trait]
impl<'a, T> Kernel for SimpleMatrixmul<'a, T>
where
    T: Float + std::ops::AddAssign + Send + Sync + std::fmt::Debug,
{
    type Error = std::convert::Infallible;

    #[gpucachesim::exec::inject_reconvergence_points]
    async fn run(&self, block: &ThreadBlock, tid: &ThreadIndex) -> Result<(), Self::Error> {
        // 2D block and 2D thread: each thread computes one cell in dev_result.

        // size_t i = blockIdx.y * blockDim.y + threadIdx.y;
        let i = (tid.block_idx.y * tid.block_dim.y + tid.thread_idx.y) as usize;
        // size_t j = blockIdx.x * blockDim.x + threadIdx.x;
        let j = (tid.block_idx.x * tid.block_dim.x + tid.thread_idx.x) as usize;

        // do not process outside the matrix.
        // do not forget the equal sign!
        if i >= self.m || j >= self.p {
            return Ok(());
        }

        // float acc_sum{0};
        let mut acc_sum = T::zero();

        // for (size_t k = 0; k < n; k++) {
        //   acc_sum += mat_a[i * n + k] * mat_b[k * p + j];
        // }
        let dev_a = self.dev_a.lock().await;
        let dev_b = self.dev_b.lock().await;
        for k in 0..self.n {
            acc_sum += dev_a[(tid, i * self.n + k)] * dev_b[(tid, k * self.p + j)];
        }

        let mut dev_result = self.dev_result.lock().await;
        dev_result[(tid, i * self.p + j)] = acc_sum;
        Ok(())
    }

    fn name(&self) -> Option<&str> {
        Some("simple_matrixmul")
    }
}

pub fn reference<T>(matrix_a: &[T], matrix_b: &[T], result: &mut [T], m: usize, n: usize, p: usize)
where
    T: Float + std::ops::AddAssign,
{
    for mi in 0..m {
        for pi in 0..p {
            let mut sum = T::zero();
            for ni in 0..n {
                sum += matrix_a[mi * n + ni] * matrix_b[ni * p + pi];
            }

            result[mi * p + pi] = sum;
        }
    }
}

/// Simple matrixmul benchmark application.
pub async fn benchmark<T>(m: usize, n: usize, p: usize) -> super::Result
where
    T: Float + Zero + std::ops::AddAssign + Send + Sync + std::fmt::Debug,
    distributions::Open01: Distribution<T>,
{
    let mut rng = rand::thread_rng();

    // create host vectors
    let mut matrix_a: Vec<T> = vec![T::zero(); m * n];
    let mut matrix_b: Vec<T> = vec![T::zero(); n * p];
    let mut result: Vec<T> = vec![T::zero(); m * p];

    // initialize vectors
    for av in &mut matrix_a {
        *av = NumCast::from(rng.gen_range(-256.0..256.0)).unwrap();
    }
    for bv in &mut matrix_b {
        *bv = NumCast::from(rng.gen_range(-256.0..256.0)).unwrap();
    }

    simple_matrixmul(&matrix_a, &matrix_b, &mut result, m, n, p).await
}

pub async fn simple_matrixmul<T>(
    matrix_a: &Vec<T>,
    matrix_b: &Vec<T>,
    result: &mut Vec<T>,
    m: usize,
    n: usize,
    p: usize,
) -> super::Result
where
    T: Float + Zero + std::ops::AddAssign + Send + Sync + std::fmt::Debug,
{
    let tracer = Tracer::new();

    // allocate memory for each vector on simulated GPU device
    let dev_a = tracer
        .allocate(matrix_a, MemorySpace::Global, Some("a"))
        .await;
    let dev_b = tracer
        .allocate(matrix_b, MemorySpace::Global, Some("b"))
        .await;
    let dev_result = tracer
        .allocate(result, MemorySpace::Global, Some("result"))
        .await;

    println!("({m} x {n}) x ({n} x {p}) = ({m} x {p})");

    // number of thread blocks in grid
    let block_dim: Dim = (BLOCK_DIM as u32, BLOCK_DIM as u32).into();
    let grid_x = p as f64 / block_dim.x as f64;
    let grid_y = m as f64 / block_dim.y as f64;
    let grid_dim: Dim = (grid_x.ceil() as u32, grid_y.ceil() as u32).into();
    println!("block dim: {block_dim}");
    println!("grid dim:  {grid_dim}");

    assert!(grid_dim.x > 0);
    assert!(grid_dim.y > 0);
    assert!(grid_dim.z > 0);

    let mut kernel: SimpleMatrixmul<T> = SimpleMatrixmul {
        dev_a: Mutex::new(dev_a),
        dev_b: Mutex::new(dev_b),
        dev_result: Mutex::new(dev_result),
        m,
        n,
        p,
    };
    let trace = tracer
        .trace_kernel(grid_dim, block_dim, &mut kernel)
        .await?;
    Ok((tracer.commands().await, vec![trace]))
}

#[cfg(test)]
mod tests {
    use color_eyre::eyre;
    use gpucachesim::exec::tracegen::fmt;
    use ndarray::Array2;
    use rand::Rng;
    use utils::diff;

    const EPSILON: f32 = 0.01;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_correctness() -> eyre::Result<()> {
        crate::tests::init_test();
        let mut rng = rand::thread_rng();

        let (m, n, p) = (4, 4, 4);

        // create host vectors
        let mut matrix_a: Vec<f32> = vec![0.0; m * n];
        let mut matrix_b: Vec<f32> = vec![0.0; n * p];
        let mut result: Vec<f32> = vec![0.0; m * p];
        let mut ref_result: Vec<f32> = vec![0.0; m * p];

        // initialize random matrix
        for av in &mut matrix_a {
            *av = rng.gen_range(-256.0..256.0);
        }
        for bv in &mut matrix_b {
            *bv = rng.gen_range(-256.0..256.0);
        }

        let ndarray_result = {
            let ref_a = Array2::from_shape_vec((m, n), matrix_a.clone())?;
            let ref_b = Array2::from_shape_vec((n, p), matrix_b.clone())?;
            ref_a.dot(&ref_b)
        };
        let (_commands, kernel_traces) =
            super::simple_matrixmul(&matrix_a, &matrix_b, &mut result, m, n, p).await?;
        assert_eq!(kernel_traces.len(), 1);
        let (_launch_config, trace) = kernel_traces.into_iter().next().unwrap();
        super::reference(&matrix_a, &matrix_b, &mut ref_result, m, n, p);

        let ref_result = Array2::from_shape_vec((m, p), ref_result)?;
        let result = Array2::from_shape_vec((m, p), result)?;
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

        let simplified_trace = fmt::simplify_warp_trace(&first_warp, true).collect::<Vec<_>>();
        for inst in &simplified_trace {
            println!("{}", inst);
        }

        // diff::assert_eq!(
        //     have: simplified_trace,
        //     want: [
        //         ("LDG", 0, "11111111111111111111000000000000", 0),
        //     ].into_iter().enumerate().map(SimplifiedTraceInstruction::from).collect::<Vec<_>>()
        // );
        Ok(())
    }
}
