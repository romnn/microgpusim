use color_eyre::eyre;
use gpucachesim::exec::model::{Dim, MemorySpace};
use gpucachesim::exec::{
    tracegen::{TraceGenerator, Tracer},
    Kernel,
};
use num_traits::{Float, NumCast, Zero};
use tokio::sync::Mutex;

// Number of threads in each thread block
pub const TILE_DIM: u32 = 16;
pub const BLOCK_ROWS: u32 = 16;

pub mod naive {
    use super::{BLOCK_ROWS, TILE_DIM};
    use gpucachesim::exec::{DevicePtr, Kernel, ThreadBlock, ThreadIndex};
    use tokio::sync::Mutex;

    #[derive(Debug)]
    pub struct Transpose<'a, T> {
        pub dev_mat: Mutex<DevicePtr<&'a Vec<T>>>,
        pub dev_result: Mutex<DevicePtr<&'a mut Vec<T>>>,
        pub rows: usize,
        pub cols: usize,
    }

    #[async_trait::async_trait]
    impl<'a, T> Kernel for Transpose<'a, T>
    where
        T: num_traits::Float + std::ops::AddAssign + Send + Sync + std::fmt::Debug,
    {
        type Error = std::convert::Infallible;

        #[gpucachesim::exec::inject_reconvergence_points]
        async fn run(&self, block: &ThreadBlock, tid: &ThreadIndex) -> Result<(), Self::Error> {
            // int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
            let x_index = (tid.block_idx.x * TILE_DIM + tid.thread_idx.x) as usize;
            // int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
            let y_index = (tid.block_idx.y * TILE_DIM + tid.thread_idx.y) as usize;

            // int index_in = xIndex + width * yIndex;
            let index_in = x_index + self.cols * y_index;
            // int index_out = yIndex + height * xIndex;
            let index_out = y_index + self.rows * x_index;

            // for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
            //   odata[index_out + i] = idata[index_in + i * width];
            // }
            let mut i = 0;
            while i < TILE_DIM as usize {
                let dev_mat = self.dev_mat.lock().await;
                let mut dev_result = self.dev_result.lock().await;
                dev_result[(tid, index_out + i)] = dev_mat[(tid, index_in + i * self.cols)];
                i += BLOCK_ROWS as usize;
            }
            Ok(())
        }

        fn name(&self) -> Option<&str> {
            Some("transpose_naive")
        }
    }
}

pub mod coalesced {
    use super::{BLOCK_ROWS, TILE_DIM};
    use gpucachesim::exec::{DevicePtr, Kernel, ThreadBlock, ThreadIndex};
    use tokio::sync::Mutex;

    #[derive(Debug)]
    pub struct Transpose<'a, T> {
        pub dev_mat: Mutex<DevicePtr<&'a Vec<T>>>,
        pub dev_result: Mutex<DevicePtr<&'a mut Vec<T>>>,
        pub rows: usize,
        pub cols: usize,

        /// Shared memory array used to store the tiles
        pub shared_mem_tiles: Mutex<DevicePtr<Vec<T>>>,
    }

    #[async_trait::async_trait]
    impl<'a, T> Kernel for Transpose<'a, T>
    where
        T: num_traits::Float + std::ops::AddAssign + Send + Sync + std::fmt::Debug,
    {
        type Error = std::convert::Infallible;

        #[gpucachesim::exec::inject_reconvergence_points]
        async fn run(&self, block: &ThreadBlock, tid: &ThreadIndex) -> Result<(), Self::Error> {
            // cg::thread_block cta = cg::this_thread_block();
            // __shared__ float tile[TILE_DIM][TILE_DIM];
            //
            // int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
            // int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
            // int index_in = xIndex + (yIndex)*width;
            let x_index = (tid.block_idx.x * TILE_DIM + tid.thread_idx.x) as usize;
            let y_index = (tid.block_idx.y * TILE_DIM + tid.thread_idx.y) as usize;
            let index_in = x_index + y_index * self.cols;

            // xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
            // yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
            // int index_out = xIndex + (yIndex)*height;
            let x_index = (tid.block_idx.y * TILE_DIM + tid.thread_idx.x) as usize;
            let y_index = (tid.block_idx.x * TILE_DIM + tid.thread_idx.y) as usize;
            let index_out = x_index + y_index * self.rows;

            // for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
            //   tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
            // }

            // load into shared memory
            let mut i = 0;
            while i < TILE_DIM {
                let dev_mat = self.dev_mat.lock().await;
                let mut tiles = self.shared_mem_tiles.lock().await;

                let tile_idx = (tid.thread_idx.y + i) * TILE_DIM + tid.thread_idx.x;
                let mat_idx = index_in + i as usize * self.cols;
                tiles[(tid, tile_idx as usize)] = dev_mat[(tid, mat_idx)];
                i += BLOCK_ROWS;
            }

            // cg::sync(cta);
            block.synchronize_threads().await;

            let mut i = 0;
            while i < TILE_DIM {
                let mut dev_result = self.dev_result.lock().await;
                let tiles = self.shared_mem_tiles.lock().await;
                let tile_idx = tid.thread_idx.x * TILE_DIM + tid.thread_idx.y + i;

                let result_idx = index_out + i as usize * self.rows;
                dev_result[(tid, result_idx)] = tiles[(tid, tile_idx as usize)];
                i += BLOCK_ROWS;
            }
            Ok(())
        }

        fn name(&self) -> Option<&str> {
            Some("transpose_coalesced")
        }
    }
}

pub mod optimized {
    use super::{BLOCK_ROWS, TILE_DIM};
    use gpucachesim::exec::{DevicePtr, Kernel, ThreadBlock, ThreadIndex};
    use tokio::sync::Mutex;

    #[derive(Debug)]
    pub struct Transpose<'a, T> {
        pub dev_mat: Mutex<DevicePtr<&'a Vec<T>>>,
        pub dev_result: Mutex<DevicePtr<&'a mut Vec<T>>>,
        pub rows: usize,
        pub cols: usize,

        /// Shared memory array used to store the tiles
        pub shared_mem_tiles: Mutex<DevicePtr<Vec<T>>>,
    }

    #[async_trait::async_trait]
    impl<'a, T> Kernel for Transpose<'a, T>
    where
        T: num_traits::Float + std::ops::AddAssign + Send + Sync + std::fmt::Debug,
    {
        type Error = std::convert::Infallible;

        #[gpucachesim::exec::inject_reconvergence_points]
        async fn run(&self, block: &ThreadBlock, tid: &ThreadIndex) -> Result<(), Self::Error> {
            // cg::thread_block cta = cg::this_thread_block();
            // __shared__ float tile[TILE_DIM][TILE_DIM + 1];
            //
            // int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
            // int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
            // int index_in = xIndex + (yIndex)*width;
            let x_index = (tid.block_idx.x * TILE_DIM + tid.thread_idx.x) as usize;
            let y_index = (tid.block_idx.y * TILE_DIM + tid.thread_idx.y) as usize;
            let index_in = x_index + y_index * self.cols;

            // xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
            // yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
            // int index_out = xIndex + (yIndex)*height;
            let x_index = (tid.block_idx.y * TILE_DIM + tid.thread_idx.x) as usize;
            let y_index = (tid.block_idx.x * TILE_DIM + tid.thread_idx.y) as usize;
            let index_out = x_index + y_index * self.rows;

            // for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
            //   tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
            // }

            // load into shared memory
            let mut i = 0;
            while i < TILE_DIM {
                let dev_mat = self.dev_mat.lock().await;
                let mut tiles = self.shared_mem_tiles.lock().await;

                let tile_idx = (tid.thread_idx.y + i) * TILE_DIM + tid.thread_idx.x;
                let mat_idx = index_in + i as usize * self.cols;
                tiles[(tid, tile_idx as usize)] = dev_mat[(tid, mat_idx)];
                i += BLOCK_ROWS;
            }

            // cg::sync(cta);
            block.synchronize_threads().await;

            let mut i = 0;
            while i < TILE_DIM {
                let mut dev_result = self.dev_result.lock().await;
                let tiles = self.shared_mem_tiles.lock().await;

                let tile_idx = tid.thread_idx.x * TILE_DIM + tid.thread_idx.y + i;
                let result_idx = index_out + i as usize * self.rows;
                dev_result[(tid, result_idx)] = tiles[(tid, tile_idx as usize)];
                i += BLOCK_ROWS;
            }
            Ok(())
        }

        fn name(&self) -> Option<&str> {
            Some("transpose_optimized")
        }
    }
}

pub fn reference<T>(mat: &[T], result: &mut [T], rows: usize, cols: usize)
where
    T: Float + std::ops::AddAssign,
{
    assert_eq!(mat.len(), result.len());
    for y in 0..rows {
        for x in 0..cols {
            result[(x * rows) + y] = mat[(y * cols) + x];
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Variant {
    Naive,
    Coalesced,
    Optimized,
}

pub async fn benchmark<T>(dim: usize, variant: Variant) -> super::Result
where
    T: Float + Zero + std::ops::AddAssign + Send + Sync + std::fmt::Debug,
{
    // create host vectors
    let mut mat: Vec<T> = vec![T::zero(); dim * dim];
    let mut result: Vec<T> = vec![T::zero(); dim * dim];

    // initialize vectors
    for (i, v) in mat.iter_mut().enumerate() {
        *v = NumCast::from(i).unwrap();
    }

    match variant {
        Variant::Naive => transpose_naive(&mat, &mut result, dim, dim).await,
        Variant::Coalesced => transpose_coalesced(&mat, &mut result, dim, dim).await,
        Variant::Optimized => transpose_optimized(&mat, &mut result, dim, dim).await,
    }
}

pub fn validate_arguments<T>(
    mat: &Vec<T>,
    result: &mut Vec<T>,
    rows: usize,
    cols: usize,
) -> eyre::Result<()> {
    if mat.len() != result.len() {
        eyre::bail!(
            "input and result matrix must have the same size: got {} and {}",
            mat.len(),
            result.len()
        );
    }
    if rows != cols {
        eyre::bail!("non-square matrices are not supported: got {rows}x{cols}");
    }

    if cols % TILE_DIM as usize != 0 || rows % TILE_DIM as usize != 0 {
        eyre::bail!("matrix size must be integral multiple of tile size {TILE_DIM}");
    }
    Ok(())
}

#[allow(clippy::module_name_repetitions)]
pub async fn transpose_naive<T>(
    mat: &Vec<T>,
    result: &mut Vec<T>,
    rows: usize,
    cols: usize,
) -> super::Result
where
    T: Float + Zero + std::ops::AddAssign + Send + Sync + std::fmt::Debug,
{
    validate_arguments(mat, result, rows, cols)?;

    let tracer = Tracer::new();

    // allocate memory for each vector on simulated GPU device
    let dev_mat = tracer
        .allocate(mat, MemorySpace::Global, Some("matrix"))
        .await;
    let dev_result = tracer
        .allocate(result, MemorySpace::Global, Some("result"))
        .await;

    let kernel: naive::Transpose<T> = naive::Transpose {
        dev_mat: Mutex::new(dev_mat),
        dev_result: Mutex::new(dev_result),
        rows,
        cols,
    };
    transpose::<T, naive::Transpose<T>>(tracer, rows, cols, kernel).await
}

#[allow(clippy::module_name_repetitions)]
pub async fn transpose_coalesced<T>(
    mat: &Vec<T>,
    result: &mut Vec<T>,
    rows: usize,
    cols: usize,
) -> super::Result
where
    T: Float + Zero + std::ops::AddAssign + Send + Sync + std::fmt::Debug,
{
    validate_arguments(mat, result, rows, cols)?;

    let tracer = Tracer::new();

    // allocate memory for each vector on simulated GPU device
    let dev_mat = tracer
        .allocate(mat, MemorySpace::Global, Some("matrix"))
        .await;
    let dev_result = tracer
        .allocate(result, MemorySpace::Global, Some("result"))
        .await;

    // shared memory
    let shared_mem_tiles = vec![T::zero(); (TILE_DIM * TILE_DIM) as usize];
    let shared_mem_tiles = tracer
        .allocate(
            shared_mem_tiles,
            MemorySpace::Shared,
            Some("shared_mem_tiles"),
        )
        .await;

    let kernel: coalesced::Transpose<T> = coalesced::Transpose {
        dev_mat: Mutex::new(dev_mat),
        dev_result: Mutex::new(dev_result),
        shared_mem_tiles: Mutex::new(shared_mem_tiles),
        rows,
        cols,
    };
    transpose::<T, coalesced::Transpose<T>>(tracer, rows, cols, kernel).await
}

#[allow(clippy::module_name_repetitions)]
pub async fn transpose_optimized<T>(
    mat: &Vec<T>,
    result: &mut Vec<T>,
    rows: usize,
    cols: usize,
) -> super::Result
where
    T: Float + Zero + std::ops::AddAssign + Send + Sync + std::fmt::Debug,
{
    validate_arguments(mat, result, rows, cols)?;

    let tracer = Tracer::new();

    // allocate memory for each vector on simulated GPU device
    let dev_mat = tracer
        .allocate(mat, MemorySpace::Global, Some("matrix"))
        .await;
    let dev_result = tracer
        .allocate(result, MemorySpace::Global, Some("result"))
        .await;

    // shared memory
    let shared_mem_tiles = vec![T::zero(); (TILE_DIM * (TILE_DIM + 1)) as usize];
    let shared_mem_tiles = tracer
        .allocate(
            shared_mem_tiles,
            MemorySpace::Shared,
            Some("shared_mem_tiles"),
        )
        .await;

    let kernel: optimized::Transpose<T> = optimized::Transpose {
        dev_mat: Mutex::new(dev_mat),
        dev_result: Mutex::new(dev_result),
        shared_mem_tiles: Mutex::new(shared_mem_tiles),
        rows,
        cols,
    };
    transpose::<T, optimized::Transpose<T>>(tracer, rows, cols, kernel).await
}

pub async fn transpose<T, K>(
    tracer: std::sync::Arc<Tracer>,
    rows: usize,
    cols: usize,
    kernel: K,
) -> super::Result
where
    T: Float + Zero + std::ops::AddAssign + Send + Sync + std::fmt::Debug,
    K: Kernel + Send + Sync,
    <K as Kernel>::Error: Send + Sync + 'static,
{
    let block_dim: Dim = (TILE_DIM, BLOCK_ROWS).into();
    let grid_x = cols / TILE_DIM as usize;
    let grid_y = rows / TILE_DIM as usize;
    let grid_dim: Dim = (grid_x as u32, grid_y as u32).into();
    println!("grid dim:  {grid_dim}");
    println!("block dim: {block_dim}");

    assert!(grid_dim.x > 0);
    assert!(grid_dim.y > 0);
    assert!(grid_dim.z > 0);

    let trace = tracer.trace_kernel(grid_dim, block_dim, kernel).await?;
    Ok((tracer.commands().await, vec![trace]))
}

#[cfg(test)]
mod tests {
    use color_eyre::eyre;
    use gpucachesim::exec::tracegen::fmt;
    use ndarray::Array2;
    use utils::diff;

    const EPSILON: f32 = 0.0001;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_correctness() -> eyre::Result<()> {
        let dim = 32;

        // create host vectors
        let mut mat: Vec<f32> = vec![0.0; dim * dim];
        let mut result: Vec<f32> = vec![0.0; dim * dim];
        let mut ref_result: Vec<f32> = vec![0.0; dim * dim];

        // initialize random matrix
        for (i, v) in mat.iter_mut().enumerate() {
            *v = i as f32;
        }

        let ndarray_result = {
            let ref_mat = Array2::from_shape_vec((dim, dim), mat.clone())?;
            ref_mat.reversed_axes()
        };
        let (_commands, kernel_traces) =
            super::transpose_optimized(&mat, &mut result, dim, dim).await?;
        assert_eq!(kernel_traces.len(), 1);
        let (_launch_config, trace) = kernel_traces.into_iter().next().unwrap();
        super::reference(&mat, &mut ref_result, dim, dim);

        let ref_result = Array2::from_shape_vec((dim, dim), ref_result)?;
        let result = Array2::from_shape_vec((dim, dim), result)?;
        dbg!(&ndarray_result);
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
        // diff::assert_eq!(
        //     have: simplified_trace,
        //     want: [
        //         ("LDG", 0, "11111111111111111111000000000000", 0),
        //     ].into_iter().enumerate().map(SimplifiedTraceInstruction::from).collect::<Vec<_>>()
        // );
        // assert!(false);
        Ok(())
    }
}
