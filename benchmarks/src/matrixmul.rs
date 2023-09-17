#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
// #![allow(warnings)]

use color_eyre::eyre;
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
use std::sync::Arc;
use tokio::sync::Mutex;

// Number of threads in each thread block
pub const BLOCK_SIZE: usize = 4;
// pub const BLOCK_SIZE: u32 = 32;

#[derive(Debug)]
struct Matrixmul<'a, T> {
    dev_a: Mutex<DevicePtr<&'a Vec<T>>>,
    dev_b: Mutex<DevicePtr<&'a Vec<T>>>,
    dev_result: Mutex<DevicePtr<&'a mut Vec<T>>>,
    num_rows: usize,
    /// Shared memory array used to store the sub-matrix of A
    shared_mem_a: Mutex<DevicePtr<Vec<T>>>,
    /// Shared memory array used to store the sub-matrix of B
    shared_mem_b: Mutex<DevicePtr<Vec<T>>>,
    // __shared__ T As[BLOCK_SIZE][BLOCK_SIZE];

    // shared memory array used to store the sub-matrix of B

    // dev_a: &'a exec::DevicePtr<'s, 'a, Vec<T>>,
    // dev_a: exec::DevicePtr<'s, Vec<T>>,
    // dev_b: &'a exec::DevicePtr<'s, 'a, Vec<T>>,
    // dev_b: exec::DevicePtr<'s, Vec<T>>,
    // dev_result: &'a mut exec::DevicePtr<'s, 'a, Vec<T>>,
    // dev_result: exec::DevicePtr<'s, Vec<T>>,
    // n: usize,
}

#[async_trait::async_trait]
impl<'a, T> Kernel for Matrixmul<'a, T>
where
    T: Float + std::ops::AddAssign + Send + Sync + std::fmt::Debug,
{
    type Error = std::convert::Infallible;

    #[gpucachesim::exec::inject_reconvergence_points]
    async fn run(&self, block: &ThreadBlock, tid: &ThreadIndex) -> Result<(), Self::Error> {
        dbg!(&tid);
        let bx = tid.block_idx.x as usize;
        let by = tid.block_idx.y as usize;

        let tx = tid.thread_idx.x as usize;
        let ty = tid.thread_idx.y as usize;

        // index of the first sub-matrix of A processed by the block
        let a_begin = self.num_rows * BLOCK_SIZE * by;

        // index of the last sub-matrix of A processed by the block
        let a_end = a_begin + self.num_rows - 1;

        // step size used to iterate through the sub-matrices of A
        let a_step = BLOCK_SIZE;

        // index of the first sub-matrix of B processed by the block
        let b_begin = BLOCK_SIZE * bx;

        // Step size used to iterate through the sub-matrices of B
        let b_step = BLOCK_SIZE * self.num_rows;

        // Csub is used to store the element of the block sub-matrix
        // that is computed by the thread
        // float Csub[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        // float Csub[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        // float Csub[4] = {0, 0, 0, 0};
        // float Csub[2] = {0, 0};
        let mut c_sub = T::zero();

        // Loop over all the sub-matrices of A and B
        // required to compute the block sub-matrix

        let mut ai = a_begin;
        let mut bi = b_begin;
        while ai <= a_end {
            // shared memory array used to store the sub-matrix of A
            // __shared__ T As[BLOCK_SIZE][BLOCK_SIZE];

            // shared memory array used to store the sub-matrix of B
            // __shared__ T Bs[BLOCK_SIZE][BLOCK_SIZE];

            // dbg!(&ty, &tx);
            {
                // load the matrices from device memory to shared memory
                // each thread loads one element of each matrix

                // As[ty][tx] = A[a + wA * ty + tx];
                let a = self.dev_a.lock().await;
                let mut shared_a = self.shared_mem_a.lock().await;
                shared_a[(tid, ty * BLOCK_SIZE + tx)] = a[(tid, ai + self.num_rows * ty + tx)];

                // Bs[ty][tx] = B[b + wB * ty + tx];
                let b = self.dev_b.lock().await;
                let mut shared_b = self.shared_mem_b.lock().await;
                shared_b[(tid, ty * BLOCK_SIZE + tx)] = b[(tid, bi + self.num_rows * ty + tx)];
            }

            block.synchronize_threads().await;

            // make sure shared mem has been loaded
            // {
            //     let shared_a = self.shared_mem_a.lock().await;
            //     dbg!(&shared_a);
            //     assert!(shared_a.inner.iter().all(|x| *x != T::zero()));
            //
            //     let shared_b = self.shared_mem_b.lock().await;
            //     dbg!(&shared_b);
            //     assert!(shared_b.inner.iter().all(|x| *x != T::zero()));
            // }

            for k in 0..BLOCK_SIZE {
                let shared_a = self.shared_mem_a.lock().await;
                let shared_b = self.shared_mem_b.lock().await;
                c_sub +=
                    shared_a[(tid, ty * BLOCK_SIZE + k)] * shared_b[(tid, k * BLOCK_SIZE + tx)];
            }

            block.synchronize_threads().await;

            ai += a_step;
            bi += b_step;
        }

        // Write the block sub-matrix to device memory;
        // each thread writes one element
        let c = self.num_rows * BLOCK_SIZE * by + BLOCK_SIZE * bx;
        let mut result = self.dev_result.lock().await;
        result[(tid, c + self.num_rows * ty + tx)] = c_sub;

        Ok(())
    }

    fn name(&self) -> Option<&str> {
        Some("matrixmul")
    }
}

pub fn reference_matrixmul<T>(a: &Vec<T>, b: &Vec<T>, result: &mut Vec<T>, size: usize)
where
    T: Float + std::ops::AddAssign,
{
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), result.len());
    assert_eq!(a.len(), size * size);
    for i in 0..size {
        for j in 0..size {
            let mut sum = T::zero();
            for k in 0..size {
                sum += a[i * size + k] * b[k * size + j];
            }

            result[i * size + j] = sum;
        }
    }
}

pub async fn default_matrixmul<T>(num_rows: usize) -> eyre::Result<()>
where
    // T: Float + Zero + NumCast + std::iter::Sum + std::fmt::Display + std::fmt::Debug,
    T: Float + Zero + std::ops::AddAssign + Send + Sync + std::fmt::Debug,
    distributions::Open01: Distribution<T>,
{
    let mut rng = rand::thread_rng();

    let matrix_size = num_rows * num_rows;
    // create host vectors
    let mut a: Vec<T> = vec![T::zero(); matrix_size];
    let mut b: Vec<T> = vec![T::zero(); matrix_size];
    let mut result: Vec<T> = vec![T::zero(); matrix_size];

    // initialize vectors
    for i in 0..matrix_size {
        a[i] = T::one() + rng.sample(distributions::Open01);
        b[i] = T::one() + rng.sample(distributions::Open01);
    }

    matrixmul(&a, &b, &mut result, num_rows).await?;
    Ok(())
}

pub async fn matrixmul<T>(
    a: &Vec<T>,
    b: &Vec<T>,
    result: &mut Vec<T>,
    num_rows: usize,
) -> eyre::Result<()>
where
    T: Float + Zero + std::ops::AddAssign + Send + Sync + std::fmt::Debug,
{
    let tracer = Tracer::new();

    // let sim = exec::Simulation::new();
    assert_eq!(a.len(), b.len());
    assert_eq!(b.len(), result.len());
    let n = a.len();

    // allocate memory for each vector on simulated GPU device
    let dev_a = tracer.allocate(a, MemorySpace::Global).await;
    let dev_b = tracer.allocate(b, MemorySpace::Global).await;
    let dev_result = tracer.allocate(result, MemorySpace::Global).await;

    // shared memory
    // let mut shared_mem_a
    let shared_mem_a = vec![T::zero(); BLOCK_SIZE * BLOCK_SIZE];
    let shared_mem_a = tracer.allocate(shared_mem_a, MemorySpace::Shared).await;

    let shared_mem_b = vec![T::zero(); BLOCK_SIZE * BLOCK_SIZE];
    let shared_mem_b = tracer.allocate(shared_mem_b, MemorySpace::Shared).await;

    // number of thread blocks in grid
    // let grid_size = (n as f64 / BLOCK_SIZE as f64).ceil() as u32;

    let block_dim: Dim = (BLOCK_SIZE as u32, (BLOCK_SIZE / 1) as u32).into();
    let grid_size = (num_rows + (BLOCK_SIZE - 1)) / BLOCK_SIZE;
    let grid_dim: Dim = (grid_size as u32, grid_size as u32).into();
    println!("grid dim:  {}", grid_dim);
    println!("block dim: {}", block_dim);

    assert!(grid_dim.x > 0);
    assert!(grid_dim.y > 0);
    assert!(grid_dim.z > 0);

    let kernel: Matrixmul<T> = Matrixmul {
        // phantom: std::marker::PhantomData,
        dev_a: Mutex::new(dev_a),
        dev_b: Mutex::new(dev_b),
        dev_result: Mutex::new(dev_result),
        shared_mem_a: Mutex::new(shared_mem_a),
        shared_mem_b: Mutex::new(shared_mem_b),
        num_rows,
    };
    tracer.trace_kernel(grid_dim, block_dim, kernel).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use approx::AbsDiffEq;
    use color_eyre::eyre;
    use gpucachesim::exec::tracegen::{TraceGenerator, Tracer};
    use gpucachesim::exec::MemorySpace;
    use ndarray::prelude::*;
    use ndarray::{linalg::Dot, Array2};
    use rand::Rng;
    use utils::diff;

    const EPSILON: f32 = 0.0001;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_correctness() -> eyre::Result<()> {
        let mut rng = rand::thread_rng();

        let size = 4;
        let matrix_size = size * size;
        let matrix_shape = (size, size);

        // create host vectors
        let mut a: Vec<f32> = vec![0.0; matrix_size];
        let mut b: Vec<f32> = vec![0.0; matrix_size];
        let mut result: Vec<f32> = vec![0.0; matrix_size];
        let mut ref_result: Vec<f32> = vec![0.0; matrix_size];

        // initialize random matrix
        for i in 0..matrix_size {
            a[i] = 1.0 + rng.gen::<f32>();
            b[i] = 1.0 + rng.gen::<f32>();
        }

        let ndarray_result = {
            let ref_a: Array2<f32> = Array2::from_shape_vec(matrix_shape, a.clone())?;
            let ref_b: Array2<f32> = Array2::from_shape_vec(matrix_shape, b.clone())?;
            ref_a.dot(&ref_b)
        };
        let _ = super::matrixmul(&a, &b, &mut result, size).await;
        super::reference_matrixmul(&a, &b, &mut ref_result, size);

        let ref_result = Array2::from_shape_vec(matrix_shape, ref_result)?;
        let result = Array2::from_shape_vec(matrix_shape, result)?;

        if !approx::abs_diff_eq!(ref_result, ndarray_result, epsilon = EPSILON) {
            diff::diff!(have: ref_result, want: ndarray_result);
        }
        if !approx::abs_diff_eq!(result, ref_result, epsilon = EPSILON) {
            diff::diff!(have: ref_result, want: ndarray_result);
        }

        assert!(false);
        Ok(())
    }
}
