#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
// #![allow(warnings)]

use color_eyre::eyre;
use gpucachesim::exec;
use num_traits::{Float, NumCast, Zero};
use rand::distributions::{self, Distribution};

#[derive(Debug)]
// struct VecAdd<'s, 'a, T> {
struct Matrixmul<'s, T> {
    phantom: std::marker::PhantomData<&'s T>,
    // dev_a: &'a exec::DevicePtr<'s, 'a, Vec<T>>,
    // dev_a: exec::DevicePtr<'s, Vec<T>>,
    // dev_b: &'a exec::DevicePtr<'s, 'a, Vec<T>>,
    // dev_b: exec::DevicePtr<'s, Vec<T>>,
    // dev_result: &'a mut exec::DevicePtr<'s, 'a, Vec<T>>,
    // dev_result: exec::DevicePtr<'s, Vec<T>>,
    // n: usize,
}

// impl<'s, 'a, T> exec::Kernel for VecAdd<'s, 'a, T>
impl<'s, T> exec::Kernel for Matrixmul<'s, T>
where
    T: Float + std::fmt::Debug,
{
    type Error = std::convert::Infallible;

    fn run(&mut self, idx: &exec::ThreadIndex) -> Result<(), Self::Error> {
        // Get our global thread ID
        // int id = blockIdx.x * blockDim.x + threadIdx.x;
        // let id: usize = (idx.block_idx.x * idx.block_dim.x + idx.thread_idx.x) as usize;

        // Make sure we do not go out of bounds
        // if (id < n) c[id] = a[id] + b[id];
        // let active = id < self.n;
        // self.dev_result[(id, active)] = self.dev_a[(id, active)] + self.dev_b[(id, active)];

        // if id < self.n {
        //     self.dev_result[id] = self.dev_a[id] + self.dev_b[id];
        // } else {
        //     // self.dev_result[id] = self.dev_a[id] + self.dev_b[id];
        // }
        Ok(())
    }

    fn name(&self) -> &str {
        "matrixmul"
    }
}

pub fn matrixmul<T>(a: &Vec<T>, b: &Vec<T>, result: &mut Vec<T>) -> eyre::Result<()>
where
    T: Float + Zero + NumCast + std::iter::Sum + std::fmt::Display + std::fmt::Debug,
{
    Ok(())
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

pub fn default_matrixmul<T>(num_rows: usize) -> eyre::Result<()>
where
    T: Float + Zero + NumCast + std::iter::Sum + std::fmt::Display + std::fmt::Debug,
    distributions::Open01: Distribution<T>,
{
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let matrix_size = num_rows * num_rows;
    // create host vectors
    let mut a: Vec<T> = vec![T::zero(); matrix_size];
    let mut b: Vec<T> = vec![T::zero(); matrix_size];
    // let mut c: Vec<T> = vec![T::zero(); matrix_size];
    let mut result: Vec<T> = vec![T::zero(); matrix_size];

    // initialize vectors
    for i in 0..matrix_size {
        a[i] = T::one() + rng.sample(distributions::Open01);
        b[i] = T::one() + rng.sample(distributions::Open01);
        // a[i] = T::one() + rng.gen::<T>();
        // b[i] = T::one() + rng.gen::<T>();
        // A[i] = ((T)rand() / (RAND_MAX)) + 1;
        // B[i] = (i%MCOL)+1;
        // B[i] = ((T)rand() / (RAND_MAX)) + 1;
        // C[i] = 0;
        // D[i] = 0;
    }

    // for i in 0..n {
    //     let angle = T::from(i).unwrap();
    //     a[i] = angle.sin() * angle.sin();
    //     b[i] = angle.cos() * angle.cos();
    //     result[i] = T::zero();
    // }
    //
    // vectoradd(&a, &b, &mut result)?;
    Ok(())
}

// Number of threads in each thread block
pub const BLOCK_SIZE: u32 = 32;

#[cfg(test)]
mod tests {
    use color_eyre::eyre;
    use gpucachesim::exec::{MemorySpace, TraceGenerator, Tracer};
    use rand::Rng;
    use utils::diff;

    #[test]
    pub fn test_correctness() -> eyre::Result<()> {
        let mut rng = rand::thread_rng();

        let size = 4;
        let matrix_size = size * size;

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

        // super::vectoradd(&mut a, &mut b, &mut result)?;
        super::reference_matrixmul(&mut a, &mut b, &mut ref_result, size);
        dbg!(&ref_result);

        // diff::diff!(have: result, want: ref_result);

        assert!(false);
        Ok(())
    }
}
