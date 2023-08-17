use crate::exec;
use color_eyre::eyre;
use num_traits::{Float, Zero};

#[test]
fn vectoradd() -> eyre::Result<()> {
    // let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
    // let trace_dir = manifest_dir.join($path);
    // run_lockstep(&trace_dir, TraceProvider::Native)
    Ok(())
}

#[test]
fn matrixmul() -> eyre::Result<()> {
    fn mult_cpu<T>(a: &[T], b: &[T], c: &mut [T], m: usize, n: usize, p: usize)
    where
        T: Float + Zero + std::ops::AddAssign,
    {
        for mi in 0..m {
            for pi in 0..p {
                let mut sum = T::zero();
                for ni in 0..n {
                    sum += a[mi * n + ni] * b[ni * p + pi];
                }
                c[mi * p + pi] = sum;
            }
        }
    }

    // // 2D block and 2D thread
    // // Each thread computes one cell in mat_3.
    // // the grid + thradidx
    // size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    // size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    //
    // printf("thread idx = (%u, %u, %u)\n", threadIdx.x, threadIdx.y, threadIdx.z);
    //
    // // do not process outside the matrix.
    // // do not forget the equal sign!
    // if ((i >= m) || (j >= p)) {
    //   return;
    // }
    //
    // float acc_sum{0};
    // for (size_t k = 0; k < n; k++) {
    //   acc_sum += mat_a[i * n + k] * mat_b[k * p + j];
    // }
    // mat_c[i * p + j] = acc_sum;

    #[derive(Debug)]
    struct MatrixMul<'s, 'a, T> {
        d_a: &'a mut exec::DevicePtr<'s, 'a, Vec<T>>,
        d_b: &'a mut exec::DevicePtr<'s, 'a, Vec<T>>,
        d_c: &'a mut exec::DevicePtr<'s, 'a, Vec<T>>,
        n: usize,
    }

    impl<'s, 'a, T> exec::Kernel for MatrixMul<'s, 'a, T>
    where
        T: Float + std::fmt::Debug,
    {
        type Error = std::convert::Infallible;

        fn run(&mut self, idx: &exec::ThreadIndex) -> Result<(), Self::Error> {
            // Get our global thread ID
            // int id = blockIdx.x * blockDim.x + threadIdx.x;
            let id: usize = (idx.block_idx.x * idx.block_dim.x + idx.thread_idx.x) as usize;

            // Make sure we do not go out of bounds
            // if (id < n) c[id] = a[id] + b[id];
            // let test2: &(dyn std::ops::IndexMut<usize, Output = T>) = self.d_a;
            if id < self.n {
                self.d_c[id] = self.d_a[id] + self.d_b[id];
            }
            Ok(())
        }
    }

    // Number of threads in each thread block
    // const BLOCK_SIZE: u32 = 1024;

    // fn vectoradd<T>(n: usize) -> eyre::Result<()>
    // where
    //     T: Float + Zero + NumCast + std::iter::Sum + std::fmt::Display + std::fmt::Debug,
    // {
    // create host vectors
    // let mut a: Vec<T> = vec![T::zero(); n];
    // let mut b: Vec<T> = vec![T::zero(); n];
    // let mut c: Vec<T> = vec![T::zero(); n];
    //
    // // initialize vectors
    // for i in 0..n {
    //     let angle = T::from(i).unwrap();
    //     a[i] = angle.sin() * angle.sin();
    //     b[i] = angle.cos() * angle.cos();
    //     c[i] = T::zero();
    // }
    //
    // let sim = exec::Simulation::new();
    //
    // // allocate memory for each vector on simulated GPU device
    // let a_size = a.len() * std::mem::size_of::<T>();
    // let b_size = b.len() * std::mem::size_of::<T>();
    // let c_size = c.len() * std::mem::size_of::<T>();
    // let mut d_a = sim.allocate(&mut a, a_size as u64);
    // let mut d_b = sim.allocate(&mut b, b_size as u64);
    // let mut d_c = sim.allocate(&mut c, c_size as u64);
    //
    // // number of thread blocks in grid
    // let grid_size = (n as f64 / <f64 as From<_>>::from(BLOCK_SIZE)).ceil() as u32;
    //
    // let kernel: MatrixMul<T> = MatrixMul {
    //     d_a: &mut d_a,
    //     d_b: &mut d_b,
    //     d_c: &mut d_c,
    //     n,
    // };
    // sim.launch_kernel(grid_size, BLOCK_SIZE, kernel)?;
    //
    // // sum up vector c and print result divided by n.
    // // this should equal 1 within
    // let total_sum: T = c.into_iter().sum();
    // println!(
    //     "Final sum = {total_sum}; sum/n = {:.2} (should be ~1)\n",
    //     total_sum / T::from(n).unwrap()
    // );
    //
    // dbg!(&sim.stats.lock().unwrap());
    Ok(())
    // }
    // Ok(())
}
