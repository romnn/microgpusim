#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use color_eyre::eyre;
use gpucachesim::exec::r#async::{DevicePtr, Kernel, ThreadBlock, TraceGenerator, Tracer};
use gpucachesim::exec::{MemorySpace, ThreadIndex};
use num_traits::{Float, NumCast, Zero};
use std::sync::Arc;
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

    async fn run(&self, block: &ThreadBlock, tid: &ThreadIndex) -> Result<(), Self::Error> {
        let idx = (tid.block_idx.x * tid.block_dim.x + tid.thread_idx.x) as usize;

        let dev_a = self.dev_a.lock().await;
        let dev_b = self.dev_b.lock().await;
        let mut dev_result = self.dev_result.lock().await;

        if idx < self.n {
            dev_result[(tid, idx)] = dev_a[(tid, idx)] + dev_b[(tid, idx)];
        } else {
            // dev_result[tid] = dev_a[tid] + dev_b[tid];
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "VecAdd"
    }
}

#[deprecated]
mod deprecated {
    use gpucachesim::exec::{self, DevicePtr, Kernel};
    use num_traits::{Float, NumCast, Zero};

    // #[derive(Debug)]
    // struct VecAdd<'s, T> {
    //     dev_a: DevicePtr<'s, Vec<T>, T>,
    //     dev_b: DevicePtr<'s, Vec<T>, T>,
    //     dev_result: DevicePtr<'s, Vec<T>, T>,
    //     n: usize,
    // }
    //
    // impl<'s, T> Kernel for VecAdd<'s, T>
    // where
    //     T: Float + std::fmt::Debug,
    // {
    //     type Error = std::convert::Infallible;
    //
    //     fn run(&mut self, idx: &exec::ThreadIndex) -> Result<(), Self::Error> {
    //         // compute global thread index
    //         let id: usize = (idx.block_idx.x * idx.block_dim.x + idx.thread_idx.x) as usize;
    //
    //         if id < self.n {
    //             self.dev_result[()] = self.dev_a[()] + self.dev_b[()];
    //             // self.dev_result[id] = self.dev_a[id] + self.dev_b[id];
    //         } else {
    //             self.dev_result[()] = self.dev_a[()] + self.dev_b[()];
    //             // self.dev_result[id] = self.dev_a[id] + self.dev_b[id];
    //         }
    //         Ok(())
    //     }
    //
    //     fn name(&self) -> &str {
    //         "VecAdd"
    //     }
    // }
}

// Number of threads in each thread block
pub const BLOCK_SIZE: u32 = 1024;

pub fn reference_vectoradd<T>(a: &Vec<T>, b: &Vec<T>, result: &mut Vec<T>)
where
    T: Float,
{
    for (i, sum) in result.iter_mut().enumerate() {
        *sum = a[i] + b[i];
    }
}

pub async fn default_vectoradd<T>(n: usize) -> eyre::Result<()>
where
    // T: Float + Zero + NumCast + std::iter::Sum + std::fmt::Display + std::fmt::Debug,
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

    vectoradd(&a, &b, &mut result).await?;
    Ok(())
}

pub async fn vectoradd<T>(a: &Vec<T>, b: &Vec<T>, result: &mut Vec<T>) -> eyre::Result<()>
where
    T: Float + Zero + Send + Sync,
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

    // number of thread blocks in grid
    let grid_size = (n as f64 / <f64 as From<_>>::from(BLOCK_SIZE)).ceil() as u32;
    let kernel: VecAdd<T> = VecAdd {
        dev_a: Mutex::new(dev_a),
        dev_b: Mutex::new(dev_b),
        dev_result: Mutex::new(dev_result),
        n,
    };
    tracer.trace_kernel(grid_size, BLOCK_SIZE, kernel).await?;
    // *result = tracer.kernel.dev_result.into_inner();

    // let kernel: VecAdd<T> = VecAdd {
    //     d_a: &mut d_a,
    //     d_b: &mut d_b,
    //     d_c: &mut d_c,
    //     n,
    // };
    // sim.launch_kernel(grid_size, BLOCK_SIZE, kernel)?;
    // let stats = sim.run_to_completion()?;
    //
    // // sum up vector c and print result divided by n.
    // // this should equal 1 within
    // let total_sum: T = c.iter().copied().sum();
    // println!(
    //     "Final sum = {total_sum}; sum/n = {:.2} (should be ~1)\n",
    //     total_sum / T::from(n).unwrap()
    // );
    //
    // eprintln!("STATS:\n");
    // eprintln!("DRAM: total reads: {}", &stats.dram.total_reads());
    // eprintln!("DRAM: total writes: {}", &stats.dram.total_writes());
    // eprintln!("SIM: {:#?}", &stats.sim);
    // eprintln!("INSTRUCTIONS: {:#?}", &stats.instructions);
    // eprintln!("ACCESSES: {:#?}", &stats.accesses);
    // eprintln!("L1I: {:#?}", &stats.l1i_stats.reduce());
    // eprintln!("L1D: {:#?}", &stats.l1d_stats.reduce());
    // eprintln!("L2D: {:#?}", &stats.l2d_stats.reduce());
    // eprintln!("completed in {:?}", start.elapsed());
    Ok(())
}

#[cfg(test)]
mod tests {
    use color_eyre::eyre;
    use gpucachesim::exec::r#async::{TraceGenerator, Tracer};
    use gpucachesim::exec::MemorySpace;
    use tokio::sync::Mutex;
    use utils::diff;

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

        super::vectoradd(&a, &b, &mut result).await?;
        super::reference_vectoradd(&a, &b, &mut ref_result);

        diff::assert_eq!(have: result, want: ref_result);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn generate_trace() -> eyre::Result<()> {
        let n = 100;
        let tracer = Tracer::new();

        let mut a: Vec<f32> = vec![0.0; n];
        let mut b: Vec<f32> = vec![0.0; n];
        let mut result: Vec<f32> = vec![0.0; n];

        // initialize vectors
        for i in 0..n {
            let angle = i as f32;
            a[i] = angle.sin() * angle.sin();
            b[i] = angle.cos() * angle.cos();
        }

        // allocate memory for each vector on simulated GPU device
        let dev_a = tracer.allocate(&a, MemorySpace::Global).await;
        let dev_b = tracer.allocate(&b, MemorySpace::Global).await;
        let dev_result = tracer.allocate(&mut result, MemorySpace::Global).await;

        // number of thread blocks in grid
        let grid_size = (n as f64 / <f64 as From<_>>::from(super::BLOCK_SIZE)).ceil() as u32;

        let kernel: super::VecAdd<f32> = super::VecAdd {
            dev_a: Mutex::new(dev_a),
            dev_b: Mutex::new(dev_b),
            dev_result: Mutex::new(dev_result),
            n,
        };
        tracer
            .trace_kernel(grid_size, super::BLOCK_SIZE, kernel)
            .await?;
        // let stats = sim.run_to_completion()?;

        // sum up vector c and print result divided by n.
        // this should equal 1 within
        // let total_sum: f32 = result.iter().copied().sum();
        // println!(
        //     "Final sum = {total_sum}; sum/n = {:.2} (should be ~1)\n",
        //     total_sum / n as f32
        // );

        // assert!(false);
        Ok(())
    }

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
