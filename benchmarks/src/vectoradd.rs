#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use color_eyre::eyre;
use gpucachesim::exec;
use num_traits::{Float, NumCast, Zero};

#[derive(Debug)]
// struct VecAdd<'s, 'a, T> {
struct VecAdd<'s, T> {
    // dev_a: &'a exec::DevicePtr<'s, 'a, Vec<T>>,
    dev_a: exec::DevicePtr<'s, Vec<T>>,
    // dev_b: &'a exec::DevicePtr<'s, 'a, Vec<T>>,
    dev_b: exec::DevicePtr<'s, Vec<T>>,
    // dev_result: &'a mut exec::DevicePtr<'s, 'a, Vec<T>>,
    dev_result: exec::DevicePtr<'s, Vec<T>>,
    n: usize,
}

// impl<'s, 'a, T> exec::Kernel for VecAdd<'s, 'a, T>
impl<'s, T> exec::Kernel for VecAdd<'s, T>
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
        let active = id < self.n;
        self.dev_result[(id, active)] = self.dev_a[(id, active)] + self.dev_b[(id, active)];

        // if id < self.n {
        //     self.dev_result[id] = self.dev_a[id] + self.dev_b[id];
        // } else {
        //     // self.dev_result[id] = self.dev_a[id] + self.dev_b[id];
        // }
        Ok(())
    }
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

pub fn default_vectoradd<T>(n: usize) -> eyre::Result<()>
where
    T: Float + Zero + NumCast + std::iter::Sum + std::fmt::Display + std::fmt::Debug,
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

    vectoradd(&a, &b, &mut result)?;
    Ok(())
}

pub fn vectoradd<T>(a: &Vec<T>, b: &Vec<T>, result: &mut Vec<T>) -> eyre::Result<()>
where
    T: Float + Zero + NumCast + std::iter::Sum + std::fmt::Display + std::fmt::Debug,
{
    let start = std::time::Instant::now();

    // let sim = exec::Simulation::new();
    // assert_eq!(a.len(), b.len());
    // assert_eq!(b.len(), c.len());
    // let n = a.len();
    //
    // // allocate memory for each vector on simulated GPU device
    // let a_size = a.len() * std::mem::size_of::<T>();
    // let b_size = b.len() * std::mem::size_of::<T>();
    // let c_size = c.len() * std::mem::size_of::<T>();
    // let mut d_a = sim.allocate(a, a_size as u64, exec::MemorySpace::Global);
    // let mut d_b = sim.allocate(b, b_size as u64, exec::MemorySpace::Global);
    // let mut d_c = sim.allocate(c, c_size as u64, exec::MemorySpace::Global);
    //
    // // number of thread blocks in grid
    // let grid_size = (n as f64 / <f64 as From<_>>::from(BLOCK_SIZE)).ceil() as u32;
    //
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
    use utils::diff;
    // use std::path::PathBuf;
    // use trace_model as model;
    use gpucachesim::exec::{MemorySpace, TraceGenerator, Tracer};

    #[test]
    pub fn test_correctness() -> eyre::Result<()> {
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

        super::vectoradd(&mut a, &mut b, &mut result)?;
        super::reference_vectoradd(&mut a, &mut b, &mut ref_result);

        diff::diff!(have: result, want: ref_result);
        Ok(())
    }

    #[test]
    pub fn generate_trace() -> eyre::Result<()> {
        let n = 100;
        let mut tracer = Tracer::default();

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
        let a_size = a.len() * std::mem::size_of::<f32>();
        let b_size = b.len() * std::mem::size_of::<f32>();
        let result_size = result.len() * std::mem::size_of::<f32>();

        // let mut dev_a = tracer.allocate(&mut a, a_size as u64, MemorySpace::Global);
        // let mut dev_b = tracer.allocate(&mut b, b_size as u64, MemorySpace::Global);
        // let mut dev_result = tracer.allocate(&mut result, result_size as u64, MemorySpace::Global);

        let mut dev_a = tracer.allocate(a, a_size as u64, MemorySpace::Global);
        let mut dev_b = tracer.allocate(b, b_size as u64, MemorySpace::Global);
        let mut dev_result = tracer.allocate(result, result_size as u64, MemorySpace::Global);

        // number of thread blocks in grid
        let grid_size = (n as f64 / <f64 as From<_>>::from(super::BLOCK_SIZE)).ceil() as u32;

        let kernel: super::VecAdd<f32> = super::VecAdd {
            dev_a,
            dev_b,
            dev_result,
            // dev_a: &mut dev_a,
            // dev_b: &mut dev_b,
            // dev_result: &mut dev_result,
            n,
        };
        tracer.trace_kernel(grid_size, super::BLOCK_SIZE, kernel)?;
        // let stats = sim.run_to_completion()?;

        // sum up vector c and print result divided by n.
        // this should equal 1 within
        // let total_sum: f32 = result.iter().copied().sum();
        // println!(
        //     "Final sum = {total_sum}; sum/n = {:.2} (should be ~1)\n",
        //     total_sum / n as f32
        // );

        assert!(false);
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
