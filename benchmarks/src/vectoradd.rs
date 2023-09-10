#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use color_eyre::eyre;
use gpucachesim::exec;
use num_traits::{Float, NumCast, Zero};

#[derive(Debug)]
struct VecAdd<'s, 'a, T> {
    d_a: &'a mut exec::DevicePtr<'s, 'a, Vec<T>>,
    d_b: &'a mut exec::DevicePtr<'s, 'a, Vec<T>>,
    d_c: &'a mut exec::DevicePtr<'s, 'a, Vec<T>>,
    n: usize,
}

impl<'s, 'a, T> exec::Kernel for VecAdd<'s, 'a, T>
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
const BLOCK_SIZE: u32 = 1024;

fn vectoradd<T>(n: usize) -> eyre::Result<()>
where
    T: Float + Zero + NumCast + std::iter::Sum + std::fmt::Display + std::fmt::Debug,
{
    let start = std::time::Instant::now();

    // create host vectors
    let mut a: Vec<T> = vec![T::zero(); n];
    let mut b: Vec<T> = vec![T::zero(); n];
    let mut c: Vec<T> = vec![T::zero(); n];

    // initialize vectors
    for i in 0..n {
        let angle = T::from(i).unwrap();
        a[i] = angle.sin() * angle.sin();
        b[i] = angle.cos() * angle.cos();
        c[i] = T::zero();
    }

    let sim = exec::Simulation::new();

    // allocate memory for each vector on simulated GPU device
    let a_size = a.len() * std::mem::size_of::<T>();
    let b_size = b.len() * std::mem::size_of::<T>();
    let c_size = c.len() * std::mem::size_of::<T>();
    let mut d_a = sim.allocate(&mut a, a_size as u64, exec::MemorySpace::Global);
    let mut d_b = sim.allocate(&mut b, b_size as u64, exec::MemorySpace::Global);
    let mut d_c = sim.allocate(&mut c, c_size as u64, exec::MemorySpace::Global);

    // number of thread blocks in grid
    let grid_size = (n as f64 / <f64 as From<_>>::from(BLOCK_SIZE)).ceil() as u32;

    let kernel: VecAdd<T> = VecAdd {
        d_a: &mut d_a,
        d_b: &mut d_b,
        d_c: &mut d_c,
        n,
    };
    sim.launch_kernel(grid_size, BLOCK_SIZE, kernel)?;
    let stats = sim.run_to_completion()?;

    // sum up vector c and print result divided by n.
    // this should equal 1 within
    let total_sum: T = c.into_iter().sum();
    println!(
        "Final sum = {total_sum}; sum/n = {:.2} (should be ~1)\n",
        total_sum / T::from(n).unwrap()
    );

    eprintln!("STATS:\n");
    eprintln!("DRAM: total reads: {}", &stats.dram.total_reads());
    eprintln!("DRAM: total writes: {}", &stats.dram.total_writes());
    eprintln!("SIM: {:#?}", &stats.sim);
    eprintln!("INSTRUCTIONS: {:#?}", &stats.instructions);
    eprintln!("ACCESSES: {:#?}", &stats.accesses);
    eprintln!("L1I: {:#?}", &stats.l1i_stats.reduce());
    eprintln!("L1D: {:#?}", &stats.l1d_stats.reduce());
    eprintln!("L2D: {:#?}", &stats.l2d_stats.reduce());
    eprintln!("completed in {:?}", start.elapsed());
    Ok(())
}

fn main() -> eyre::Result<()> {
    env_logger::init();
    vectoradd::<f32>(100)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use color_eyre::eyre;
    use std::path::PathBuf;
    use trace_model as model;

    #[test]
    pub fn trace_instructions() -> eyre::Result<()> {
        let traces_dir = PathBuf::from(file!())
            .parent()
            .unwrap()
            .join("../results/vectorAdd/vectorAdd-dtype-32-length-100/trace");
        dbg!(&traces_dir);
        let rmp_trace_file_path = traces_dir.join("kernel-0.msgpack");
        dbg!(&rmp_trace_file_path);

        let mut reader = utils::fs::open_readable(rmp_trace_file_path)?;
        let full_trace: model::MemAccessTrace = rmp_serde::from_read(&mut reader)?;
        let warp_traces = full_trace.to_warp_traces();
        dbg!(&warp_traces[&(model::Dim::ZERO, 0)]
            .iter()
            .map(|entry| (&entry.instr_opcode, &entry.active_mask))
            .collect::<Vec<_>>());

        assert!(false);
        Ok(())
    }
}
