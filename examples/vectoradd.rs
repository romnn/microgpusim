#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use anyhow::Result;
use num_traits::{Float, Zero};

#[derive(Debug)]
struct VecAdd<'s, 'a, T> {
    d_a: &'a mut casimu::DevicePtr<'s, 'a, Vec<T>>,
    d_b: &'a mut casimu::DevicePtr<'s, 'a, Vec<T>>,
    d_c: &'a mut casimu::DevicePtr<'s, 'a, Vec<T>>,
    n: usize,
}

impl<'s, 'a, T> casimu::Kernel for VecAdd<'s, 'a, T>
where
    T: num_traits::Float + std::fmt::Debug,
{
    type Error = std::convert::Infallible;

    fn run(&mut self, idx: &casimu::ThreadIndex) -> Result<(), Self::Error> {
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

fn vectoradd<T>(n: usize) -> Result<()>
where
    T: Float + Zero + num_traits::NumCast + std::iter::Sum + std::fmt::Display + std::fmt::Debug,
{
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

    let sim = casimu::Simulation::new();

    // allocate memory for each vector on simulated GPU device
    let mut d_a = sim.allocate(&mut a);
    let mut d_b = sim.allocate(&mut b);
    let mut d_c = sim.allocate(&mut c);

    // number of thread blocks in grid
    let grid_size = (n as f64 / f64::from(BLOCK_SIZE)).ceil() as u32;

    let kernel: VecAdd<T> = VecAdd {
        d_a: &mut d_a,
        d_b: &mut d_b,
        d_c: &mut d_c,
        n,
    };
    sim.launch_kernel(grid_size, BLOCK_SIZE, kernel)?;

    // sum up vector c and print result divided by n, this should equal 1 within
    let total_sum: T = c.into_iter().sum();
    println!(
        "Final sum = {total_sum}; sum/n = {} (should be ~1)\n",
        total_sum / T::from(n).unwrap()
    );
    Ok(())
}

fn main() -> Result<()> {
    vectoradd::<f32>(100)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use serde::Deserializer;
    use std::path::PathBuf;

    #[test]
    pub fn test_trace_based() -> Result<()> {
        let traces_dir = PathBuf::from(file!())
            .parent()
            .unwrap()
            .join("../validation/vectoradd/traces/vectoradd-100-32-trace");
        dbg!(&traces_dir);
        let rmp_trace_file_path = traces_dir.join("trace.msgpack");
        dbg!(&rmp_trace_file_path);

        let reader = std::io::BufReader::new(
            std::fs::OpenOptions::new()
                .read(true)
                .open(&rmp_trace_file_path)
                .unwrap(),
        );
        let mut reader = rmp_serde::Deserializer::new(reader);

        let decoder = nvbit_io::Decoder::new(|access: trace_model::MemAccessTraceEntry| {
            println!("{:#?}", &access);
        });
        reader.deserialize_seq(decoder)?;

        assert!(false);
        Ok(())
    }
}
