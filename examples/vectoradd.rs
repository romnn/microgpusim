fn cuda_malloc<T>(host_ptr: &mut T) -> DevicePtr<'_, T> {
    DevicePtr { inner: host_ptr }
}

#[derive(Debug)]
struct DevicePtr<'a, T> {
    inner: &'a mut T,
}

pub trait ToFlatIndex {
    fn flatten(&self) -> usize;
}

impl ToFlatIndex for usize {
    fn flatten(&self) -> usize {
        *self
    }
}

impl<T, O, I> std::ops::Index<I> for DevicePtr<'_, T>
where
    T: std::ops::Index<I, Output = O> + std::fmt::Debug,
    I: ToFlatIndex + std::fmt::Debug,
{
    type Output = O;

    fn index(&self, idx: I) -> &Self::Output {
        let elem_size = std::mem::size_of::<O>();
        let flat_idx = idx.flatten();
        let addr = elem_size * flat_idx;
        println!("{:?}[{:?}] => {}", &self, &idx, &addr);
        &self.inner[idx]
    }
}

impl<T, O, I> std::ops::IndexMut<I> for DevicePtr<'_, T>
where
    T: std::ops::IndexMut<I, Output = O> + std::fmt::Debug,
    I: ToFlatIndex + std::fmt::Debug,
{
    fn index_mut(&mut self, idx: I) -> &mut Self::Output {
        let elem_size = std::mem::size_of::<O>();
        let flat_idx = idx.flatten();
        let addr = elem_size * flat_idx;
        println!("{:?}[{:?}] => {}", &self, &idx, &addr);
        &mut self.inner[idx]
    }
}

fn vec_add_gpu(
    d_a: &mut DevicePtr<Vec<f32>>,
    d_b: &mut DevicePtr<Vec<f32>>,
    d_c: &mut DevicePtr<Vec<f32>>,
    n: usize,
) {
    // Get our global thread ID
    // int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we do not go out of bounds
    // if (id < n) c[id] = a[id] + b[id];
    for id in 0..n {
        d_c[id] = d_a[id] + d_b[id];
    }
}

struct Simualation {}

fn vectoradd(n: usize) {
    // create host vectors
    let mut a: Vec<f32> = vec![0.0; n];
    let mut b: Vec<f32> = vec![0.0; n];
    let mut c: Vec<f32> = vec![0.0; n];

    for i in 0..n {
        let angle = i as f32;
        a[i] = angle.sin() * angle.sin();
        b[i] = angle.cos() * angle.cos();
        c[i] = 0.0;
    }

    // allocate memory for each vector on GPU
    let mut d_a = cuda_malloc(&mut a);
    let mut d_b = cuda_malloc(&mut b);
    let mut d_c = cuda_malloc(&mut c);

    // let d_a = cuda_malloc(a);
    dbg!(&d_a);
    // cudaMalloc(&d_a, bytes);

    // int blockSize, gridSize;

    // // Number of threads in each thread block
    // blockSize = 1024;

    // // Number of thread blocks in grid
    // gridSize = (int)ceil((float)n / blockSize);

    vec_add_gpu(&mut d_a, &mut d_b, &mut d_c, n);
    // // Execute the kernel
    // CUDA_SAFECALL((vecAdd<T><<<gridSize, blockSize>>>(d_a, d_b, d_c, n)));

    // // Copy array back to host
    // cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Sum up vector c and print result divided by n, this should equal 1 within
    let sum: f32 = c.iter().sum();
    println!(
        "Final sum = {sum}; sum/n = {} (should be ~1)\n",
        sum / n as f32
    );
}

fn main() {
    // todo: add num traits
    vectoradd(10);
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

        let mut reader = std::io::BufReader::new(
            std::fs::OpenOptions::new()
                .read(true)
                .open(&rmp_trace_file_path)
                .unwrap(),
        );
        let mut reader = rmp_serde::Deserializer::new(reader);

        // todo: factor this out of trace crate
        // #[derive(Debug, Clone, serde::Deserialize)]
        // struct MemAccessTraceEntry {
        //     pub cuda_ctx: u64,
        //     pub grid_launch_id: u64,
        //     pub cta_id: nvbit_rs::Dim,
        //     pub warp_id: u32,
        //     pub instr_opcode: String,
        //     pub instr_offset: u32,
        //     pub instr_idx: u32,
        //     pub instr_predicate: nvbit_rs::Predicate,
        //     pub instr_mem_space: nvbit_rs::MemorySpace,
        //     pub instr_is_load: bool,
        //     pub instr_is_store: bool,
        //     pub instr_is_extended: bool,
        //     /// Accessed address per thread of a warp
        //     pub addrs: [u64; 32],
        // }

        let decoder = nvbit_rs::Decoder::new(|access: String| {
            println!("{:?}", &access);
        });
        reader.deserialize_seq(decoder)?;

        assert!(false);
        Ok(())
    }
}
