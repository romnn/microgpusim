use gpucachesim::exec::tracegen::{self, TraceGenerator, Tracer};
use gpucachesim::exec::{alloc, Kernel, MemorySpace, ThreadBlock, ThreadIndex};
use num_traits::{Float, NumCast, Zero};
use tokio::sync::Mutex;

pub const DOT_NUM_BLOCKS: u32 = 256;
pub const SCALAR: f32 = 0.4;
pub const START_A: f32 = 0.1;
pub const START_B: f32 = 0.2;
pub const START_C: f32 = 0.0;

struct InitKernel<'a, T> {
    dev_a: Mutex<alloc::DevicePtr<&'a mut Vec<T>>>,
    dev_b: Mutex<alloc::DevicePtr<&'a mut Vec<T>>>,
    dev_c: Mutex<alloc::DevicePtr<&'a mut Vec<T>>>,
    initial_value_a: T,
    initial_value_b: T,
    initial_value_c: T,
}

#[async_trait::async_trait]
impl<'a, T> Kernel for InitKernel<'a, T>
where
    T: Float + Send + Sync,
{
    type Error = std::convert::Infallible;

    #[gpucachesim::exec::instrument_control_flow]
    async fn run(&self, block: &ThreadBlock, tid: &ThreadIndex) -> Result<(), Self::Error> {
        let idx = (tid.block_idx.x * tid.block_dim.x + tid.thread_idx.x) as usize;

        let mut dev_a = self.dev_a.lock().await;
        let mut dev_b = self.dev_b.lock().await;
        let mut dev_c = self.dev_c.lock().await;
        dev_a[(tid, idx)] = self.initial_value_a;
        dev_b[(tid, idx)] = self.initial_value_b;
        dev_c[(tid, idx)] = self.initial_value_c;
        Ok(())
    }

    fn name(&self) -> Option<&str> {
        Some("init_kernel")
    }
}

struct CopyKernel<'a, T> {
    dev_a: Mutex<alloc::DevicePtr<&'a mut Vec<T>>>,
    dev_c: Mutex<alloc::DevicePtr<&'a mut Vec<T>>>,
}

#[async_trait::async_trait]
impl<'a, T> Kernel for CopyKernel<'a, T>
where
    T: Float + Send + Sync,
{
    type Error = std::convert::Infallible;

    #[gpucachesim::exec::instrument_control_flow]
    async fn run(&self, block: &ThreadBlock, tid: &ThreadIndex) -> Result<(), Self::Error> {
        let idx = (tid.block_idx.x * tid.block_dim.x + tid.thread_idx.x) as usize;

        let dev_a = self.dev_a.lock().await;
        let mut dev_c = self.dev_c.lock().await;
        dev_c[(tid, idx)] = dev_a[(tid, idx)];
        Ok(())
    }

    fn name(&self) -> Option<&str> {
        Some("copy_kernel")
    }
}

struct MulKernel<'a, T> {
    dev_b: Mutex<alloc::DevicePtr<&'a mut Vec<T>>>,
    dev_c: Mutex<alloc::DevicePtr<&'a mut Vec<T>>>,
}

#[async_trait::async_trait]
impl<'a, T> Kernel for MulKernel<'a, T>
where
    T: Float + Send + Sync,
{
    type Error = std::convert::Infallible;

    #[gpucachesim::exec::instrument_control_flow]
    async fn run(&self, block: &ThreadBlock, tid: &ThreadIndex) -> Result<(), Self::Error> {
        let idx = (tid.block_idx.x * tid.block_dim.x + tid.thread_idx.x) as usize;

        let mut dev_b = self.dev_b.lock().await;
        let dev_c = self.dev_c.lock().await;
        let scalar: T = NumCast::from(SCALAR).unwrap();
        dev_b[(tid, idx)] = scalar * dev_c[(tid, idx)];
        Ok(())
    }

    fn name(&self) -> Option<&str> {
        Some("mul_kernel")
    }
}

struct AddKernel<'a, T> {
    dev_a: Mutex<alloc::DevicePtr<&'a mut Vec<T>>>,
    dev_b: Mutex<alloc::DevicePtr<&'a mut Vec<T>>>,
    dev_c: Mutex<alloc::DevicePtr<&'a mut Vec<T>>>,
}

#[async_trait::async_trait]
impl<'a, T> Kernel for AddKernel<'a, T>
where
    T: Float + Send + Sync,
{
    type Error = std::convert::Infallible;

    #[gpucachesim::exec::instrument_control_flow]
    async fn run(&self, block: &ThreadBlock, tid: &ThreadIndex) -> Result<(), Self::Error> {
        let idx = (tid.block_idx.x * tid.block_dim.x + tid.thread_idx.x) as usize;

        let dev_a = self.dev_a.lock().await;
        let dev_b = self.dev_b.lock().await;
        let mut dev_c = self.dev_c.lock().await;

        dev_c[(tid, idx)] = dev_a[(tid, idx)] + dev_b[(tid, idx)];
        Ok(())
    }

    fn name(&self) -> Option<&str> {
        Some("add_kernel")
    }
}

struct TriadKernel<'a, T> {
    dev_a: Mutex<alloc::DevicePtr<&'a mut Vec<T>>>,
    dev_b: Mutex<alloc::DevicePtr<&'a mut Vec<T>>>,
    dev_c: Mutex<alloc::DevicePtr<&'a mut Vec<T>>>,
}

#[async_trait::async_trait]
impl<'a, T> Kernel for TriadKernel<'a, T>
where
    T: Float + Send + Sync,
{
    type Error = std::convert::Infallible;

    #[gpucachesim::exec::instrument_control_flow]
    async fn run(&self, block: &ThreadBlock, tid: &ThreadIndex) -> Result<(), Self::Error> {
        let idx = (tid.block_idx.x * tid.block_dim.x + tid.thread_idx.x) as usize;

        let mut dev_a = self.dev_a.lock().await;
        let dev_b = self.dev_b.lock().await;
        let dev_c = self.dev_c.lock().await;

        let scalar: T = NumCast::from(SCALAR).unwrap();

        dev_a[(tid, idx)] = dev_b[(tid, idx)] + scalar * dev_c[(tid, idx)];
        Ok(())
    }

    fn name(&self) -> Option<&str> {
        Some("triad_kernel")
    }
}

struct DotKernel<'a, T> {
    dev_a: Mutex<alloc::DevicePtr<&'a mut Vec<T>>>,
    dev_b: Mutex<alloc::DevicePtr<&'a mut Vec<T>>>,
    dev_sums: Mutex<alloc::DevicePtr<&'a mut Vec<T>>>,
    shared_block_sums: Mutex<alloc::DevicePtr<&'a mut Vec<T>>>,
}

#[async_trait::async_trait]
impl<'a, T> Kernel for DotKernel<'a, T>
where
    T: Float + std::ops::AddAssign + Send + Sync,
{
    type Error = std::convert::Infallible;

    #[gpucachesim::exec::instrument_control_flow]
    async fn run(&self, tracer: &ThreadBlock, tid: &ThreadIndex) -> Result<(), Self::Error> {
        let mut idx = (tid.block_idx.x * tid.block_dim.x + tid.thread_idx.x) as usize;
        let local_idx = tid.thread_idx.x as usize;
        let array_size = self.dev_a.lock().await.inner.len();

        // initialize to zero
        self.shared_block_sums.lock().await[(tid, local_idx)] = T::zero();

        while idx < array_size {
            let dev_a = self.dev_a.lock().await;
            let dev_b = self.dev_b.lock().await;
            let mut block_sums = self.shared_block_sums.lock().await;
            block_sums[(tid, local_idx)] += dev_a[(tid, idx)] * dev_b[(tid, idx)];
            idx += (tid.block_dim.x * tid.grid_dim.x) as usize;
        }

        let mut offset = tid.block_dim.x as usize / 2;
        while offset > 0 {
            tracer.synchronize_threads().await;

            if local_idx < offset {
                let mut block_sums = self.shared_block_sums.lock().await;
                let partial = block_sums[(tid, local_idx + offset)];
                block_sums[(tid, local_idx)] += partial;
            }
            offset /= 2;
        }

        if local_idx == 0 {
            let mut sums = self.dev_sums.lock().await;
            let block_sums = self.shared_block_sums.lock().await;
            let block_idx = tid.block_idx.x as usize;
            sums[(tid, block_idx)] = block_sums[(tid, local_idx)];
        }
        Ok(())
    }

    fn name(&self) -> Option<&str> {
        Some("dot_kernel")
    }
}

/// Generate correct reference solution
pub fn reference<T>(n: usize, repetitions: usize) -> (T, T, T, T)
where
    T: Float,
{
    let mut gold_a: T = NumCast::from(START_A).unwrap();
    let mut gold_b: T = NumCast::from(START_B).unwrap();
    let mut gold_c: T = NumCast::from(START_C).unwrap();

    let scalar: T = NumCast::from(SCALAR).unwrap();

    for _ in 0..repetitions {
        gold_c = gold_a;
        gold_b = scalar * gold_c;
        gold_c = gold_a + gold_b;
        gold_a = gold_b + scalar * gold_c;
    }

    let gold_sum = gold_a * gold_b * NumCast::from(n).unwrap();
    (gold_a, gold_b, gold_c, gold_sum)
}

/// Babelstream benchmark application.
pub async fn benchmark<T>(n: usize) -> super::Result
where
    T: Float + std::ops::AddAssign + Zero + Send + Sync,
{
    // create host vectors
    let mut a: Vec<T> = vec![T::zero(); n];
    let mut b: Vec<T> = vec![T::zero(); n];
    let mut c: Vec<T> = vec![T::zero(); n];
    let mut sum: T = T::zero();

    babelstream(&mut a, &mut b, &mut c, &mut sum, 1024).await
}

pub async fn babelstream<T>(
    a: &mut Vec<T>,
    b: &mut Vec<T>,
    c: &mut Vec<T>,
    sum: &mut T,
    block_size: u32,
) -> super::Result
where
    T: Float + std::ops::AddAssign + Zero + Send + Sync,
{
    let tracer = Tracer::new();

    assert_eq!(a.len(), b.len());
    assert_eq!(b.len(), c.len());
    let n = a.len();

    // allocate memory for each vector on simulated GPU device
    let dev_a = tracer
        .allocate(
            a,
            Some(alloc::Options {
                mem_space: MemorySpace::Global,
                name: Some("a".to_string()),
                ..alloc::Options::default()
            }),
        )
        .await;
    let dev_b = tracer
        .allocate(
            b,
            Some(alloc::Options {
                mem_space: MemorySpace::Global,
                name: Some("b".to_string()),
                ..alloc::Options::default()
            }),
        )
        .await;
    let dev_c = tracer
        .allocate(
            c,
            Some(alloc::Options {
                mem_space: MemorySpace::Global,
                name: Some("c".to_string()),
                ..alloc::Options::default()
            }),
        )
        .await;

    let mut sums = vec![T::zero(); DOT_NUM_BLOCKS as usize];
    let dev_sums = tracer
        .allocate(
            &mut sums,
            Some(alloc::Options {
                mem_space: MemorySpace::Global,
                name: Some("sums".to_string()),
                ..alloc::Options::default()
            }),
        )
        .await;

    let options = tracegen::Options::default();
    let dev_a = Mutex::new(dev_a);
    let dev_b = Mutex::new(dev_b);
    let dev_c = Mutex::new(dev_c);
    let dev_sums = Mutex::new(dev_sums);

    let grid_size = (n as f32 / block_size as f32).ceil() as u32;

    let mut traces: Vec<(
        trace_model::command::KernelLaunch,
        trace_model::MemAccessTrace,
    )> = Vec::new();

    // 0. init
    let mut init_kernel: InitKernel<T> = InitKernel {
        dev_a,
        dev_b,
        dev_c,
        initial_value_a: NumCast::from(START_A).unwrap(),
        initial_value_b: NumCast::from(START_B).unwrap(),
        initial_value_c: NumCast::from(START_C).unwrap(),
    };
    traces.push(
        tracer
            .trace_kernel(grid_size, block_size, &mut init_kernel, &options)
            .await?,
    );

    let InitKernel {
        dev_a,
        dev_b,
        dev_c,
        ..
    } = init_kernel;

    // 1. copy
    let mut copy_kernel: CopyKernel<T> = CopyKernel { dev_a, dev_c };
    traces.push(
        tracer
            .trace_kernel(grid_size, block_size, &mut copy_kernel, &options)
            .await?,
    );
    let CopyKernel { dev_a, dev_c } = copy_kernel;

    // 2. mul
    let mut mul_kernel: MulKernel<T> = MulKernel { dev_b, dev_c };
    traces.push(
        tracer
            .trace_kernel(grid_size, block_size, &mut mul_kernel, &options)
            .await?,
    );
    let MulKernel { dev_b, dev_c } = mul_kernel;

    // 3. add
    let mut add_kernel: AddKernel<T> = AddKernel {
        dev_a,
        dev_b,
        dev_c,
    };
    traces.push(
        tracer
            .trace_kernel(grid_size, block_size, &mut add_kernel, &options)
            .await?,
    );
    let AddKernel {
        dev_a,
        dev_b,
        dev_c,
    } = add_kernel;

    // 4. triad
    let mut triad_kernel: TriadKernel<T> = TriadKernel {
        dev_a,
        dev_b,
        dev_c,
    };
    traces.push(
        tracer
            .trace_kernel(grid_size, block_size, &mut triad_kernel, &options)
            .await?,
    );
    let TriadKernel { dev_a, dev_b, .. } = triad_kernel;

    // 5. dot
    let mut block_sums = vec![T::zero(); block_size as usize];
    let shared_block_sums = tracer
        .allocate(
            &mut block_sums,
            Some(alloc::Options {
                mem_space: MemorySpace::Shared,
                name: Some("shared_block_sums".to_string()),
                ..alloc::Options::default()
            }),
        )
        .await;

    let mut dot_kernel: DotKernel<T> = DotKernel {
        dev_a,
        dev_b,
        dev_sums,
        shared_block_sums: Mutex::new(shared_block_sums),
    };
    traces.push(
        tracer
            .trace_kernel(DOT_NUM_BLOCKS, block_size, &mut dot_kernel, &options)
            .await?,
    );

    *sum = T::zero();
    for i in 0..DOT_NUM_BLOCKS as usize {
        *sum += sums[i];
    }

    Ok((tracer.commands().await, traces))
}

#[cfg(test)]
mod tests {
    use color_eyre::eyre;
    use gpucachesim::exec::tracegen::fmt::{self, Addresses, SimplifiedTraceInstruction};
    use ndarray::Array1;
    use utils::diff;

    const EPSILON: f32 = 0.0001;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_correctness() -> eyre::Result<()> {
        crate::tests::init_test();

        // create host vectors
        let n = 128;
        let repetitions = 1;
        let block_size = 64;

        let mut a: Vec<f32> = vec![super::START_A; n];
        let mut b: Vec<f32> = vec![super::START_B; n];
        let mut c: Vec<f32> = vec![super::START_C; n];

        let (gold_a, gold_b, gold_c, gold_sum) = super::reference::<f32>(n, repetitions);

        let gold_a_vec = Array1::from_elem(n, gold_a);
        let gold_b_vec = Array1::from_elem(n, gold_b);
        let gold_c_vec = Array1::from_elem(n, gold_c);

        let mut sum = 0.0;
        let (_commands, kernel_traces) =
            super::babelstream(&mut a, &mut b, &mut c, &mut sum, block_size).await?;

        // there are 5 main kernels and the init kernel
        assert_eq!(kernel_traces.len(), 6);

        let a_vec = Array1::from_shape_vec(n, a)?;
        let b_vec = Array1::from_shape_vec(n, b)?;
        let c_vec = Array1::from_shape_vec(n, c)?;

        dbg!(&a_vec[0], &b_vec[0], &c_vec[0]);
        dbg!(&gold_a_vec[0], &gold_b_vec[0], &gold_c_vec[0]);
        dbg!(&sum, gold_sum);

        if !approx::abs_diff_eq!(a_vec, gold_a_vec, epsilon = EPSILON) {
            diff::assert_eq!(have: a_vec, want: gold_a_vec);
        }
        if !approx::abs_diff_eq!(b_vec, gold_b_vec, epsilon = EPSILON) {
            diff::assert_eq!(have: b_vec, want: gold_b_vec);
        }
        if !approx::abs_diff_eq!(c_vec, gold_c_vec, epsilon = EPSILON) {
            diff::assert_eq!(have: c_vec, want: gold_c_vec);
        }
        if !approx::abs_diff_eq!(sum, gold_sum, epsilon = EPSILON) {
            diff::assert_eq!(have: sum, want: gold_sum);
        }

        let (_dot_kernel_launch, dot_kernel_trace) = &kernel_traces[5];
        let warp_traces = dot_kernel_trace.clone().to_warp_traces();
        let first_warp = &warp_traces[&(trace_model::Dim::ZERO, 0)];

        let have: Vec<_> = fmt::simplify_warp_trace(&first_warp, true).collect();
        for inst in &have {
            println!("{}", inst);
        }
        let want: Vec<_> = [
            (
                "STS.E",
                Addresses::BaseStride { base: 0, stride: 4 },
                "11111111111111111111111111111111",
                0,
            ),
            (
                "STS.E",
                Addresses::BaseStride { base: 0, stride: 4 },
                "11111111111111111111111111111111",
                1,
            ),
            (
                "LDG.E",
                Addresses::BaseStride { base: 0, stride: 4 },
                "11111111111111111111111111111111",
                2,
            ),
            (
                "LDG.E",
                Addresses::BaseStride {
                    base: 512,
                    stride: 4,
                },
                "11111111111111111111111111111111",
                3,
            ),
            (
                "MEMBAR",
                Addresses::None,
                "11111111111111111111111111111111",
                4,
            ),
            (
                "LDS.E",
                Addresses::BaseStride {
                    base: 128,
                    stride: 4,
                },
                "11111111111111111111111111111111",
                5,
            ),
            (
                "STS.E",
                Addresses::BaseStride { base: 0, stride: 4 },
                "11111111111111111111111111111111",
                6,
            ),
            (
                "MEMBAR",
                Addresses::None,
                "11111111111111111111111111111111",
                7,
            ),
            (
                "LDS.E",
                Addresses::BaseStride {
                    base: 64,
                    stride: 4,
                },
                "11111111111111110000000000000000",
                8,
            ),
            (
                "STS.E",
                Addresses::BaseStride { base: 0, stride: 4 },
                "11111111111111110000000000000000",
                9,
            ),
            (
                "MEMBAR",
                Addresses::None,
                "11111111111111111111111111111111",
                10,
            ),
            (
                "LDS.E",
                Addresses::BaseStride {
                    base: 32,
                    stride: 4,
                },
                "11111111000000000000000000000000",
                11,
            ),
            (
                "STS.E",
                Addresses::BaseStride { base: 0, stride: 4 },
                "11111111000000000000000000000000",
                12,
            ),
            (
                "MEMBAR",
                Addresses::None,
                "11111111111111111111111111111111",
                13,
            ),
            (
                "LDS.E",
                Addresses::BaseStride {
                    base: 16,
                    stride: 4,
                },
                "11110000000000000000000000000000",
                14,
            ),
            (
                "STS.E",
                Addresses::BaseStride { base: 0, stride: 4 },
                "11110000000000000000000000000000",
                15,
            ),
            (
                "MEMBAR",
                Addresses::None,
                "11111111111111111111111111111111",
                16,
            ),
            (
                "LDS.E",
                Addresses::BaseStride { base: 8, stride: 4 },
                "11000000000000000000000000000000",
                17,
            ),
            (
                "STS.E",
                Addresses::BaseStride { base: 0, stride: 4 },
                "11000000000000000000000000000000",
                18,
            ),
            (
                "MEMBAR",
                Addresses::None,
                "11111111111111111111111111111111",
                19,
            ),
            (
                "LDS.E",
                Addresses::Uniform(4),
                "10000000000000000000000000000000",
                20,
            ),
            (
                "STS.E",
                Addresses::Uniform(0),
                "10000000000000000000000000000000",
                21,
            ),
            (
                "LDS.E",
                Addresses::Uniform(0),
                "10000000000000000000000000000000",
                22,
            ),
            (
                "STG.E",
                Addresses::Uniform(1536),
                "10000000000000000000000000000000",
                23,
            ),
            (
                "EXIT",
                Addresses::None,
                "11111111111111111111111111111111",
                24,
            ),
            // (
            //     "LDG.E",
            //     Addresses::BaseStride { base: 0, stride: 4 },
            //     "11111111111111111111111111111111",
            //     0,
            // ),
            // (
            //     "LDG.E",
            //     Addresses::BaseStride {
            //         base: 512,
            //         stride: 4,
            //     },
            //     "11111111111111111111111111111111",
            //     1,
            // ),
            // (
            //     "STG.E",
            //     Addresses::BaseStride {
            //         base: 1024,
            //         stride: 4,
            //     },
            //     "11111111111111111111111111111111",
            //     2,
            // ),
            // (
            //     "EXIT",
            //     Addresses::None,
            //     "11111111111111111111111111111111",
            //     3,
            // ),
        ]
        .into_iter()
        .enumerate()
        .map(SimplifiedTraceInstruction::from)
        .collect();

        dbg!(&have);
        diff::assert_eq!(have: have, want: want);

        Ok(())
    }
}
