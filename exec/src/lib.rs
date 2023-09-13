#![allow(clippy::missing_panics_doc, clippy::missing_errors_doc)]
#![allow(warnings)]

pub mod r#async;
pub mod kernel;
pub mod model;
pub mod nop;
pub mod tracegen;

use std::sync::{atomic, Arc, Mutex};
// use crate::{interconn as ic, mem_fetch};
use bitvec::field::BitField;
use color_eyre::eyre;
use itertools::Itertools;
use num_traits::Zero;
// use nvbit_model;
// use trace_model as model;

pub use kernel::{Kernel, ThreadIndex};
pub use model::MemorySpace;
pub use nop::ArithmeticNop;
pub use tracegen::{TraceGenerator, Tracer};

// pub trait Container<O> {}
// pub trait Container {
//     type Elem;
// }

// get rid of this trait to support also just heap allocated memory we want to get in full?
// impl<T, E> Container for T
// where
//     T: AsRef<[E]>,
//     // T: std::ops::Index<usize, Output = E>,
// {
//     type Elem = E;
// }

#[derive()]
pub struct DevicePtr<'s, C, T>
// where
//     T: Container,
{
    // pub struct DevicePtr<'s, T, O> {
    // inner: &'a mut T,
    inner: C,
    marker: std::marker::PhantomData<T>,
    // TODO: refactor this
    // spare: <T as Container>::Elem,
    nop: ArithmeticNop,
    memory: &'s dyn tracegen::MemoryAccess,
    mem_space: model::MemorySpace,
    offset: u64,
}

impl<'s, C, T> DevicePtr<'s, C, T>
// where
//     T: Container,
{
    pub fn into_inner(self) -> C {
        self.inner
    }
}

impl<'s, C, T> std::fmt::Debug for DevicePtr<'s, C, T>
where
    // T: Container + std::fmt::Debug,
    C: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.inner, f)
    }
}

impl<'s, C, T> std::fmt::Display for DevicePtr<'s, C, T>
where
    C: std::fmt::Display,
    // T: Container + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.inner, f)
    }
}

/// Convert multi-dimensional index into flat linear index.
///
/// Users may override this to provide complex index transformations.
pub trait ToLinear {
    fn to_linear(&self) -> usize;
}

/// Simple linear index.
impl ToLinear for usize {
    fn to_linear(&self) -> usize {
        *self
    }
}

// impl<T, I> std::ops::Index<I> for DevicePtr<'_, T> {
//     T: Container + std::ops::Index<usize, Output = T::Elem> + std::fmt::Debug {
//     pub fn get(&self) -> {
//         let elem_size = std::mem::size_of::<T::Elem>();
//         let flat_idx = idx.flatten();
//         let addr = self.offset + elem_size as u64 * flat_idx as u64;
//         self.memory
//             .load(addr, elem_size as u32, self.mem_space, true);
//         log::trace!("{:?}[{:?}] => {}", &self.inner, &idx, &addr);
//         &self.inner[idx]
//     }
// }

// impl<T, I> std::ops::Index<I> for DevicePtr<'_, T>
// where
//     T: std::ops::Index<I, Output = T::Elem> + std::fmt::Debug,
//     // T: Container + std::ops::Index<I, Output = T::Elem> + std::fmt::Debug,
//     I: ToLinear + std::fmt::Debug,
// {
//     type Output = T::Elem;
//
//     fn index(&self, idx: I) -> &Self::Output {
//         let elem_size = std::mem::size_of::<T::Elem>();
//         let flat_idx = idx.flatten();
//         let addr = self.offset + elem_size as u64 * flat_idx as u64;
//         self.memory
//             .load(addr, elem_size as u32, self.mem_space, true);
//         log::trace!("{:?}[{:?}] => {}", &self.inner, &idx, &addr);
//         &self.inner[idx]
//     }
// }

impl<C, T> std::ops::Index<()> for DevicePtr<'_, C, T>
// where
//     T: Container,
{
    type Output = ArithmeticNop;

    fn index(&self, _idx: ()) -> &Self::Output {
        &self.nop
    }
}

impl<C, T> std::ops::IndexMut<()> for DevicePtr<'_, C, T>
// where
//     T: Container,
{
    fn index_mut(&mut self, _idx: ()) -> &mut Self::Output {
        &mut self.nop
    }
}

// // impl<T, I> std::ops::Index<(I, bool)> for DevicePtr<'_, T>
// impl<T, I, E> std::ops::Index<(I, bool)> for DevicePtr<'_, T>
// where
//     T: AsRef<[E]> + std::fmt::Debug,
//     // T: Container + std::ops::Index<I, Output = T::Elem> + std::fmt::Debug,
//     I: ToFlatIndex + std::fmt::Debug,
// {
//     // type Output = T::Elem;
//
//     fn index(&self, idx: (I, bool)) -> &Self::Output {
//         let elem_size = std::mem::size_of::<T::Elem>();
//         let (idx, active) = idx;
//         let flat_idx = idx.flatten();
//         let addr = self.offset + elem_size as u64 * flat_idx as u64;
//         self.memory
//             .load(addr, elem_size as u32, self.mem_space, active);
//         log::trace!("{:?}[{:?}] => {}", &self.inner, &idx, &addr);
//         &NOP
//         // &self.spare
//     }
// }

// impl<T, I> std::ops::IndexMut<I> for DevicePtr<'_, T>
// where
//     T: std::ops::IndexMut<I, Output = T::Elem> + std::fmt::Debug,
//     // T: Container + std::ops::IndexMut<I, Output = T::Elem> + std::fmt::Debug,
//     I: ToFlatIndex + std::fmt::Debug,
// {
//     fn index_mut(&mut self, idx: I) -> &mut Self::Output {
//         let elem_size = std::mem::size_of::<T::Elem>();
//         let flat_idx = idx.flatten();
//         let addr = self.offset + elem_size as u64 * flat_idx as u64;
//         self.memory
//             .store(addr, elem_size as u32, self.mem_space, true);
//         log::trace!("{:?}[{:?}] => {}", &self.inner, &idx, &addr);
//         &mut self.inner[idx]
//     }
// }

// impl<T, I> std::ops::IndexMut<(I, bool)> for DevicePtr<'_, T>
// where
//     T: Container + std::ops::IndexMut<I, Output = T::Elem> + std::fmt::Debug,
//     I: ToFlatIndex + std::fmt::Debug,
// {
//     fn index_mut(&mut self, idx: (I, bool)) -> &mut Self::Output {
//         let elem_size = std::mem::size_of::<T::Elem>();
//         let (idx, active) = idx;
//         let flat_idx = idx.flatten();
//         let addr = self.offset + elem_size as u64 * flat_idx as u64;
//         self.memory
//             .store(addr, elem_size as u32, self.mem_space, active);
//         log::trace!("{:?}[{:?}] => {}", &self.inner, &idx, &addr);
//         if active {
//             &mut self.inner[idx]
//         } else {
//             &mut self.spare
//         }
//     }
// }

// /// Simulation
// #[derive()]
// pub struct Simulation {
//     offset: Mutex<u64>,
//     pub inner: Mutex<
//         crate::MockSimulator<crate::interconn::ToyInterconnect<ic::Packet<mem_fetch::MemFetch>>>,
//     >,
//     thread_instructions: Mutex<Vec<WarpInstruction>>,
//     kernel_id: atomic::AtomicU64,
// }
//
// impl Default for Simulation {
//     fn default() -> Self {
//         let config = Arc::new(crate::config::GPU::default());
//         let interconn = Arc::new(crate::interconn::ToyInterconnect::new(
//             config.num_simt_clusters,
//             config.num_memory_controllers * config.num_sub_partitions_per_memory_controller,
//         ));
//         let inner = Mutex::new(crate::MockSimulator::new(interconn, Arc::clone(&config)));
//
//         Self {
//             offset: Mutex::new(DEV_GLOBAL_HEAP_START),
//             thread_instructions: Mutex::new(Vec::new()),
//             kernel_id: atomic::AtomicU64::new(0),
//             inner,
//         }
//     }
// }
//
// impl Simulation {
//     #[must_use]
//     pub fn new() -> Self {
//         Self::default()
//     }
//
//     pub fn load(&self, addr: u64, size: u32, mem_space: MemorySpace) {
//         self.thread_instructions
//             .lock()
//             .push(WarpInstruction::Access(MemInstruction {
//                 kind: MemAccessKind::Load,
//                 addr,
//                 mem_space,
//                 size,
//             }));
//     }
//
//     pub fn store(&self, addr: u64, size: u32, mem_space: MemorySpace) {
//         self.thread_instructions
//             .lock()
//             .push(WarpInstruction::Access(MemInstruction {
//                 kind: MemAccessKind::Store,
//                 addr,
//                 mem_space,
//                 size,
//             }));
//     }
//
//     /// Allocate a variable.
//     pub fn allocate<'s, 'a, T>(
//         &'s self,
//         var: &'a mut T,
//         size: u64,
//         mem_space: MemorySpace,
//     ) -> DevicePtr<'s, 'a, T> {
//         let mut offset_lock = self.offset.lock();
//         let offset = *offset_lock;
//         *offset_lock += size;
//
//         self.inner.lock().gpu_mem_alloc(offset, size, None, 0);
//         self.inner.lock().memcopy_to_gpu(offset, size, None, 0);
//
//         DevicePtr {
//             inner: var,
//             mem_space,
//             sim: self,
//             offset,
//         }
//     }
//
//     pub fn run_to_completion(&self) -> eyre::Result<stats::Stats> {
//         let mut inner = self.inner.lock();
//         inner.run_to_completion()?;
//         Ok(inner.stats())
//     }
//
//     /// Launches a kernel.
//     ///
//     /// # Errors
//     /// When the kernel fails.
//     pub fn launch_kernel<G, B, K>(
//         &self,
//         grid: G,
//         block_size: B,
//         mut kernel: K,
//     ) -> Result<(), K::Error>
//     where
//         G: Into<model::Dim>,
//         B: Into<model::Dim>,
//         K: Kernel,
//     {
//         let grid: model::Dim = grid.into();
//         let block_size: model::Dim = block_size.into();
//
//         let mut trace = Vec::new();
//
//         // loop over the grid
//         for block_id in grid.clone() {
//             log::debug!("block {}", &block_id);
//
//             let mut thread_id = ThreadIndex {
//                 block_idx: block_id.to_dim(),
//                 block_dim: block_size.clone(),
//                 thread_idx: block_size.clone(),
//             };
//
//             // loop over the block size and form warps
//             let thread_ids = block_size.clone().into_iter();
//             for (warp_id_in_block, warp) in thread_ids
//                 .chunks(WARP_SIZE as usize)
//                 .into_iter()
//                 .enumerate()
//             {
//                 // log::info!("START WARP #{} ({:?})", &warp_id_in_block, &thread_id);
//                 let mut warp_instructions = [(); WARP_SIZE as usize].map(|_| Vec::new());
//
//                 for (thread_idx, warp_thread_idx) in warp.enumerate() {
//                     // log::debug!(
//                     //     "warp #{} thread {:?}",
//                     //     &warp_num,
//                     //     model::Dim::from(warp_thread_idx)
//                     // );
//                     thread_id.thread_idx = warp_thread_idx.into();
//                     kernel.run(&thread_id)?;
//                     warp_instructions[thread_idx].extend(self.thread_instructions.lock().drain(..));
//                 }
//
//                 let warp_instruction = model::MemAccessTraceEntry {
//                     cuda_ctx: 0,
//                     sm_id: 0,
//                     kernel_id: 0,
//                     block_id: block_id.clone().into(),
//                     warp_id_in_sm: warp_id_in_block as u32,
//                     warp_id_in_block: warp_id_in_block as u32,
//                     warp_size: WARP_SIZE,
//                     line_num: 0,
//                     instr_data_width: 0,
//                     instr_opcode: String::new(),
//                     instr_offset: 0,
//                     instr_idx: 0,
//                     instr_predicate: nvbit_model::Predicate::default(),
//                     instr_mem_space: nvbit_model::MemorySpace::None,
//                     instr_is_mem: false,
//                     instr_is_load: false,
//                     instr_is_store: false,
//                     instr_is_extended: false,
//                     dest_regs: [0; 1],
//                     num_dest_regs: 0,
//                     src_regs: [0; 5],
//                     num_src_regs: 0,
//                     active_mask: 0,
//                     addrs: [0; 32],
//                 };
//
//                 // check that all instructions match
//                 let longest = warp_instructions.iter().map(Vec::len).max().unwrap_or(0);
//
//                 let mut instr_idx = 0;
//                 for i in 0..longest {
//                     let instructions: Vec<_> =
//                         warp_instructions.iter().map(|inst| inst.get(i)).collect();
//                     // assert!(instructions.map(|i| (i.kind, i.size)).all_equal());
//                     // assert!(
//                     //     instructions.windows(2).all(|w| match (w[0], w[1]) {
//                     //         (
//                     //             Some(WarpInstruction::Access(a)),
//                     //             Some(WarpInstruction::Access(b)),
//                     //         ) => a.kind == b.kind && a.size == b.size,
//                     //         _ => false,
//                     //     }),
//                     //     "all threads in a warp need to have equal instructions"
//                     // );
//
//                     assert_eq!(instructions.len(), WARP_SIZE as usize);
//                     let first_valid = instructions.iter().find_map(std::option::Option::as_ref);
//
//                     if let Some(WarpInstruction::Access(access)) = first_valid {
//                         let accesses: Vec<_> = instructions
//                             .iter()
//                             .map(|i| match i {
//                                 Some(WarpInstruction::Access(access)) => Some(access),
//                                 _ => None,
//                             })
//                             .collect();
//
//                         let mut active_mask = crate::warp::ActiveMask::ZERO;
//                         let mut addrs = [0; WARP_SIZE as usize];
//
//                         for (thread_idx, acc) in accesses.iter().enumerate() {
//                             if let Some(acc) = acc {
//                                 active_mask.set(thread_idx, true);
//                                 addrs[thread_idx] = acc.addr;
//                             }
//                         }
//
//                         let is_load = access.kind == MemAccessKind::Load;
//                         let is_store = access.kind == MemAccessKind::Store;
//                         let instr_opcode = match access.mem_space {
//                             MemorySpace::Local if is_load => "LDL".to_string(),
//                             MemorySpace::Global if is_load => "LDG".to_string(),
//                             MemorySpace::Shared if is_load => "LDS".to_string(),
//                             // MemorySpace::Texture if is_load => "LDG".to_string(),
//                             MemorySpace::Constant if is_load => "LDC".to_string(),
//                             MemorySpace::Local if is_store => "STL".to_string(),
//                             MemorySpace::Global if is_store => "STG".to_string(),
//                             MemorySpace::Shared if is_store => "STS".to_string(),
//                             // MemorySpace::Texture if is_store => "LDG".to_string(),
//                             MemorySpace::Constant if is_store => panic!("constant store"),
//                             other => panic!("unknown memory space {other:?}"),
//                         };
//
//                         trace.push(model::MemAccessTraceEntry {
//                             instr_opcode: instr_opcode.to_string(),
//                             instr_is_mem: true,
//                             instr_is_store: is_store,
//                             instr_is_load: is_load,
//                             instr_idx,
//                             active_mask: active_mask.load(),
//                             addrs,
//                             ..warp_instruction.clone()
//                         });
//                         instr_idx += 1;
//                     };
//                 }
//
//                 trace.push(model::MemAccessTraceEntry {
//                     instr_opcode: "EXIT".to_string(),
//                     instr_idx,
//                     active_mask: (!crate::warp::ActiveMask::ZERO).load(),
//                     ..warp_instruction.clone()
//                 });
//
//                 // log::info!("END WARP #{} ({:?})", &warp_id_in_block, &thread_id);
//             }
//         }
//
//         let trace = model::MemAccessTrace(trace);
//         // dbg!(&trace);
//
//         let warp_traces = trace.clone().to_warp_traces();
//         dbg!(&warp_traces[&(model::Dim::ZERO, 0)]
//             .iter()
//             .map(|entry| (&entry.instr_opcode, &entry.active_mask))
//             .collect::<Vec<_>>());
//
//         let launch_config = model::KernelLaunch {
//             name: String::new(),
//             trace_file: String::new(),
//             id: self.kernel_id.fetch_add(1, atomic::Ordering::SeqCst),
//             grid,
//             block: block_size,
//             shared_mem_bytes: 0,
//             num_registers: 0,
//             binary_version: 61,
//             stream_id: 0,
//             shared_mem_base_addr: 0,
//             local_mem_base_addr: 0,
//             nvbit_version: "none".to_string(),
//         };
//         let kernel = Arc::new(crate::kernel::Kernel::new(launch_config, trace));
//         let mut inner = self.inner.lock();
//         inner.kernels.push_back(Arc::clone(&kernel));
//         inner.launch(kernel).unwrap();
//
//         Ok(())
//     }
// }

#[cfg(test)]
mod tests {
    use color_eyre::eyre;
    use num_traits::{Float, Zero};
    // let test2: &(dyn std::ops::IndexMut<usize, Output = T>) = self.d_a;

    #[test]
    fn vectoradd() -> eyre::Result<()> {
        // let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
        // let trace_dir = manifest_dir.join($path);
        // run_lockstep(&trace_dir, TraceProvider::Native)
        Ok(())
    }

    #[test]
    fn matrixmul() -> eyre::Result<()> {
        #[allow(clippy::many_single_char_names)]
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

        // #[derive(Debug)]
        // struct MatrixMul<'s, 'a, T> {
        //     d_a: &'a mut exec::DevicePtr<'s, Vec<T>>,
        //     d_b: &'a mut exec::DevicePtr<'s, Vec<T>>,
        //     d_c: &'a mut exec::DevicePtr<'s, Vec<T>>,
        //     // d_a: &'a mut exec::DevicePtr<'s, 'a, Vec<T>>,
        //     // d_b: &'a mut exec::DevicePtr<'s, 'a, Vec<T>>,
        //     // d_c: &'a mut exec::DevicePtr<'s, 'a, Vec<T>>,
        //     n: usize,
        // }
        //
        // impl<'s, 'a, T> exec::Kernel for MatrixMul<'s, 'a, T>
        // where
        //     T: Float + std::fmt::Debug,
        // {
        //     type Error = std::convert::Infallible;
        //
        //     fn run(&mut self, idx: &exec::ThreadIndex) -> Result<(), Self::Error> {
        //         // Get our global thread ID
        //         // int id = blockIdx.x * blockDim.x + threadIdx.x;
        //         let id: usize = (idx.block_idx.x * idx.block_dim.x + idx.thread_idx.x) as usize;
        //
        //         // Make sure we do not go out of bounds
        //         // if (id < n) c[id] = a[id] + b[id];
        //         if id < self.n {
        //             self.d_c[id] = self.d_a[id] + self.d_b[id];
        //         }
        //         Ok(())
        //     }
        //
        //     fn name(&self) -> &str {
        //         "MatrixMul"
        //     }
        // }

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
        // dbg!(&sim.statslock());
        Ok(())
        // }
        // Ok(())
    }
}
