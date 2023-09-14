use super::kernel::ThreadIndex;
use super::model;
use super::nop::ArithmeticNop;
use super::tracegen::{Error, WARP_SIZE};
use bitvec::field::BitField;
use itertools::Itertools;
use std::sync::{atomic, Arc};
use tokio::sync::{mpsc, Mutex};

#[derive()]
// pub struct DevicePtr<'s, C, T> {
// pub struct DevicePtr<'s, C, T> {
pub struct DevicePtr<T> {
    // pub struct DevicePtr<'s, T, O> {
    // inner: &'a mut T,
    inner: T,
    // marker: std::marker::PhantomData<T>,
    // TODO: refactor this
    // spare: <T as Container>::Elem,
    nop: ArithmeticNop,
    // memory: &'s dyn tracegen::MemoryAccess,
    memory: Arc<dyn MemoryAccess + Send + Sync>,
    mem_space: model::MemorySpace,

    // allocation
    offset: u64,
    // stride: u64,
    // length: u64,
}

impl<T> DevicePtr<T> {
    fn get(&self, thread_idx: &ThreadIndex) -> &T {
        // todo: access at offset and of length stride
        &self.inner
    }

    fn get_mut(&mut self, thread_idx: &ThreadIndex) -> &mut T {
        // todo: access at offset and of length stride
        &mut self.inner
    }
}

impl<T> std::fmt::Debug for DevicePtr<T>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.inner, f)
    }
}

impl<T> std::fmt::Display for DevicePtr<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.inner, f)
    }
}

pub trait Index<Idx: ?Sized> {
    type Output: ?Sized;
    fn index(&self, idx: Idx) -> (&Self::Output, u64, u64);
}

pub trait IndexMut<Idx: ?Sized>: Index<Idx> {
    fn index_mut(&mut self, index: Idx) -> (&mut Self::Output, u64, u64);
}

// TODO: consolidate
impl<'a, T, Idx> Index<Idx> for &'a mut Vec<T>
where
    Idx: super::ToLinear,
{
    type Output = T;
    fn index(&self, idx: Idx) -> (&Self::Output, u64, u64) {
        let idx = idx.to_linear();
        let rel_addr = idx as u64 * self.stride() as u64;
        let size = self.stride() as u64;
        (&self[idx], rel_addr, size)
    }
}

impl<T, Idx> Index<Idx> for Vec<T>
where
    Idx: super::ToLinear,
{
    type Output = T;
    fn index(&self, idx: Idx) -> (&Self::Output, u64, u64) {
        let idx = idx.to_linear();
        let rel_addr = idx as u64 * self.stride() as u64;
        let size = self.stride() as u64;
        (&self[idx], rel_addr, size)
    }
}

// TODO consolidate
impl<'a, T, Idx> IndexMut<Idx> for &'a mut Vec<T>
where
    Idx: super::ToLinear,
{
    fn index_mut(&mut self, idx: Idx) -> (&mut Self::Output, u64, u64) {
        let idx = idx.to_linear();
        let rel_addr = idx as u64 * self.stride() as u64;
        let size = self.stride() as u64;
        (&mut self[idx], rel_addr, size)
    }
}

impl<T, Idx> IndexMut<Idx> for Vec<T>
where
    Idx: super::ToLinear,
{
    fn index_mut(&mut self, idx: Idx) -> (&mut Self::Output, u64, u64) {
        let idx = idx.to_linear();
        let rel_addr = idx as u64 * self.stride() as u64;
        let size = self.stride() as u64;
        (&mut self[idx], rel_addr, size)
    }
}

// impl<T, Idx> Index<Idx> for T where T: std::ops::Index<Idx> {
//     Out
// }

// pub trait IndexMut<I> {
//     type Output;
//     fn index(&self, idx: I) -> (&mut Self::Output, u64);
// }

pub trait Allocatable {
    fn length(&self) -> usize;
    fn stride(&self) -> usize;
    fn size(&self) -> usize {
        self.length() * self.stride()
    }
}

impl<'a, T> Allocatable for &'a mut Vec<T> {
    fn length(&self) -> usize {
        self.len()
    }

    fn stride(&self) -> usize {
        std::mem::size_of::<T>()
    }
}

impl<T> Allocatable for Vec<T> {
    fn length(&self) -> usize {
        self.len()
    }

    fn stride(&self) -> usize {
        std::mem::size_of::<T>()
    }
}

impl<T> std::ops::Index<&ThreadIndex> for DevicePtr<T> {
    type Output = ArithmeticNop;

    fn index(&self, thread_idx: &ThreadIndex) -> &Self::Output {
        println!(
            "inactive: block={} warp={} thread={}",
            &thread_idx.block_id, &thread_idx.warp_id_in_block, &thread_idx.thread_id_in_warp
        );

        self.memory.inactive(thread_idx);
        &self.nop
    }
}

impl<T, Idx> std::ops::Index<(&ThreadIndex, Idx)> for DevicePtr<T>
where
    T: Index<Idx> + std::fmt::Debug,
{
    type Output = <T as Index<Idx>>::Output;

    fn index(&self, (thread_idx, idx): (&ThreadIndex, Idx)) -> &Self::Output {
        // let elem_size = std::mem::size_of::<T::Elem>();
        // let flat_idx = idx.to_linear();
        // let addr = self.offset + elem_size as u64 * flat_idx as u64;
        let (elem, rel_offset, size) = self.inner.index(idx);
        let addr = self.offset + rel_offset as u64;
        // println!("access addr {}", addr);
        self.memory
            .load(thread_idx, addr, size as u32, self.mem_space);
        // log::trace!("{:?}[{:?}] => {}", &self.inner, &idx, &addr);
        // self.inner.index(flat_idx)
        elem
        // &self.inner[idx]
    }
}

impl<T, Idx> std::ops::IndexMut<(&ThreadIndex, Idx)> for DevicePtr<T>
where
    T: IndexMut<Idx> + std::fmt::Debug,
{
    fn index_mut(&mut self, (thread_idx, idx): (&ThreadIndex, Idx)) -> &mut Self::Output {
        let (elem, rel_offset, size) = self.inner.index_mut(idx);
        let addr = self.offset + rel_offset as u64;
        self.memory
            .store(thread_idx, addr, size as u32, self.mem_space);
        // .await;
        elem
    }
}

impl<T> std::ops::IndexMut<&ThreadIndex> for DevicePtr<T> {
    fn index_mut(&mut self, thread_idx: &ThreadIndex) -> &mut Self::Output {
        println!(
            "inactive: block={} warp={} thread={}",
            &thread_idx.block_id, &thread_idx.warp_id_in_block, &thread_idx.thread_id_in_warp
        );

        self.memory.inactive(thread_idx);
        &mut self.nop
    }
}

#[derive(Debug)]
pub struct ThreadBlock {
    barrier: tokio::sync::Barrier,
    id: trace_model::Point,
    size: trace_model::Dim,
}

impl ThreadBlock {
    pub fn new(id: trace_model::Point, size: trace_model::Dim) -> Self {
        Self {
            barrier: tokio::sync::Barrier::new(size.size() as usize),
            id,
            size,
        }
    }

    pub fn id(&self) -> &trace_model::Point {
        &self.id
    }

    pub fn size(&self) -> &trace_model::Dim {
        &self.size
    }

    pub async fn synchronize_threads(&self) {
        self.barrier.wait().await;
    }
}

/// A kernel implementation.
#[async_trait::async_trait]
pub trait Kernel {
    type Error: std::error::Error;

    /// Run an instance of the kernel on a thread identified by its index
    async fn run(&self, block: &ThreadBlock, idx: &ThreadIndex) -> Result<(), Self::Error>;

    fn name(&self) -> &str;
}

// #[async_trait::async_trait]
// pub trait TraceGenerator {
//     type Error;
//
//     /// Trace kernel.
//     async fn trace_kernel<G, B, K>(
//         // &mut self,
//         &self,
//         grid: G,
//         block_size: B,
//         kernel: K,
//     ) -> Result<(), Error<K::Error, Self::Error>>
//     where
//         G: Into<trace_model::Dim>,
//         B: Into<trace_model::Dim>,
//         K: Kernel;
//
//     /// Allocate a variable.
//     // fn allocate<'s, 'a, T>(
//     // fn allocate<'s, T, O>(
//     fn allocate<'s, C, T>(
//         &'s self,
//         var: C,
//         // var: &'a mut T,
//         size: u64,
//         mem_space: model::MemorySpace,
//         // ) -> DevicePtr<'s, 'a, T>;
//         // ) -> DevicePtr<'s, T, O>
//     ) -> DevicePtr<'s, C, T>;
//     // where
//     //     T: Container,
//     //     <T as Container>::Elem: Zero;
// }

#[async_trait::async_trait]
pub trait MemoryAccess {
    fn inactive(&self, thread_idx: &ThreadIndex);

    /// Load address.
    fn load(&self, thread_idx: &ThreadIndex, addr: u64, size: u32, mem_space: model::MemorySpace);

    /// Store address.
    fn store(&self, thread_idx: &ThreadIndex, addr: u64, size: u32, mem_space: model::MemorySpace);
}

use std::collections::HashMap;

#[derive(Debug)]
pub struct Tracer {
    offset: Mutex<u64>,
    // thread_instructions: Mutex<Vec<model::ThreadInstruction>>,
    kernels: std::sync::RwLock<
        Vec<(
            mpsc::Sender<(ThreadIndex, model::ThreadInstruction)>,
            mpsc::Receiver<(ThreadIndex, model::ThreadInstruction)>,
        )>,
    >,
    // rx: Mutex<mpsc::Receiver<(ThreadIndex, model::ThreadInstruction)>>,
    // tx: mpsc::Sender<(ThreadIndex, model::ThreadInstruction)>,
    // thread_instructions: std::sync::Mutex<HashMap<ThreadIndex, Vec<model::ThreadInstruction>>>,
    // (block x warp id) => Vec<thread instructions>
    // thread_instructions:
    //     std::sync::Mutex<Vec<(trace_model::Point, usize, Vec<model::ThreadInstruction>)>>,
    thread_instructions: std::sync::Mutex<
        HashMap<(trace_model::Point, usize), [Vec<model::ThreadInstruction>; WARP_SIZE as usize]>,
    >,
    kernel_launch_id: atomic::AtomicU64,
}

impl Tracer {
    pub fn new() -> Self {
        // let (tx, rx) = mpsc::channel(100);
        Self {
            offset: Mutex::new(0),
            // kernels: Mutex::new(Vec::new()),
            kernels: std::sync::RwLock::new(Vec::new()),
            // rx: Mutex::new(rx),
            // tx,
            thread_instructions: std::sync::Mutex::new(HashMap::new()),
            // thread_instructions: std::sync::Mutex::new(Vec::new()),
            kernel_launch_id: atomic::AtomicU64::new(0),
        }
    }

    fn push_thread_instruction(
        &self,
        block_id: trace_model::Point,
        warp_id_in_block: usize,
        thread_id: usize,
        instruction: model::ThreadInstruction,
    ) {
        let mut block_instructions = self.thread_instructions.lock().unwrap();
        let warp_instructions = block_instructions
            .entry((block_id, warp_id_in_block))
            .or_default();
        warp_instructions[thread_id].push(instruction);
    }
}

#[async_trait::async_trait]
impl MemoryAccess for Tracer {
    fn inactive(&self, thread_idx: &ThreadIndex) {
        // println!(
        //     "inactive: block={} warp={} thread={}",
        //     &thread_idx.block_id, &thread_idx.warp_id_in_block, &thread_idx.thread_id_in_warp
        // );

        self.push_thread_instruction(
            thread_idx.block_id.clone(),
            thread_idx.warp_id_in_block,
            thread_idx.thread_id_in_warp,
            model::ThreadInstruction::Inactive,
        );
    }

    fn load(&self, thread_idx: &ThreadIndex, addr: u64, size: u32, mem_space: model::MemorySpace) {
        let inst = model::ThreadInstruction::Access(model::MemInstruction {
            kind: model::MemAccessKind::Load,
            addr,
            mem_space,
            size,
        });
        self.push_thread_instruction(
            thread_idx.block_id.clone(),
            thread_idx.warp_id_in_block,
            thread_idx.thread_id_in_warp,
            inst,
        );

        // let (tx, rx) = &self.kernels.read().unwrap()[thread_idx.kernel_launch_id];
        // tx.send((thread_idx.clone(), inst));
        // self.thread_instructions
        //     .lock()
        //     .unwrap()
        //     // .await
        //     .entry(thread_idx.clone())
        //     .or_default()
        //     .push(inst);
    }

    fn store(&self, thread_idx: &ThreadIndex, addr: u64, size: u32, mem_space: model::MemorySpace) {
        let inst = model::ThreadInstruction::Access(model::MemInstruction {
            kind: model::MemAccessKind::Store,
            addr,
            mem_space,
            size,
        });
        self.push_thread_instruction(
            thread_idx.block_id.clone(),
            thread_idx.warp_id_in_block,
            thread_idx.thread_id_in_warp,
            inst,
        );

        // let (tx, rx) = &self.kernels.read().unwrap()[thread_idx.kernel_launch_id];
        // tx.send((thread_idx.clone(), inst));

        // self.thread_instructions
        //     .lock()
        //     .unwrap()
        //     // .await
        //     .entry(thread_idx.clone())
        //     .or_default()
        //     .push(inst);
    }
}

// impl<T> StridedAllocation for Vec<T>
// where
//     T: AsRef<[Self::Item]>,
// {
//     type Item;
//     fn offset_at_index() -> usize {}
// }

// pub trait StridedAllocation {
//     type Item;
//     fn offset_at_index() -> usize;
//     // pub fn len() -> usize;
//     // pub fn stride() -> usize;
// }

// #[async_trait::async_trait]
// impl TraceGenerator for Tracer {
impl Tracer {
    // type Error = super::tracegen::TraceError;

    // fn to_device<T, E>(
    //     &self,
    //     value: T,
    //     // size: u64,
    //     // mem_space: model::MemorySpace,
    //     // ) -> DevicePtr<'s, C, T> {
    // ) -> DevicePtr<T>
    // where
    //     T: StridedAllocation<Item = E>,
    // {
    //     std::mem::size_of::<T>();
    //     todo!()
    // }

    // fn allocate<'s, C, T>(
    async fn allocate<T>(
        self: &Arc<Self>,
        value: T,
        // stride: u64,
        // length: u64,
        mem_space: model::MemorySpace,
        // ) -> DevicePtr<'s, C, T> {
    ) -> DevicePtr<T>
    where
        T: Allocatable,
    {
        let mut offset_lock = self.offset.lock().await;
        let offset = *offset_lock;
        *offset_lock += value.size() as u64;

        DevicePtr {
            inner: value,
            mem_space,
            // marker: std::marker::PhantomData,
            nop: ArithmeticNop::default(),
            memory: self.clone(),
            offset,
        }
    }

    #[allow(irrefutable_let_patterns)]
    async fn trace_kernel<G, B, K>(
        self: &Arc<Self>,
        // &self,
        grid: G,
        block_size: B,
        mut kernel: K,
    ) -> Result<(), Error<K::Error, super::tracegen::TraceError>>
    where
        G: Into<trace_model::Dim> + Send,
        B: Into<trace_model::Dim> + Send,
        K: Kernel + Send,
    {
        let kernel_launch_id = self.kernel_launch_id.fetch_add(1, atomic::Ordering::SeqCst);
        self.thread_instructions.lock().unwrap().clear();

        // create a new channel for kernel and launch id combination
        // let tracer_clone = self.clone();
        // tokio::spawn(async move {
        //     let mut rx = tracer_clone.rx.lock().await;
        //     while let Some(msg) = rx.recv().await {
        //         dbg!(msg);
        //     }
        // });

        let grid: trace_model::Dim = grid.into();
        let block_size: trace_model::Dim = block_size.into();

        // let mut trace = Vec::new();
        let kernel = Arc::new(kernel);

        // loop over the grid
        for block_id in grid.clone() {
            println!("block {block_id}");

            // let mut thread_id = ThreadIndex {
            //     kernel_launch_id,
            //     block_idx: block_id.to_dim(),
            //     block_dim: block_size.clone(),
            //     thread_idx: block_size.clone(),
            // };

            // loop over the block size and form warps
            let thread_ids = block_size.clone().into_iter();
            let warp_iter = thread_ids.chunks(WARP_SIZE as usize);
            let thread_iter =
                warp_iter
                    .into_iter()
                    .enumerate()
                    .flat_map(|(warp_id_in_block, threads)| {
                        threads
                            .enumerate()
                            .map(move |(thread_idx, warp_thread_idx)| {
                                (warp_id_in_block, thread_idx, warp_thread_idx)
                            })
                    });

            // for (warp_id_in_block, thread_idx, warp_thread_idx) in thread_iter.iter() {
            //     println!(
            //         "block {block_id} warp #{warp_id_in_block} thread {:?}",
            //         trace_model::Dim::from(warp_thread_idx.clone())
            //     );
            // }
            //
            let block = Arc::new(ThreadBlock::new(block_id.clone(), block_size.clone()));

            let futures = thread_iter.map(|(warp_id_in_block, thread_idx, warp_thread_idx)| {
                // let block_id = block_id.clone();
                // let block_size = block_size.clone();
                let kernel = kernel.clone();
                let block = block.clone();
                async move {
                    println!(
                        "block {} warp #{warp_id_in_block} thread {:?}",
                        block.id(),
                        trace_model::Dim::from(warp_thread_idx.clone())
                    );
                    let thread_id = ThreadIndex {
                        kernel_launch_id,
                        warp_id_in_block,
                        thread_id_in_warp: thread_idx,
                        block_id: block.id().clone(),
                        // thread_id_in_warp: warp_thread_idx.id(),
                        // unique_block_id: block.id().id(),
                        // idx: block.id().id() + warp_thread_idx.id(),
                        block_idx: block.id().to_dim(),
                        block_dim: block.size().clone(),
                        thread_idx: warp_thread_idx.to_dim(),
                        // thread_idx: warp_thread_idx.into(),
                    };

                    kernel.run(&block, &thread_id).await?; // .map_err(Error::Kernel)?;
                    Result::<_, K::Error>::Ok(())
                }
            });

            // use futures::StreamExt;
            futures::future::join_all(futures).await;
            // let stream = futures::stream::iter(futures).buffered(block.size().size() as usize);
            // let results = stream.collect::<Vec<_>>().await;

            // for (warp_id_in_block, threads) in thread_ids
            //     .chunks(WARP_SIZE as usize)
            //     .into_iter()
            //     .enumerate()
            // {
            //     println!("START WARP #{warp_id_in_block}");
            //     let mut thread_instructions = [(); WARP_SIZE as usize].map(|_| Vec::new());
            //
            //     for (thread_idx, warp_thread_idx) in threads.enumerate() {
            //         println!(
            //             "warp #{} thread {:?}",
            //             &warp_id_in_block,
            //             trace_model::Dim::from(warp_thread_idx.clone())
            //         );
            //         thread_id.thread_idx = warp_thread_idx.into();
            //         kernel.run(&thread_id).await.map_err(Error::Kernel)?;
            //         thread_instructions[thread_idx]
            //             .extend(self.thread_instructions.lock().unwrap().drain(..));
            //     }
            // }
        }

        let mut trace = Vec::new();
        let mut traced_instructions = self.thread_instructions.lock().unwrap();

        // for ((block_id, warp_id_in_block), thread_instructions) in traced_instructions.iter() {
        //     println!(
        //         "{block_id} {warp_id_in_block} => {:?}",
        //         thread_instructions
        //             .iter()
        //             .map(|i| i.len())
        //             .collect::<Vec<_>>()
        //     );
        // }
        //
        // return Ok(());

        for ((block_id, warp_id_in_block), thread_instructions) in traced_instructions.drain() {
            assert_eq!(thread_instructions.len(), WARP_SIZE as usize);
            dbg!((&block_id, &warp_id_in_block, &thread_instructions[0].len()));

            if !thread_instructions.iter().all_equal() {
                return Err(Error::Tracer(
                    super::tracegen::TraceError::InconsistentNumberOfWarpInstructions,
                ));
            }

            let warp_instruction = trace_model::MemAccessTraceEntry {
                cuda_ctx: 0,
                sm_id: 0,
                kernel_id: 0,
                block_id: block_id.clone().into(),
                warp_id_in_sm: warp_id_in_block as u32,
                warp_id_in_block: warp_id_in_block as u32,
                warp_size: WARP_SIZE,
                line_num: 0,
                instr_data_width: 0,
                instr_opcode: String::new(),
                instr_offset: 0,
                instr_idx: 0,
                instr_predicate: trace_model::Predicate::default(),
                instr_mem_space: trace_model::MemorySpace::None,
                instr_is_mem: false,
                instr_is_load: false,
                instr_is_store: false,
                instr_is_extended: false,
                dest_regs: [0; 1],
                num_dest_regs: 0,
                src_regs: [0; 5],
                num_src_regs: 0,
                active_mask: 0,
                addrs: [0; 32],
            };

            let num_instructions = thread_instructions[0].len();
            for instr_idx in 0..num_instructions {
                let mut active_mask = trace_model::ActiveMask::ZERO;
                let mut addrs = [0; WARP_SIZE as usize];

                for thread_idx in 0..(WARP_SIZE as usize) {
                    let thread_instruction = &thread_instructions[thread_idx][instr_idx];
                    if let model::ThreadInstruction::Access(ref access) = thread_instruction {
                        active_mask.set(thread_idx, true);
                        addrs[thread_idx] = access.addr;
                    }
                }

                // first_thread_instruction
                if let model::ThreadInstruction::Access(ref access) =
                    thread_instructions[0][instr_idx]
                {
                    let is_load = access.kind == model::MemAccessKind::Load;
                    let is_store = access.kind == model::MemAccessKind::Store;
                    let instr_opcode = match access.mem_space {
                        model::MemorySpace::Local if is_load => "LDL".to_string(),
                        model::MemorySpace::Global if is_load => "LDG".to_string(),
                        model::MemorySpace::Shared if is_load => "LDS".to_string(),
                        // MemorySpace::Texture if is_load => "LDG".to_string(),
                        model::MemorySpace::Constant if is_load => "LDC".to_string(),
                        model::MemorySpace::Local if is_store => "STL".to_string(),
                        model::MemorySpace::Global if is_store => "STG".to_string(),
                        model::MemorySpace::Shared if is_store => "STS".to_string(),
                        // MemorySpace::Texture if is_store => "LDG".to_string(),
                        model::MemorySpace::Constant if is_store => panic!("constant store"),
                        other => panic!("unknown memory space {other:?}"),
                    };

                    trace.push(trace_model::MemAccessTraceEntry {
                        instr_opcode: instr_opcode.to_string(),
                        instr_is_mem: true,
                        instr_is_store: is_store,
                        instr_is_load: is_load,
                        instr_idx: instr_idx as u32,
                        active_mask: active_mask.load(),
                        addrs,
                        ..warp_instruction.clone()
                    });
                }
            }

            // EXIT instruction
            trace.push(trace_model::MemAccessTraceEntry {
                instr_opcode: "EXIT".to_string(),
                instr_idx: num_instructions as u32,
                active_mask: (!trace_model::ActiveMask::ZERO).load(),
                ..warp_instruction.clone()
            });
        }

        let trace = trace_model::MemAccessTrace(trace);
        let warp_traces = trace.clone().to_warp_traces();

        dbg!(&warp_traces[&(trace_model::Dim::ZERO, 0)]
            .iter()
            .map(|entry| (&entry.instr_opcode, &entry.active_mask))
            .collect::<Vec<_>>());

        let launch_config = trace_model::command::KernelLaunch {
            mangled_name: kernel.name().to_string(),
            unmangled_name: kernel.name().to_string(),
            trace_file: String::new(),
            id: kernel_launch_id,
            grid,
            block: block_size,
            shared_mem_bytes: 0,
            num_registers: 0,
            binary_version: 61,
            stream_id: 0,
            shared_mem_base_addr: 0,
            local_mem_base_addr: 0,
            nvbit_version: "none".to_string(),
        };
        dbg!(launch_config);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use color_eyre::eyre;
    use num_traits::Float;
    use std::sync::Arc;
    use tokio::sync::Mutex;
    // use utils::diff;
    // use std::path::PathBuf;
    // use trace_model as model;
    // use gpucachesim::exec::{self, MemorySpace, TraceGenerator, Tracer};

    #[derive(Debug)]
    // struct VecAdd<'s, T> {
    struct VecAdd<'a, T> {
        // dev_a: super::DevicePtr<'s, Vec<T>, T>,
        // dev_a: tokio::sync::Mutex<&mut super::DevicePtr<Vec<T>>>,
        dev_a: tokio::sync::Mutex<super::DevicePtr<&'a mut Vec<T>>>,
        dev_b: tokio::sync::Mutex<super::DevicePtr<Vec<T>>>,
        dev_result: tokio::sync::Mutex<super::DevicePtr<Vec<T>>>,
        // marker: std::marker::PhantomData<&'s T>,
        n: usize,
    }

    #[async_trait::async_trait]
    // impl<'s, T> super::Kernel for VecAdd<'s, T>
    // impl<T> super::Kernel for VecAdd<T>
    impl<'a, T> super::Kernel for VecAdd<'a, T>
    where
        T: Float + std::fmt::Debug + Send + Sync,
    {
        type Error = std::convert::Infallible;

        async fn run(
            &self,
            block: &super::ThreadBlock,
            tid: &super::ThreadIndex,
        ) -> Result<(), Self::Error> {
            let idx = (tid.block_idx.x * tid.block_dim.x + tid.thread_idx.x) as usize;

            // println!("thread {:?} before", idx);
            // block.synchronize_threads().await;
            // println!("thread {:?} after", idx);
            // block.synchronize_threads().await;
            // println!("thread {:?} after after", idx);

            let dev_a = self.dev_a.lock().await;
            let dev_b = self.dev_b.lock().await;
            let mut dev_result = self.dev_result.lock().await;

            // block.access(self.dev_a, )
            if idx < self.n {
                // self.dev_result[()] = self.dev_a[()] + self.dev_b[()];
                // self.dev_result[(tid, idx)] = self.dev_a[(tid, idx)] + self.dev_b[(tid, idx)];
                dev_result[(tid, idx)] = dev_a[(tid, idx)] + dev_b[(tid, idx)];
            } else {
                dev_result[tid] = dev_a[tid] + dev_b[tid];
            }
            Ok(())
        }

        fn name(&self) -> &str {
            "VecAdd"
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_it_works() -> eyre::Result<()> {
        let block_size = 64;
        // let n = 128;
        let n = 120;
        let mut tracer = Arc::new(super::Tracer::new());

        let mut a: Vec<f32> = vec![0.0; n];
        let mut b: Vec<f32> = vec![0.0; n];
        let mut result: Vec<f32> = vec![0.0; n];
        // let a_size = a.len() * std::mem::size_of::<f32>();
        // let b_size = b.len() * std::mem::size_of::<T>();
        // let c_size = c.len() * std::mem::size_of::<T>();
        // let mut dev_a = tracer.allocate(&mut a, a_size as u64, crate::MemorySpace::Global);
        // let mut dev_a = tracer.allocate(a, a_size as u64, crate::MemorySpace::Global);
        let mut dev_a = tracer.allocate(&mut a, crate::MemorySpace::Global).await;
        let mut dev_b = tracer.allocate(b, crate::MemorySpace::Global).await;
        let mut dev_result = tracer.allocate(result, crate::MemorySpace::Global).await;

        let kernel: VecAdd<f32> = VecAdd {
            dev_a: Mutex::new(dev_a),
            dev_b: Mutex::new(dev_b),
            dev_result: Mutex::new(dev_result),
            // marker: std::marker::PhantomData,
            n,
        };
        let grid_size = (n as f64 / block_size as f64).ceil() as u32;
        tracer.trace_kernel(grid_size, block_size, kernel).await?;
        assert!(false);
        Ok(())
    }

    #[test]
    fn test_size_of() {
        assert_eq!(std::mem::size_of::<f32>(), 4);
        assert_eq!(std::mem::size_of::<Vec<f32>>(), 24);
        let v = vec![0.0f32; 4];
        assert_eq!(std::mem::size_of_val(&v), 24);
        let v = [0.0f32; 4];
        assert_eq!(std::mem::size_of_val(&v), 4 * 4);
        let v = [0.0f32; 8];
        assert_eq!(std::mem::size_of_val(&v), 4 * 8);
    }
}
