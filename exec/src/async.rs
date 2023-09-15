use crate::model::MemInstruction;

use super::kernel::ThreadIndex;
use super::model::{self, ThreadInstruction};
use super::nop::ArithmeticNop;
use bitvec::field::BitField;
use futures::{future::Future, StreamExt};
use itertools::Itertools;
use petgraph::visit::{VisitMap, Visitable};
use std::collections::{HashMap, VecDeque};
use std::pin::Pin;
use std::sync::{atomic, Arc};
use tokio::sync::{mpsc, Mutex};
use trace_model::ToBitString;

pub const DEV_GLOBAL_HEAP_START: u64 = 0xC000_0000;
pub const WARP_SIZE: u32 = 32;

#[derive(thiserror::Error, Debug)]
pub enum Error<K, T> {
    #[error(transparent)]
    Kernel(K),
    #[error(transparent)]
    Tracer(T),
}

#[derive(thiserror::Error, Debug)]
pub enum TraceError {
    #[error("inconsistent number of warp instructions")]
    InconsistentNumberOfWarpInstructions,

    #[error("missing reconvergence points")]
    MissingReconvergencePoints,
}

#[derive(Clone)]
pub struct DevicePtr<T> {
    pub inner: T,
    nop: ArithmeticNop,
    memory: Arc<dyn MemoryAccess + Send + Sync>,
    mem_space: model::MemorySpace,
    offset: u64,
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

impl<'a, T, Idx> Index<Idx> for &'a Vec<T>
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

impl<'a, T> Allocatable for &'a Vec<T> {
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

// impl<T> std::ops::Index<&ThreadIndex> for DevicePtr<T> {
//     type Output = ArithmeticNop;
//
//     fn index(&self, thread_idx: &ThreadIndex) -> &Self::Output {
//         // println!(
//         //     "inactive: block={} warp={} thread={}",
//         //     &thread_idx.block_id, &thread_idx.warp_id_in_block, &thread_idx.thread_id_in_warp
//         // );
//         self.memory.inactive(thread_idx);
//         &self.nop
//     }
// }

impl<T, Idx> std::ops::Index<(&ThreadIndex, Idx)> for DevicePtr<T>
where
    T: Index<Idx>, //  + std::fmt::Debug,
{
    type Output = <T as Index<Idx>>::Output;

    fn index(&self, (thread_idx, idx): (&ThreadIndex, Idx)) -> &Self::Output {
        let (elem, rel_offset, size) = self.inner.index(idx);
        let addr = self.offset + rel_offset as u64;
        self.memory
            .load(thread_idx, addr, size as u32, self.mem_space);
        elem
    }
}

impl<T, Idx> std::ops::IndexMut<(&ThreadIndex, Idx)> for DevicePtr<T>
where
    T: IndexMut<Idx>, //  + std::fmt::Debug,
{
    fn index_mut(&mut self, (thread_idx, idx): (&ThreadIndex, Idx)) -> &mut Self::Output {
        let (elem, rel_offset, size) = self.inner.index_mut(idx);
        let addr = self.offset + rel_offset as u64;
        self.memory
            .store(thread_idx, addr, size as u32, self.mem_space);
        elem
    }
}

// impl<T> std::ops::IndexMut<&ThreadIndex> for DevicePtr<T> {
//     fn index_mut(&mut self, thread_idx: &ThreadIndex) -> &mut Self::Output {
//         // println!(
//         //     "inactive: block={} warp={} thread={}",
//         //     &thread_idx.block_id, &thread_idx.warp_id_in_block, &thread_idx.thread_id_in_warp
//         // );
//         self.memory.inactive(thread_idx);
//         &mut self.nop
//     }
// }

pub struct ThreadBlock {
    pub(crate) barrier: Arc<tokio::sync::Barrier>,
    pub memory: Arc<dyn MemoryAccess + Send + Sync>,
    // id: trace_model::Point,
    // size: trace_model::Dim,
    pub thread_id: ThreadIndex,
}

impl ThreadBlock {
    // pub fn new(id: trace_model::Point, size: trace_model::Dim) -> Self {
    // pub fn new(thread_id: ThreadIndex, memory: Arc<dyn MemoryAccess + Send + Sync>) -> Self {
    //     let block_size =
    //     Self {
    //         barrier: Arc::new(tokio::sync::Barrier::new(size.size() as usize)),
    //         thread_id,
    //         memory,
    //         // id,
    //         // size,
    //     }
    // }

    // pub fn id(&self) -> &trace_model::Point {
    //     &self.id
    // }
    //
    // pub fn size(&self) -> &trace_model::Dim {
    //     &self.size
    // }

    pub async fn synchronize_threads(&self) {
        self.barrier.wait().await;
    }

    pub fn reconverge_branch(&self, branch_id: usize) {
        let inst = model::ThreadInstruction::Reconverge(branch_id);
        self.memory.push_thread_instruction(&self.thread_id, inst);
    }

    pub fn took_branch(&self, branch_id: usize) {
        let inst = model::ThreadInstruction::TookBranch(branch_id);
        self.memory.push_thread_instruction(&self.thread_id, inst);
        // self.memory.reconverge(&self.thread_id, branch_id);
    }

    pub fn start_branch(&self, branch_id: usize) {
        let inst = model::ThreadInstruction::Branch(branch_id);
        self.memory.push_thread_instruction(&self.thread_id, inst);
        // self.memory.start_branch(&self.thread_id, branch_id);
    }
}

/// A kernel implementation.
#[async_trait::async_trait]
pub trait Kernel {
    type Error: std::error::Error;

    /// Run an instance of the kernel on a thread identified by its index
    async fn run(&self, block: &ThreadBlock, idx: &ThreadIndex) -> Result<(), Self::Error>;
    // fn run(&self, block: &ThreadBlock, idx: &ThreadIndex) -> Pin<Box<dyn Future<Output = Result<(), Self::Error>> + Send + '_>>;

    fn name(&self) -> &str;
}

#[async_trait::async_trait]
pub trait TraceGenerator {
    type Error;

    /// Trace kernel.
    async fn trace_kernel<G, B, K>(
        self: &Arc<Self>,
        grid: G,
        block_size: B,
        mut kernel: K,
    ) -> Result<(), Error<K::Error, Self::Error>>
    where
        G: Into<trace_model::Dim> + Send,
        B: Into<trace_model::Dim> + Send,
        K: Kernel + Send + Sync,
        <K as Kernel>::Error: Send;

    /// Allocate a variable.
    async fn allocate<T>(self: &Arc<Self>, value: T, mem_space: model::MemorySpace) -> DevicePtr<T>
    where
        T: Allocatable + Send;
}

#[async_trait::async_trait]
pub trait MemoryAccess {
    /// Push thread instruction
    fn push_thread_instruction(
        &self,
        thread_idx: &ThreadIndex,
        // block_id: trace_model::Point,
        // warp_id_in_block: usize,
        // thread_id: usize,
        instruction: model::ThreadInstruction,
    );

    /// Load address.
    fn load(&self, thread_idx: &ThreadIndex, addr: u64, size: u32, mem_space: model::MemorySpace);

    /// Store address.
    fn store(&self, thread_idx: &ThreadIndex, addr: u64, size: u32, mem_space: model::MemorySpace);
}

// #[derive(Debug)]
pub struct Tracer {
    offset: Mutex<u64>,
    thread_instructions: std::sync::Mutex<
        HashMap<(trace_model::Point, usize), [Vec<model::ThreadInstruction>; WARP_SIZE as usize]>,
    >,
    kernel_launch_id: atomic::AtomicU64,
}

impl Tracer {
    pub fn new() -> Arc<Self> {
        // let (tx, rx) = mpsc::channel(100);
        Arc::new(Self {
            offset: Mutex::new(0),
            thread_instructions: std::sync::Mutex::new(HashMap::new()),
            kernel_launch_id: atomic::AtomicU64::new(0),
        })
    }
}

#[async_trait::async_trait]
impl MemoryAccess for Tracer {
    fn push_thread_instruction(
        &self,
        // block_id: trace_model::Point,
        // warp_id_in_block: usize,
        // thread_id: usize,
        thread_idx: &ThreadIndex,
        instruction: model::ThreadInstruction,
    ) {
        let block_id = thread_idx.block_id.clone();
        let warp_id_in_block = thread_idx.warp_id_in_block;
        let thread_id = thread_idx.thread_id_in_warp;

        let mut block_instructions = self.thread_instructions.lock().unwrap();
        let warp_instructions = block_instructions
            .entry((block_id, warp_id_in_block))
            .or_default();
        warp_instructions[thread_id].push(instruction);
    }

    fn load(&self, thread_idx: &ThreadIndex, addr: u64, size: u32, mem_space: model::MemorySpace) {
        let inst = model::ThreadInstruction::Access(model::MemInstruction {
            kind: model::MemAccessKind::Load,
            addr,
            mem_space,
            size,
        });
        self.push_thread_instruction(
            &thread_idx,
            // thread_idx.block_id.clone(),
            // thread_idx.warp_id_in_block,
            // thread_idx.thread_id_in_warp,
            inst,
        );
    }

    fn store(&self, thread_idx: &ThreadIndex, addr: u64, size: u32, mem_space: model::MemorySpace) {
        let inst = model::ThreadInstruction::Access(model::MemInstruction {
            kind: model::MemAccessKind::Store,
            addr,
            mem_space,
            size,
        });
        self.push_thread_instruction(
            &thread_idx,
            // thread_idx.block_id.clone(),
            // thread_idx.warp_id_in_block,
            // thread_idx.thread_id_in_warp,
            inst,
        );
    }
}

#[async_trait::async_trait]
impl TraceGenerator for Tracer {
    type Error = TraceError;

    async fn allocate<T>(self: &Arc<Self>, value: T, mem_space: model::MemorySpace) -> DevicePtr<T>
    where
        T: Allocatable + Send,
    {
        let mut offset_lock = self.offset.lock().await;
        let offset = *offset_lock;
        *offset_lock += value.size() as u64;

        DevicePtr {
            inner: value,
            mem_space,
            nop: ArithmeticNop::default(),
            memory: self.clone(),
            offset,
        }
    }

    async fn trace_kernel<G, B, K>(
        self: &Arc<Self>,
        grid: G,
        block_size: B,
        mut kernel: K,
    ) -> Result<(), Error<K::Error, Self::Error>>
    where
        G: Into<trace_model::Dim> + Send,
        B: Into<trace_model::Dim> + Send,
        K: Kernel + Send + Sync,
        <K as Kernel>::Error: Send,
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

        let kernel = Arc::new(kernel);

        let block_ids: Vec<_> = grid.clone().into_iter().collect();
        println!("launching {} blocks", block_ids.len());

        // loop over the grid
        futures::stream::iter(block_ids)
            .then(|block_id| {
                let block_size = block_size.clone();
                let block_id = block_id.clone();
                let kernel = kernel.clone();
                async move {
                    // println!("block {block_id}");

                    // loop over the block size and form warps
                    let thread_ids = block_size.clone().into_iter();
                    let warp_iter = thread_ids.chunks(WARP_SIZE as usize);
                    let thread_iter: Vec<(_, _, _)> = warp_iter
                        .into_iter()
                        .enumerate()
                        .flat_map(|(warp_id_in_block, threads)| {
                            threads
                                .enumerate()
                                .map(move |(thread_idx, warp_thread_idx)| {
                                    (warp_id_in_block, thread_idx, warp_thread_idx)
                                })
                        })
                        .collect();

                    let block_barrier =
                        Arc::new(tokio::sync::Barrier::new(block_size.size() as usize));
                    let memory = self.clone();
                    // let block = ThreadBlock {
                    //     barrier: ,
                    //     memory: self.clone(),
                    //     thread_id: ThreadIndex::default(),
                    // };
                    // let block = Arc::new(ThreadBlock::new(block_id.clone(), block_size.clone()));

                    let futures = thread_iter.into_iter().map(
                        |(warp_id_in_block, thread_idx, warp_thread_idx)| {
                            let kernel = kernel.clone();
                            let barrier = block_barrier.clone();
                            let memory = memory.clone();
                            let block_size = block_size.clone();
                            let block_id = block_id.clone();
                            async move {
                                // println!(
                                //     "block {} warp #{warp_id_in_block} thread {:?}",
                                //     block.id(),
                                //     trace_model::Dim::from(warp_thread_idx.clone())
                                // );
                                let thread_id = ThreadIndex {
                                    kernel_launch_id,
                                    warp_id_in_block,
                                    thread_id_in_warp: thread_idx,
                                    block_id: block_id.clone(),
                                    block_idx: block_id.to_dim(),
                                    block_dim: block_size.clone(),
                                    thread_idx: warp_thread_idx.to_dim(),
                                };

                                let block = ThreadBlock {
                                    barrier,
                                    memory,
                                    thread_id: thread_id.clone(),
                                };

                                kernel.run(&block, &thread_id).await?;
                                Result::<_, K::Error>::Ok(())
                            }
                        },
                    );

                    futures::future::join_all(futures).await;
                }
            })
            // .buffer_unordered(4)
            .collect::<Vec<_>>()
            .await;

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

        // check for reconvergence point
        if !traced_instructions.values().all(|warp_instructions| {
            warp_instructions.iter().all(|thread_instructions| {
                thread_instructions[0] == ThreadInstruction::TookBranch(0)
            })
        }) {
            return Err(Error::Tracer(TraceError::MissingReconvergencePoints));
        }

        // todo this needs to be a while / loop

        // for ((block_id, warp_id_in_block), warp_instructions) in traced_instructions.drain() {
        let traced_instructions_vec: Vec<_> = traced_instructions.clone().into_iter().collect();
        for ((block_id, warp_id_in_block), warp_instructions) in
            traced_instructions_vec.into_iter().rev()
        {
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

            // let mut thread_pos = [0; WARP_SIZE as usize];
            // for (ti, thread_instructions) in warp_instructions.iter().enumerate() {
            //     println!(
            //         "thread[{:2}] = {:?}",
            //         ti,
            //         &thread_instructions
            //             .iter()
            //             .map(ToString::to_string)
            //             .collect::<Vec<_>>()
            //     );
            // }

            use petgraph::{algo, prelude::*};
            use std::collections::HashSet;

            #[derive(Debug)]
            enum TraceNode {
                Branch {
                    id: usize,
                    instructions: Vec<MemInstruction>,
                },
                Reconverge {
                    id: usize,
                    instructions: Vec<MemInstruction>,
                },
            }

            impl TraceNode {
                pub fn id(&self) -> usize {
                    match self {
                        Self::Branch { id, .. } | Self::Reconverge { id, .. } => *id,
                    }
                }
            }

            #[derive(Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
            enum CFGNode {
                Branch(usize),
                Reconverge(usize),
            }

            impl CFGNode {
                pub fn id(&self) -> usize {
                    match self {
                        Self::Branch(id) | Self::Reconverge(id) => *id,
                    }
                }
            }

            pub trait UniqueGraph<N, E, Ix> {
                fn add_unique_edge(
                    &mut self,
                    a: NodeIndex<Ix>,
                    b: NodeIndex<Ix>,
                    weight: E,
                ) -> EdgeIndex<Ix>
                where
                    E: PartialEq;

                fn find_node(&self, weight: &N) -> Option<NodeIndex<Ix>>
                where
                    N: PartialEq;

                fn add_unique_node(&mut self, weight: N) -> NodeIndex<Ix>
                where
                    N: PartialEq;
            }

            impl<N, E, D, Ix> UniqueGraph<N, E, Ix> for Graph<N, E, D, Ix>
            where
                D: petgraph::EdgeType,
                Ix: petgraph::graph::IndexType,
            {
                fn add_unique_edge(
                    &mut self,
                    a: NodeIndex<Ix>,
                    b: NodeIndex<Ix>,
                    weight: E,
                ) -> EdgeIndex<Ix>
                where
                    E: PartialEq,
                {
                    match self.find_edge(a, b) {
                        Some(edge) if self.edge_weight(edge) == Some(&weight) => edge,
                        _ => Graph::<N, E, D, Ix>::add_edge(self, a, b, weight),
                    }
                }

                fn find_node(&self, weight: &N) -> Option<NodeIndex<Ix>>
                where
                    N: PartialEq,
                {
                    self.node_indices()
                        .find(|idx| self.node_weight(*idx) == Some(weight))
                }

                fn add_unique_node(&mut self, weight: N) -> NodeIndex<Ix>
                where
                    N: PartialEq,
                {
                    match self.find_node(&weight) {
                        Some(idx) => idx,
                        _ => Graph::<N, E, D, Ix>::add_node(self, weight),
                    }
                }
            }

            // let mut super_nodes = std::collections::HashSet::new();
            // let mut super_cfg: &dyn UniqueGraph<_, _, _> = &DiGraph::<CFGNode, bool>::new();
            let mut super_cfg = DiGraph::<CFGNode, bool>::new();
            let mut super_cfg_root_idx = super_cfg.add_unique_node(CFGNode::Reconverge(0));

            let mut thread_graphs = Vec::with_capacity(WARP_SIZE as usize);
            for (ti, thread_instructions) in warp_instructions.iter().enumerate() {
                let mut cfg = DiGraph::<TraceNode, bool>::new();
                let mut took_branch = false;
                let mut last_super_node_idx = super_cfg_root_idx;
                let mut last_node_idx = cfg.add_node(TraceNode::Reconverge {
                    id: 0,
                    instructions: vec![],
                });
                let mut current_instructions = Vec::new();

                for thread_instruction in thread_instructions {
                    match thread_instruction {
                        ThreadInstruction::Nop => unreachable!(),
                        ThreadInstruction::TookBranch(branch_id) => {
                            took_branch = true;
                        }
                        ThreadInstruction::Branch(branch_id) => {
                            took_branch = false;
                            {
                                let super_node_idx =
                                    super_cfg.add_unique_node(CFGNode::Branch(*branch_id));
                                super_cfg.add_unique_edge(
                                    last_super_node_idx,
                                    super_node_idx,
                                    true,
                                );
                                last_super_node_idx = super_node_idx;
                            }
                            {
                                let node_idx = cfg.add_node(TraceNode::Branch {
                                    id: *branch_id,
                                    instructions: current_instructions.drain(..).collect(),
                                });
                                assert!(!matches!(cfg[last_node_idx], TraceNode::Branch { .. }));
                                cfg.add_edge(last_node_idx, node_idx, true);
                                last_node_idx = node_idx;
                            }
                            // current_instructions.clear();
                        }
                        ThreadInstruction::Reconverge(branch_id) => {
                            {
                                let super_node_idx =
                                    super_cfg.add_unique_node(CFGNode::Reconverge(*branch_id));
                                super_cfg.add_unique_edge(
                                    last_super_node_idx,
                                    super_node_idx,
                                    took_branch,
                                );
                                last_super_node_idx = super_node_idx;
                            }
                            {
                                let node_idx = cfg.add_node(TraceNode::Reconverge {
                                    id: *branch_id,
                                    instructions: current_instructions.drain(..).collect(),
                                    // instructions: vec![],
                                });
                                cfg.add_edge(last_node_idx, node_idx, took_branch);
                                assert_eq!(cfg[last_node_idx].id(), cfg[node_idx].id());
                                last_node_idx = node_idx;
                            }
                        }
                        ThreadInstruction::Access(access) => {
                            current_instructions.push(access.clone());
                        }
                    }
                }

                // each edge connects two distinct nodes, as each thread takes
                // a single control flow path
                assert_eq!(cfg.node_count(), cfg.edge_count() + 1);
                assert_eq!(cfg.edge_count(), cfg.edge_weights().count());
                // dbg!(&cfg);
                // println!("{:?}", Dot::with_config(&graph, &[Config::EdgeNoLabel]));
                // let mut f = File::create("example1.dot").unwrap();
                // let output = format!("{}", Dot::with_config(&graph, &[Config::EdgeNoLabel]));
                // f.write_all(&output.as_bytes())

                println!(
                    "thread[{:2}] = {:?} edge weights: {:?}",
                    ti,
                    &thread_instructions
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>(),
                    cfg.edge_weights().collect::<Vec<_>>(),
                );

                thread_graphs.push(cfg);
            }

            // fill remaining edges
            for node_idx in super_cfg.node_indices() {
                let CFGNode::Branch(branch_id) = super_cfg[node_idx] else {
                    continue;
                };
                let edges: HashSet<bool> = super_cfg
                    .edges(node_idx)
                    .map(|edge| edge.weight())
                    .copied()
                    .collect();
                assert!(!edges.is_empty());
                assert!(edges.len() <= 2);
                let reconvergence_node_idx = super_cfg
                    .find_node(&CFGNode::Reconverge(branch_id))
                    .unwrap();
                if !edges.contains(&true) {
                    super_cfg.add_unique_edge(node_idx, reconvergence_node_idx, true);
                }
                if !edges.contains(&false) {
                    super_cfg.add_unique_edge(node_idx, reconvergence_node_idx, false);
                }
            }

            println!(
                "super cfg: {} nodes, {} edges, {} edge weights",
                super_cfg.node_count(),
                super_cfg.edge_count(),
                super_cfg.edge_weights().count(),
            );

            let super_cfg_final_reconvergence_id = super_cfg
                .node_indices()
                .filter_map(|idx| match super_cfg[idx] {
                    CFGNode::Reconverge(branch_id) => Some(branch_id),
                    _ => None,
                })
                .max()
                .unwrap_or(0);
            let super_cfg_sink_idx = super_cfg
                .find_node(&CFGNode::Reconverge(super_cfg_final_reconvergence_id))
                .unwrap();

            let ways = algo::all_simple_paths::<Vec<_>, _>(
                &super_cfg,
                super_cfg_root_idx,
                super_cfg_sink_idx,
                0,
                None,
            )
            .collect::<Vec<_>>();
            dbg!(ways.len());

            // BFS start here
            // let mut discovered = super_cfg.visit_map();
            // discovered.visit(super_cfg_root_idx);
            let mut dominators = VecDeque::new();
            assert!(matches!(
                super_cfg[super_cfg_root_idx],
                CFGNode::Reconverge(..)
            ));
            // let mut edges = super_cfg
            //     .neighbors_directed(super_cfg_root_idx, Outgoing)
            // .detach();
            // while let Some(child) = edges.next(&super_cfg) {
            //     dominators.push_front(child);
            // }
            dominators.extend(super_cfg.neighbors_directed(super_cfg_root_idx, Outgoing));
            dbg!(&dominators);

            while let Some(dominator_node_idx) = dominators.pop_front() {
                // println!("current dominator: {:?}", super_cfg[dominator_node_idx]);

                assert!(matches!(super_cfg[dominator_node_idx], CFGNode::Branch(..)));

                // let reconvergence_point = CFGNode::Reconverge(super_cfg[dominator_node_idx].id());
                let reconvergence_node = CFGNode::Reconverge(super_cfg[dominator_node_idx].id());
                let reconvergence_node_idx = super_cfg.find_node(&reconvergence_node).unwrap();

                // nested bfs
                let mut stack = VecDeque::new();
                let mut edges = super_cfg
                    .neighbors_directed(dominator_node_idx, Outgoing)
                    .detach();
                while let Some(child) = edges.next(&super_cfg) {
                    stack.push_front(child);
                }

                println!(
                    "current dominator: {:?} stack = {:?}",
                    super_cfg[dominator_node_idx], &stack
                );

                while let Some((edge_idx, node_idx)) = stack.pop_front() {
                    // use a detached neighbors walker

                    if node_idx == reconvergence_node_idx {
                        // stop here, never go beyond the domninators reconvergence point
                        println!("stop: found reconvergence point {:?}", reconvergence_node);
                        continue;
                    }

                    let mut edges = super_cfg.neighbors_directed(node_idx, Outgoing).detach();
                    while let Some((outgoing_edge_idx, next_node_idx)) = edges.next(&super_cfg) {
                        // stack.push_back((next_node, outgoing_edge));
                        // stack.push_back((super_cfg[outgoing_edge], outgoing_edge));
                        // stack.push_back((super_cfg[edge], edge.));

                        println!(
                            "for dominator={:?} \t {:?} --> taken={} --> {:?}",
                            super_cfg[dominator_node_idx],
                            super_cfg[node_idx],
                            super_cfg[edge_idx],
                            super_cfg[next_node_idx],
                        );
                        stack.push_back((outgoing_edge_idx, next_node_idx));
                    }
                }

                println!("dominator {:?} completed", super_cfg[dominator_node_idx]);

                // // use a detached neighbors walker
                // let mut edges = super_cfg.neighbors_directed(node, Outgoing).detach();
                // while let Some((outgoing_edge, next_node)) = edges.next(&super_cfg) {
                //     stack.push_back((next_node, outgoing_edge));
                //     // stack.push_back((super_cfg[outgoing_edge], outgoing_edge));
                //     // stack.push_back((super_cfg[edge], edge.));
                //     // gr[node] += gr[edge];
                // }
                // // for succ in super_cfg.neighbors(node) {
                // //     // if discovered.visit(succ) {
                // //     stack.push_back((succ, ));
                // //     // }
                // // }
                // // return Some(node);
            }

            // if false {
            //     let mut stack = VecDeque::new();
            //     let mut edges = super_cfg
            //         .neighbors_directed(super_cfg_root_idx, Outgoing)
            //         .detach();
            //     let edges =
            //         stack.push_front((super_cfg_root_idx, edges.next_edge(&super_cfg).unwrap()));
            //
            //     while let Some((node, edge)) = stack.pop_front() {
            //         println!("current node: {:?}", &node);
            //         // use a detached neighbors walker
            //         let mut edges = super_cfg.neighbors_directed(node, Outgoing).detach();
            //         while let Some((outgoing_edge, next_node)) = edges.next(&super_cfg) {
            //             stack.push_back((next_node, outgoing_edge));
            //             // stack.push_back((super_cfg[outgoing_edge], outgoing_edge));
            //             // stack.push_back((super_cfg[edge], edge.));
            //             // gr[node] += gr[edge];
            //         }
            //         // for succ in super_cfg.neighbors(node) {
            //         //     // if discovered.visit(succ) {
            //         //     stack.push_back((succ, ));
            //         //     // }
            //         // }
            //         // return Some(node);
            //     }
            // }

            // let mut bfs = Bfs::new(&super_cfg, super_cfg_root_idx);
            // while let Some(nx) = bfs.next(&super_cfg) {
            //     dbg!(&super_cfg[nx]);
            // }

            break;
        }

        return Ok(());

        // println!(
        //     "edges: {:#?}",
        //     super_cfg
        //         .node_indices()
        //         .map(|node| super_cfg.edges(node).collect::<Vec<_>>())
        //         .collect::<Vec<_>>(),
        // );
        //
        // let mut threads_in_branch = Vec::with_capacity(WARP_SIZE as usize);
        //             threads_in_branch.extend(0..WARP_SIZE as usize);
        //
        //             for rec_point_id in 0..1 {
        //                 // compute active mask for the current branch
        //                 let mut active_mask = trace_model::ActiveMask::ZERO;
        //                 for tid in threads_in_branch.iter() {
        //                     if let Some(ThreadInstruction::TookBranch(branch_id)) =
        //                         warp_instructions[*tid].get(0)
        //                     {
        //                         assert_eq!(*branch_id, rec_point_id);
        //                         active_mask.set(*tid, true);
        //                     }
        //                 }
        //                 println!(
        //                     "rec point {:3} \t active={}",
        //                     rec_point_id,
        //                     active_mask.to_bit_string()
        //                 );
        //             }

        for ((block_id, warp_id_in_block), warp_instructions) in traced_instructions.drain() {
            // assert_eq!(thread_instructions.len(), WARP_SIZE as usize);
            // dbg!((&block_id, &warp_id_in_block, &thread_instructions[0].len()));

            // todo
            // if !thread_instructions.iter().all_equal() {
            //     return Err(Error::Tracer(
            //         super::tracegen::TraceError::InconsistentNumberOfWarpInstructions,
            //     ));
            // }

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

            let num_instructions = warp_instructions[0].len();
            for instr_idx in 0..num_instructions {
                let mut active_mask = trace_model::ActiveMask::ZERO;
                let mut addrs = [0; WARP_SIZE as usize];

                for thread_idx in 0..(WARP_SIZE as usize) {
                    let thread_instruction = &warp_instructions[thread_idx][instr_idx];
                    if let model::ThreadInstruction::Access(ref access) = thread_instruction {
                        active_mask.set(thread_idx, true);
                        addrs[thread_idx] = access.addr;
                    }
                }

                if let model::ThreadInstruction::Access(ref access) =
                    warp_instructions[0][instr_idx]
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

        // dbg!(&warp_traces[&(trace_model::Dim::ZERO, 0)]
        //     .iter()
        //     .map(|entry| (&entry.instr_opcode, &entry.active_mask))
        //     .collect::<Vec<_>>());

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
        // dbg!(launch_config);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{DevicePtr, ThreadBlock, ThreadIndex, TraceGenerator};
    use crate::model::MemorySpace;
    use color_eyre::eyre;
    use futures::future::Future;
    use num_traits::Float;
    use std::pin::Pin;
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use utils::diff;

    fn reference_vectoradd<T>(a: &Vec<T>, b: &Vec<T>, result: &mut Vec<T>)
    where
        T: Float,
    {
        for (i, sum) in result.iter_mut().enumerate() {
            *sum = a[i] + b[i];
        }
    }

    #[derive(Debug)]
    struct VecAdd<'a, T> {
        dev_a: tokio::sync::Mutex<DevicePtr<&'a mut Vec<T>>>,
        dev_b: tokio::sync::Mutex<DevicePtr<&'a mut Vec<T>>>,
        dev_result: tokio::sync::Mutex<DevicePtr<&'a mut Vec<T>>>,
        n: usize,
    }

    #[async_trait::async_trait]
    impl<'a, T> super::Kernel for VecAdd<'a, T>
    where
        T: Float + std::fmt::Debug + Send + Sync,
    {
        type Error = std::convert::Infallible;

        #[crate::inject_reconvergence_points]
        async fn run(&self, block: &ThreadBlock, tid: &ThreadIndex) -> Result<(), Self::Error> {
            // fn run(&self, block: &ThreadBlock, tid: &ThreadIndex) -> Pin<Box<dyn Future<Output = Result<(), Self::Error>> + Send + '_>> {
            // Box::pin(async move {
            let idx = (tid.block_idx.x * tid.block_dim.x + tid.thread_idx.x) as usize;

            // println!("thread {:?} before", idx);
            block.synchronize_threads().await;
            // println!("thread {:?} after", idx);
            {
                let dev_a = self.dev_a.lock().await;
                let dev_b = self.dev_b.lock().await;
                let mut dev_result = self.dev_result.lock().await;

                if idx < self.n {
                    dev_result[(tid, idx)] = dev_a[(tid, idx)] + dev_b[(tid, idx)];
                }
                // else {
                //     // dev_result[tid] = dev_a[tid] + dev_b[tid];
                // }
            }
            // println!("thread {:?} before 2", idx);
            block.synchronize_threads().await;
            // println!("thread {:?} after 2", idx);
            Ok(())
            // })
        }

        fn name(&self) -> &str {
            "VecAdd"
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_vectoradd() -> eyre::Result<()> {
        // let block_size = 64;
        // let n = 120;
        let block_size = 32;
        let n = 20;

        let mut tracer = super::Tracer::new();

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

        let mut dev_a = tracer.allocate(&mut a, MemorySpace::Global).await;
        let mut dev_b = tracer.allocate(&mut b, MemorySpace::Global).await;
        let mut dev_result = tracer.allocate(&mut result, MemorySpace::Global).await;

        let kernel: VecAdd<f32> = VecAdd {
            dev_a: Mutex::new(dev_a),
            dev_b: Mutex::new(dev_b),
            dev_result: Mutex::new(dev_result),
            n,
        };
        let grid_size = (n as f64 / block_size as f64).ceil() as u32;
        tracer.trace_kernel(grid_size, block_size, kernel).await?;

        reference_vectoradd(&mut a, &mut b, &mut ref_result);

        diff::assert_eq!(have: result, want: ref_result);
        // assert!(false);
        Ok(())
    }

    #[test]
    fn test_size_of_assumptions() {
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
