use super::alloc::{Allocatable, DevicePtr};
use super::cfg::{self, UniqueGraph};
use super::kernel::{Kernel, ThreadBlock, ThreadIndex};
use super::model::{self, MemInstruction, ThreadInstruction};
use futures::StreamExt;
use itertools::Itertools;
use std::collections::HashMap;
use std::sync::{atomic, Arc};
use tokio::sync::Mutex;

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

#[async_trait::async_trait]
pub trait TraceGenerator {
    type Error;

    /// Trace kernel.
    async fn trace_kernel<G, B, K>(
        self: &Arc<Self>,
        grid: G,
        block_size: B,
        mut kernel: K,
    ) -> Result<
        (
            trace_model::command::KernelLaunch,
            trace_model::MemAccessTrace,
        ),
        Error<K::Error, Self::Error>,
    >
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
        instruction: model::ThreadInstruction,
    );

    /// Load address.
    fn load(&self, thread_idx: &ThreadIndex, addr: u64, size: u32, mem_space: model::MemorySpace);

    /// Store address.
    fn store(&self, thread_idx: &ThreadIndex, addr: u64, size: u32, mem_space: model::MemorySpace);
}

#[derive(Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct WarpId {
    pub block_id: trace_model::Point,
    pub warp_id_in_block: usize,
}

pub type WarpInstructionTraces = [Vec<model::ThreadInstruction>; WARP_SIZE as usize];

pub struct Tracer {
    offset: Mutex<u64>,
    traced_instructions: std::sync::Mutex<HashMap<WarpId, WarpInstructionTraces>>,
    kernel_launch_id: atomic::AtomicU64,
}

impl Tracer {
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            offset: Mutex::new(0),
            traced_instructions: std::sync::Mutex::new(HashMap::new()),
            kernel_launch_id: atomic::AtomicU64::new(0),
        })
    }
}

#[async_trait::async_trait]
impl MemoryAccess for Tracer {
    fn push_thread_instruction(
        &self,
        thread_idx: &ThreadIndex,
        instruction: model::ThreadInstruction,
    ) {
        let block_id = thread_idx.block_id.clone();
        let warp_id_in_block = thread_idx.warp_id_in_block;
        let thread_id = thread_idx.thread_id_in_warp;

        let mut block_instructions = self.traced_instructions.lock().unwrap();
        let warp_instructions = block_instructions
            .entry(WarpId {
                block_id,
                warp_id_in_block,
            })
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
        self.push_thread_instruction(thread_idx, inst);
    }

    fn store(&self, thread_idx: &ThreadIndex, addr: u64, size: u32, mem_space: model::MemorySpace) {
        let inst = model::ThreadInstruction::Access(model::MemInstruction {
            kind: model::MemAccessKind::Store,
            addr,
            mem_space,
            size,
        });
        self.push_thread_instruction(thread_idx, inst);
    }
}

pub(crate) fn active_threads<'a>(
    thread_graphs: &'a [cfg::ThreadCFG; WARP_SIZE as usize],
    branch_node: &'a cfg::Node,
    took_branch: bool,
) -> impl Iterator<Item = (usize, &'a [MemInstruction])> + 'a {
    thread_graphs
        .iter()
        .enumerate()
        .filter_map(move |(tid, thread_cfg)| {
            let thread_node_idx = thread_cfg.find_node(branch_node)?;
            let mut edges: Vec<_> = thread_cfg
                .edges_directed(thread_node_idx, petgraph::Incoming)
                .collect();
            assert!(
                edges.len() == 1 || thread_node_idx == petgraph::graph::NodeIndex::new(0),
                "each node has one incoming edge except the source node"
            );
            let active = match edges.pop() {
                Some(edge) => *edge.weight() == took_branch,
                None => true,
            };
            assert!(edges.is_empty());

            if active {
                let instructions = thread_cfg[thread_node_idx].instructions();
                Some((tid, instructions))
            } else {
                None
            }
        })
}

impl Tracer {
    pub async fn run_kernel<K>(
        self: &Arc<Self>,
        grid: trace_model::Dim,
        block_size: trace_model::Dim,
        kernel: K,
        kernel_launch_id: u64,
    ) where
        K: Kernel + Send + Sync,
        <K as Kernel>::Error: Send,
    {
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
                    println!("block {block_id}");

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
            // cannot run blocks in parallel because now they reuse shared memory
            // .buffered(1)
            .collect::<Vec<_>>()
            .await;
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
            memory: self.clone(),
            offset,
        }
    }

    #[allow(clippy::too_many_lines)]
    async fn trace_kernel<G, B, K>(
        self: &Arc<Self>,
        grid: G,
        block_size: B,
        kernel: K,
    ) -> Result<
        (
            trace_model::command::KernelLaunch,
            trace_model::MemAccessTrace,
        ),
        Error<K::Error, Self::Error>,
    >
    where
        G: Into<trace_model::Dim> + Send,
        B: Into<trace_model::Dim> + Send,
        K: Kernel + Send + Sync,
        <K as Kernel>::Error: Send,
    {
        let kernel_launch_id = self.kernel_launch_id.fetch_add(1, atomic::Ordering::SeqCst);
        let kernel_name = kernel.name().unwrap_or("unnamed").to_string();

        self.traced_instructions.lock().unwrap().clear();

        let grid = grid.into();
        let block_size = block_size.into();
        self.run_kernel(grid.clone(), block_size.clone(), kernel, kernel_launch_id)
            .await;

        let mut trace = Vec::new();
        let mut traced_instructions = self.traced_instructions.lock().unwrap();

        dbg!(&traced_instructions.keys().collect::<Vec<_>>());
        // dbg!(&traced_instructions);

        // check for reconvergence points
        if !traced_instructions.values().all(|warp_instructions| {
            warp_instructions
                .iter()
                .all(|thread_instructions| match thread_instructions.get(0) {
                    Some(first_instruction) => {
                        *first_instruction == ThreadInstruction::TookBranch(0)
                    }
                    None => true,
                })
        }) {
            return Err(Error::Tracer(TraceError::MissingReconvergencePoints));
        }

        for (
            WarpId {
                block_id,
                warp_id_in_block,
            },
            warp_instructions,
        ) in traced_instructions.drain()
        {
            let mut super_cfg = cfg::CFG::new();
            let super_cfg_root_node_idx = super_cfg.add_unique_node(cfg::Node::Branch(0));

            let mut thread_graphs = [(); WARP_SIZE as usize].map(|_| cfg::ThreadCFG::default());
            for (ti, thread_instructions) in warp_instructions.iter().enumerate() {
                let (thread_cfg, (thread_cfg_root_node_idx, thread_cfg_sink_node_idx)) =
                    cfg::build_control_flow_graph(thread_instructions, &mut super_cfg);

                let paths: Vec<Vec<_>> = cfg::all_simple_paths(
                    &thread_cfg,
                    thread_cfg_root_node_idx,
                    thread_cfg_sink_node_idx,
                )
                .collect();

                // each edge connects two distinct nodes, resulting in a
                // single control flow path each thread takes
                assert_eq!(paths.len(), 1);
                println!(
                    "thread[{:2}] = {:?}",
                    ti,
                    cfg::format_control_flow_path(&thread_cfg, &paths[0]).join(" ")
                );

                thread_graphs[ti] = thread_cfg;
            }

            // fill remaining edges (this should be optional step)
            cfg::add_missing_control_flow_edges(&mut super_cfg);

            println!(
                "super cfg: {} nodes, {} edges, {} edge weights",
                super_cfg.node_count(),
                super_cfg.edge_count(),
                super_cfg.edge_weights().count(),
            );

            let super_cfg_final_reconvergence_id = super_cfg
                .node_indices()
                .filter_map(|idx| match super_cfg[idx] {
                    cfg::Node::Reconverge(branch_id) => Some(branch_id),
                    cfg::Node::Branch(_) => None,
                })
                .max()
                .unwrap_or(0);
            let super_cfg_sink_node_idx = super_cfg
                .find_node(&cfg::Node::Reconverge(super_cfg_final_reconvergence_id))
                .unwrap();

            let ways = super::cfg::all_simple_paths::<Vec<_>, _>(
                &super_cfg,
                super_cfg_root_node_idx,
                super_cfg_sink_node_idx,
            )
            .collect::<Vec<_>>();
            dbg!(ways.len());

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
                active_mask: trace_model::ActiveMask::ZERO,
                addrs: [0; 32],
            };

            let iter = cfg::visit::DominatedBfs::new(&super_cfg, super_cfg_root_node_idx);

            for (edge_idx, node_idx) in iter {
                let took_branch = super_cfg[edge_idx];

                // add the instructions
                let active_threads: Vec<_> =
                    active_threads(&thread_graphs, &super_cfg[node_idx], took_branch).collect();

                // find longest branch
                // the length can differ if we have loops with different number of repetitions
                let (_, longest_thread_trace) = active_threads
                    .iter()
                    .max_by_key(|(_, instructions)| instructions.len())
                    .copied()
                    .unwrap_or_default();

                println!(
                    "node {:?} has {} instructions",
                    super_cfg[node_idx],
                    longest_thread_trace.len()
                );

                let mut branch_trace: Vec<_> = longest_thread_trace
                    .iter()
                    .enumerate()
                    .map(|(instr_idx, access)| {
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
                            model::MemorySpace::Constant if is_store => {
                                panic!("constant store")
                            }
                            other => panic!("unknown memory space {other:?}"),
                        };

                        trace_model::MemAccessTraceEntry {
                            instr_opcode: instr_opcode.to_string(),
                            instr_is_mem: true,
                            instr_is_store: is_store,
                            instr_is_load: is_load,
                            instr_idx: instr_idx as u32,
                            ..warp_instruction.clone()
                        }
                    })
                    .collect();

                // push the instructions for this branch
                for (instr_idx, instr) in branch_trace.iter_mut().enumerate() {
                    for (tid, instructions) in &active_threads {
                        if let Some(access) = instructions.get(instr_idx) {
                            instr.active_mask.set(*tid, true);
                            instr.addrs[*tid] = access.addr;
                        }
                    }
                }

                // here we have the warp_trace ready to be added into the global trace
                // dbg!(branch_trace.len());
                trace.extend(branch_trace.into_iter());

                let mut active_mask = trace_model::ActiveMask::ZERO;
                for (tid, _) in &active_threads {
                    active_mask.set(*tid, true);
                }
            }

            // end of warp: add EXIT instruction
            trace.push(trace_model::MemAccessTraceEntry {
                instr_opcode: "EXIT".to_string(),
                instr_idx: trace.len() as u32,
                active_mask: trace_model::ActiveMask::all_ones(),
                ..warp_instruction.clone()
            });
        }

        let trace = trace_model::MemAccessTrace(trace);
        let launch_config = trace_model::command::KernelLaunch {
            mangled_name: kernel_name.clone(),
            unmangled_name: kernel_name.clone(),
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
        Ok((launch_config, trace))
    }
}

pub mod testing {
    #[macro_export]
    macro_rules! mem_inst {
        ($kind:ident[$space:ident]@$addr:expr, $size:expr) => {{
            $crate::model::MemInstruction {
                kind: $crate::model::MemAccessKind::$kind,
                mem_space: $crate::model::MemorySpace::$space,
                addr: $addr,
                size: $size,
            }
        }};
    }
    pub use mem_inst;

    #[derive(Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub struct SimplifiedTraceInstruction {
        opcode: String,
        first_addr: u64,
        active_mask: String,
        pc: u32,
        idx: usize,
    }

    impl From<(usize, (&str, u64, &str, u32))> for SimplifiedTraceInstruction {
        fn from(value: (usize, (&str, u64, &str, u32))) -> Self {
            let (idx, (opcode, first_addr, active_mask, pc)) = value;
            assert_eq!(active_mask.len(), 32, "invalid active mask {active_mask:?}");
            Self {
                opcode: opcode.to_string(),
                first_addr,
                active_mask: active_mask.to_string(),
                pc,
                idx,
            }
        }
    }

    pub fn simplify_warp_trace(
        warp_trace: &[trace_model::MemAccessTraceEntry],
    ) -> impl Iterator<Item = SimplifiedTraceInstruction> + '_ {
        warp_trace
            .iter()
            .enumerate()
            .map(|(warp_instr_idx, warp_instr)| {
                let human_readable_active_mask = warp_instr
                    .active_mask
                    .to_string()
                    .chars()
                    .rev()
                    .collect::<String>();

                SimplifiedTraceInstruction {
                    opcode: warp_instr.instr_opcode.clone(),
                    first_addr: warp_instr
                        .addrs
                        .iter()
                        .enumerate()
                        .filter(|(ti, _)| warp_instr.active_mask[*ti])
                        .map(|(_, addr)| addr)
                        .next()
                        .copied()
                        .unwrap_or(0),
                    active_mask: human_readable_active_mask,
                    pc: warp_instr.instr_offset,
                    idx: warp_instr_idx,
                }
            })
    }

    pub fn print_warp_trace(trace: &[trace_model::MemAccessTraceEntry]) {
        for inst in simplify_warp_trace(trace) {
            println!(
                "{:<10}{:?}\t active={} \tpc={} idx={}",
                inst.opcode, inst.first_addr, inst.active_mask, inst.pc, inst.idx,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{self, mem_inst, SimplifiedTraceInstruction};
    use super::{DevicePtr, ThreadBlock, ThreadIndex, TraceGenerator};
    use crate::model::MemorySpace;
    use color_eyre::eyre;
    use num_traits::Float;
    use tokio::sync::Mutex;
    use utils::diff;

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_for_loop_kernel() -> eyre::Result<()> {
        struct ForLoopKernel {}
        #[async_trait::async_trait]
        impl super::Kernel for ForLoopKernel {
            type Error = std::convert::Infallible;
            #[crate::inject_reconvergence_points]
            async fn run(&self, block: &ThreadBlock, tid: &ThreadIndex) -> Result<(), Self::Error> {
                let is_even = tid.thread_idx.x % 2 == 0;
                let num_iterations = if is_even { 3 } else { 1 };
                for _ in 0..num_iterations {
                    let inst = mem_inst!(Load[Global]@1, 4);
                    block.memory.push_thread_instruction(tid, inst.into());
                }
                let inst = mem_inst!(Load[Global]@100, 4);
                block.memory.push_thread_instruction(tid, inst.into());
                Ok(())
            }
        }

        let tracer = super::Tracer::new();
        let (_launch_config, trace) = tracer.trace_kernel(1, 32, ForLoopKernel {}).await?;
        let warp_traces = trace.clone().to_warp_traces();
        let first_warp = &warp_traces[&(trace_model::Dim::ZERO, 0)];
        testing::print_warp_trace(first_warp);
        diff::assert_eq!(
            have: testing::simplify_warp_trace(first_warp).collect::<Vec<_>>(),
            want: [
                ("LDG", 1, "11111111111111111111111111111111", 0),
                ("LDG", 1, "10101010101010101010101010101010", 0),
                ("LDG", 1, "10101010101010101010101010101010", 0),
                ("LDG", 100, "11111111111111111111111111111111", 0),
                ("EXIT", 0, "11111111111111111111111111111111", 0),
            ].into_iter().enumerate().map(SimplifiedTraceInstruction::from).collect::<Vec<_>>()
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_if_kernel() -> eyre::Result<()> {
        struct IfKernel {}
        #[async_trait::async_trait]
        impl super::Kernel for IfKernel {
            type Error = std::convert::Infallible;
            #[crate::inject_reconvergence_points]
            async fn run(&self, block: &ThreadBlock, tid: &ThreadIndex) -> Result<(), Self::Error> {
                if tid.thread_idx.x < 16 {
                    let inst = mem_inst!(Load[Global]@1, 4);
                    block.memory.push_thread_instruction(tid, inst.into());
                }
                Ok(())
            }
        }

        let tracer = super::Tracer::new();
        let (_launch_config, trace) = tracer.trace_kernel(1, 32, IfKernel {}).await?;
        let warp_traces = trace.clone().to_warp_traces();
        let first_warp = &warp_traces[&(trace_model::Dim::ZERO, 0)];
        testing::print_warp_trace(first_warp);
        diff::assert_eq!(
            have: testing::simplify_warp_trace(first_warp).collect::<Vec<_>>(),
            want: [
                ("LDG", 1, "11111111111111110000000000000000", 0),
                ("EXIT", 0, "11111111111111111111111111111111", 0),
            ].into_iter().enumerate().map(SimplifiedTraceInstruction::from).collect::<Vec<_>>()
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_full_imbalance() -> eyre::Result<()> {
        struct FullImbalanceKernel {}
        #[async_trait::async_trait]
        impl super::Kernel for FullImbalanceKernel {
            type Error = std::convert::Infallible;
            #[crate::inject_reconvergence_points]
            async fn run(&self, block: &ThreadBlock, tid: &ThreadIndex) -> Result<(), Self::Error> {
                if tid.thread_idx.x < 16 {
                    if tid.thread_idx.x < 8 {
                        let inst = mem_inst!(Load[Global]@1, 4);
                        block.memory.push_thread_instruction(tid, inst.into());
                    }
                    let inst = mem_inst!(Load[Global]@2, 4);
                    block.memory.push_thread_instruction(tid, inst.into());
                } else {
                    if tid.thread_idx.x < 24 {
                        let inst = mem_inst!(Load[Global]@10, 4);
                        block.memory.push_thread_instruction(tid, inst.into());
                    }
                    let inst = mem_inst!(Load[Global]@11, 4);
                    block.memory.push_thread_instruction(tid, inst.into());
                }
                let inst = mem_inst!(Load[Global]@100, 4);
                block.memory.push_thread_instruction(tid, inst.into());
                Ok(())
            }
        }

        let tracer = super::Tracer::new();
        let (_launch_config, trace) = tracer.trace_kernel(1, 32, FullImbalanceKernel {}).await?;
        let warp_traces = trace.clone().to_warp_traces();
        let first_warp = &warp_traces[&(trace_model::Dim::ZERO, 0)];
        testing::print_warp_trace(first_warp);
        diff::assert_eq!(
            have: testing::simplify_warp_trace(first_warp).collect::<Vec<_>>(),
            want: [
                ("LDG", 1, "11111111000000000000000000000000", 0),
                ("LDG", 10, "00000000000000001111111100000000", 0),
                ("LDG", 11, "00000000000000001111111111111111", 0),
                ("LDG", 2, "11111111111111110000000000000000", 0),
                ("LDG", 100, "11111111111111111111111111111111", 0),
                ("EXIT", 0, "11111111111111111111111111111111", 0),
            ].into_iter().enumerate().map(SimplifiedTraceInstruction::from).collect::<Vec<_>>()
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_two_level_nested_if_kernel() -> eyre::Result<()> {
        struct Imbalanced {}
        #[async_trait::async_trait]
        impl super::Kernel for Imbalanced {
            type Error = std::convert::Infallible;
            #[crate::inject_reconvergence_points]
            async fn run(&self, block: &ThreadBlock, tid: &ThreadIndex) -> Result<(), Self::Error> {
                // fully parallel
                let inst = mem_inst!(Load[Global]@100, 4);
                block.memory.push_thread_instruction(tid, inst.into());

                if tid.thread_idx.x < 16 {
                    let inst = mem_inst!(Load[Global]@1, 4);
                    block.memory.push_thread_instruction(tid, inst.into());
                    if tid.thread_idx.x < 8 {
                        let inst = mem_inst!(Load[Global]@2, 4);
                        block.memory.push_thread_instruction(tid, inst.into());
                    }
                }

                // have reconverged to fully parallel
                let inst = mem_inst!(Load[Global]@100, 4);
                block.memory.push_thread_instruction(tid, inst.into());

                Ok(())
            }
        }

        struct Balanced {}
        #[async_trait::async_trait]
        impl super::Kernel for Balanced {
            type Error = std::convert::Infallible;
            #[crate::inject_reconvergence_points]
            async fn run(&self, block: &ThreadBlock, tid: &ThreadIndex) -> Result<(), Self::Error> {
                // fully parallel
                let inst = mem_inst!(Load[Global]@100, 4);
                block.memory.push_thread_instruction(tid, inst.into());

                if tid.thread_idx.x < 16 {
                    let inst = mem_inst!(Load[Global]@10, 4);
                    block.memory.push_thread_instruction(tid, inst.into());
                    if tid.thread_idx.x < 8 {
                        let inst = mem_inst!(Load[Global]@11, 4);
                        block.memory.push_thread_instruction(tid, inst.into());
                    }
                } else {
                    let inst = mem_inst!(Load[Global]@20, 4);
                    block.memory.push_thread_instruction(tid, inst.into());
                    if tid.thread_idx.x < 8 {
                        let inst = mem_inst!(Load[Global]@21, 4);
                        block.memory.push_thread_instruction(tid, inst.into());
                    }
                }

                // have reconverged to fully parallel
                let inst = mem_inst!(Load[Global]@100, 4);
                block.memory.push_thread_instruction(tid, inst.into());
                Ok(())
            }
        }

        let tracer = super::Tracer::new();
        let (_launch_config, trace) = tracer.trace_kernel(1, 32, Imbalanced {}).await?;
        let warp_traces = trace.clone().to_warp_traces();
        let first_warp = &warp_traces[&(trace_model::Dim::ZERO, 0)];
        testing::print_warp_trace(first_warp);
        diff::assert_eq!(
            have: testing::simplify_warp_trace(first_warp).collect::<Vec<_>>(),
            want: [
                ("LDG", 100, "11111111111111111111111111111111", 0),
                ("LDG", 1, "11111111111111110000000000000000", 0),
                ("LDG", 2, "11111111000000000000000000000000", 0),
                ("LDG", 100, "11111111111111111111111111111111", 0),
                ("EXIT", 0, "11111111111111111111111111111111", 0),
            ].into_iter().enumerate().map(SimplifiedTraceInstruction::from).collect::<Vec<_>>()
        );
        Ok(())
    }

    #[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_vectoradd() -> eyre::Result<()> {
        fn reference_vectoradd<T>(a: &[T], b: &[T], result: &mut [T])
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
                let idx = (tid.block_idx.x * tid.block_dim.x + tid.thread_idx.x) as usize;

                block.synchronize_threads().await;
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
                block.synchronize_threads().await;
                Ok(())
            }
        }

        // let block_size = 64;
        // let n = 120;
        let block_size = 32;
        let n = 20;

        let tracer = super::Tracer::new();

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

        let dev_a = tracer.allocate(&mut a, MemorySpace::Global).await;
        let dev_b = tracer.allocate(&mut b, MemorySpace::Global).await;
        let dev_result = tracer.allocate(&mut result, MemorySpace::Global).await;

        let kernel: VecAdd<f32> = VecAdd {
            dev_a: Mutex::new(dev_a),
            dev_b: Mutex::new(dev_b),
            dev_result: Mutex::new(dev_result),
            n,
        };
        let grid_size = (n as f64 / f64::from(block_size)).ceil() as u32;
        let (_launch_config, trace) = tracer.trace_kernel(grid_size, block_size, kernel).await?;
        let warp_traces = trace.clone().to_warp_traces();
        let first_warp = &warp_traces[&(trace_model::Dim::ZERO, 0)];

        reference_vectoradd(&a, &b, &mut ref_result);
        diff::assert_eq!(have: result, want: ref_result);

        testing::print_warp_trace(first_warp);
        diff::assert_eq!(
            have: testing::simplify_warp_trace(first_warp).collect::<Vec<_>>(),
            want: [
                ("LDG", 0, "11111111111111111111000000000000", 0),
                ("LDG", 80, "11111111111111111111000000000000", 0),
                ("STG", 160, "11111111111111111111000000000000", 0),
                ("EXIT", 0, "11111111111111111111111111111111", 0),
            ].into_iter().enumerate().map(SimplifiedTraceInstruction::from).collect::<Vec<_>>()
        );
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
