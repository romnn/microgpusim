use super::alloc::{self, Allocatable, DevicePtr};
use super::cfg::{self, UniqueGraph};
use super::kernel::{Kernel, ThreadBlock, ThreadIndex};
use super::model::{self, Instruction, ThreadInstruction};
use futures::StreamExt;
use itertools::Itertools;
use std::collections::HashMap;
use std::sync::{atomic, Arc};
use std::time::Instant;
use tokio::sync::Mutex;
use trace_model::ActiveMask;
use trace_model::WARP_SIZE;

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

#[derive(Debug, Default, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Options {
    pub no_data_dependency: bool,
}

#[async_trait::async_trait]
pub trait TraceGenerator {
    type Error;

    /// Trace kernel.
    async fn trace_kernel<G, B, K>(
        self: &Arc<Self>,
        grid: G,
        block_size: B,
        kernel: &mut K,
        options: &Options,
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
    async fn allocate<T>(
        self: &Arc<Self>,
        value: T,
        options: Option<alloc::Options>,
        // mem_space: model::MemorySpace,
        // name: Option<S>,
    ) -> DevicePtr<T>
    where
        T: Allocatable + Send;
    // S: ToString + Send;

    /// Get commands
    async fn commands<'a>(self: &'a Arc<Self>) -> Vec<trace_model::Command>;
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
    fn load(
        &self,
        thread_idx: &ThreadIndex,
        addr: u64,
        size: u32,
        mem_space: model::MemorySpace,
        bypass_l1: bool,
        bypass_l2: bool,
    );

    /// Store address.
    fn store(
        &self,
        thread_idx: &ThreadIndex,
        addr: u64,
        size: u32,
        mem_space: model::MemorySpace,
        bypass_l1: bool,
        bypass_l2: bool,
    );
}

#[derive(Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct WarpId {
    pub block_id: trace_model::Point,
    pub warp_id_in_block: usize,
}

pub type WarpInstructionTraces = [Vec<model::ThreadInstruction>; WARP_SIZE as usize];

pub struct Tracer {
    /// Offsets per memory space
    offsets: Mutex<[u64; trace_model::MemorySpace::count()]>,
    /// Traced instructions for the current kernel launch.
    ///
    /// Instructions are cleared after each kernel and there may only run one kernel at any time.
    traced_instructions: std::sync::Mutex<HashMap<WarpId, WarpInstructionTraces>>,
    kernel_launch_id: atomic::AtomicU64,
    commands: std::sync::Mutex<Vec<trace_model::command::Command>>,
}

impl Tracer {
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            offsets: Mutex::new([0; trace_model::MemorySpace::count()]),
            traced_instructions: std::sync::Mutex::new(HashMap::new()),
            kernel_launch_id: atomic::AtomicU64::new(0),
            commands: std::sync::Mutex::new(Vec::new()),
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

        // if let model::ThreadInstruction::Access(ref mut access) = instruction {
        //     access.addr += access.mem_space.base_addr();
        // }

        let mut block_instructions = self.traced_instructions.lock().unwrap();
        let warp_instructions = block_instructions
            .entry(WarpId {
                block_id,
                warp_id_in_block,
            })
            .or_default();
        warp_instructions[thread_id].push(instruction);
    }

    fn load(
        &self,
        thread_idx: &ThreadIndex,
        addr: u64,
        size: u32,
        mem_space: model::MemorySpace,
        bypass_l1: bool,
        bypass_l2: bool,
    ) {
        let inst = model::ThreadInstruction::Access(model::MemInstruction {
            kind: model::MemAccessKind::Load,
            addr,
            mem_space,
            bypass_l1,
            bypass_l2,
            size,
        });
        self.push_thread_instruction(thread_idx, inst);
    }

    fn store(
        &self,
        thread_idx: &ThreadIndex,
        addr: u64,
        size: u32,
        mem_space: model::MemorySpace,
        bypass_l1: bool,
        bypass_l2: bool,
    ) {
        let inst = model::ThreadInstruction::Access(model::MemInstruction {
            kind: model::MemAccessKind::Store,
            addr,
            mem_space,
            bypass_l1,
            bypass_l2,
            size,
        });
        self.push_thread_instruction(thread_idx, inst);
    }
}

pub(crate) fn active_threads<'a>(
    thread_graphs: &'a [cfg::ThreadCFG; WARP_SIZE as usize],
    branch_node: &'a cfg::WarpNode,
    took_branch: bool,
) -> impl Iterator<Item = (usize, &'a [Instruction])> + 'a {
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
                Some(edge) => edge.weight().took_branch() == took_branch,
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

/// Texture allocation byte alignment (cudaDeviceProp::textureAlignment)
///
/// Guaranteed to be 256B or greater.
/// On Pascal GTX1080, its 512B.
pub const ALIGNMENT_BYTES: u64 = 256;

type WarpTrace = (WarpId, cfg::WarpCFG, [cfg::ThreadCFG; WARP_SIZE]);

impl Tracer {
    pub async fn run_kernel<K>(
        self: &Arc<Self>,
        grid: &trace_model::Dim,
        block_size: &trace_model::Dim,
        kernel: &mut K,
        kernel_launch_id: u64,
    ) where
        K: Kernel + Send + Sync,
        <K as Kernel>::Error: Send,
    {
        let start = Instant::now();
        let kernel = Arc::new(kernel);

        let block_ids: Vec<_> = grid.clone().into_iter().collect();
        let num_blocks = block_ids.len();
        log::debug!("launching {} blocks", num_blocks);

        // loop over the grid
        futures::stream::iter(block_ids)
            .then(|block_id| {
                let block_size = block_size.clone();
                let block_id = block_id.clone();
                let kernel = kernel.clone();
                async move {
                    log::trace!("block {block_id}");

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
                                // log::trace!(
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

        log::debug!(
            "ran {} blocks of kernel {:?} in {:?}",
            num_blocks,
            kernel.name(),
            start.elapsed()
        );
    }

    pub async fn trace_control_flow_graphs<K>(
        self: &Arc<Self>,
        grid: &trace_model::Dim,
        block_size: &trace_model::Dim,
        kernel: &mut K,
        kernel_launch_id: u64,
    ) -> Result<impl Iterator<Item = WarpTrace>, Error<K::Error, TraceError>>
    where
        K: Kernel + Send + Sync,
        <K as Kernel>::Error: Send,
    {
        self.traced_instructions.lock().unwrap().clear();

        self.run_kernel(grid, block_size, kernel, kernel_launch_id)
            .await;

        let mut traced_instructions = self.traced_instructions.lock().unwrap();

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

        // sort warps
        let mut traced_instructions: Vec<_> = traced_instructions.drain().collect();
        traced_instructions
            .sort_by_key(|(warp, _)| (warp.block_id.accelsim_id(), warp.warp_id_in_block));

        let iter = traced_instructions.into_iter()
            .map(|(warp_id, per_thread_instructions)| {
            let WarpId { ref block_id, warp_id_in_block } = warp_id;
            if log::log_enabled!(log::Level::Debug) {
                let per_thread_instruction_count: Vec<_> = per_thread_instructions
                    .iter()
                    .map(|per_thread| per_thread.iter().map(|inst| inst.is_access()).count())
                    .collect();
                let total_thread_instruction_count =
                    per_thread_instruction_count.iter().sum::<usize>();
                let mean_thread_instruction_count =
                    per_thread_instruction_count.iter().sum::<usize>() as f32
                        / per_thread_instruction_count
                            .iter()
                            .filter(|n| **n > 0)
                            .count() as f32;

                log::debug!(
                    "==> block {:?} warp {:<3} has {} trace instructions ({:.2} per thread)",
                    block_id,
                    warp_id_in_block,
                    total_thread_instruction_count,
                    mean_thread_instruction_count,
                );
            }

            let mut warp_cfg = cfg::WarpCFG::new();
            let mut warp_active_mask = ActiveMask::ZERO;
            let warp_cfg_root_node_idx = warp_cfg.add_unique_node(cfg::WarpNode::Branch {
                id: 0,
                branch_id: 0,
            });

            let start = Instant::now();
            let mut thread_graphs = [(); WARP_SIZE as usize].map(|_| cfg::ThreadCFG::default());
            for (tid, thread_instructions) in per_thread_instructions.iter().enumerate() {
                let (thread_cfg, (thread_cfg_root_node_idx, thread_cfg_sink_node_idx)) =
                    cfg::build_control_flow_graph(thread_instructions, &mut warp_cfg, &mut warp_active_mask, tid);

                #[cfg(debug_assertions)]
                {
                    let paths: Vec<Vec<_>> = cfg::all_simple_paths(
                        &thread_cfg,
                        thread_cfg_root_node_idx,
                        thread_cfg_sink_node_idx,
                    )
                    .collect();

                    // each edge connects two distinct nodes, resulting in a
                    // single control flow path each thread takes
                    debug_assert_eq!(paths.len(), 1);
                    log::trace!(
                        "thread[{:2}] = {:?}",
                        tid,
                        cfg::format_control_flow_path(&thread_cfg, &paths[0]).join(" ")
                    );
                }

                thread_graphs[tid] = thread_cfg;
            }

            if log::log_enabled!(log::Level::Debug) {
                let per_thread_cfg_node_count: Vec<_> =
                    thread_graphs.iter().map(|tg| tg.node_count()).collect();
                log::debug!(
                    "==> block {:?} warp {:<3} built thread graphs in {:?} (nodes: mean={:.2} max={} min={})",
                    block_id,
                    warp_id_in_block,
                    start.elapsed(),
                    per_thread_cfg_node_count.iter().sum::<usize>() as f32 / WARP_SIZE as f32,
                    per_thread_cfg_node_count.iter().max().copied().unwrap_or(0),
                    per_thread_cfg_node_count.iter().min().copied().unwrap_or(0),
                );
            }

            // fill remaining edges (this should be optional step)
            // cfg::add_missing_control_flow_edges(&mut warp_cfg);

            let mut unique_control_flow_path_count: Option<usize> = None;
            #[cfg(debug_assertions)]
            if false {
                let warp_cfg_sink_node_idx = warp_cfg
                    .find_node(&cfg::WarpNode::Reconverge {
                        id: 0,
                        branch_id: 0,
                    })
                    .unwrap();

                unique_control_flow_path_count = Some(
                    super::cfg::all_simple_paths::<Vec<_>, _>(
                        &warp_cfg,
                        warp_cfg_root_node_idx,
                        warp_cfg_sink_node_idx,
                    )
                    .count(),
                );
            };

            log::debug!(
                "super CFG: {} nodes, {} edges, {} edge weights, {} unique control flow paths",
                warp_cfg.node_count(),
                warp_cfg.edge_count(),
                warp_cfg.edge_weights().count(),
                unique_control_flow_path_count
                    .as_ref()
                    .map(ToString::to_string)
                    .unwrap_or("?".to_string()),
            );
            (warp_id, warp_cfg, thread_graphs)
        });
        Ok(iter)
    }
}

#[async_trait::async_trait]
impl TraceGenerator for Tracer {
    type Error = TraceError;

    async fn allocate<T>(
        self: &Arc<Self>,
        value: T,
        options: Option<alloc::Options>,
    ) -> DevicePtr<T>
    where
        T: Allocatable + Send,
    {
        let options = options.unwrap_or_default();
        let mut offsets_lock = self.offsets.lock().await;
        let offset = &mut offsets_lock[options.mem_space as usize];
        let base_addr = options.mem_space.base_addr();
        let addr = utils::next_multiple(*offset, ALIGNMENT_BYTES);
        let num_bytes = value.size() as u64;

        // align offset too
        *offset = addr + num_bytes;
        *offset = utils::next_multiple(*offset, ALIGNMENT_BYTES);

        self.commands
            .lock()
            .unwrap()
            .push(trace_model::command::Command::MemAlloc(
                trace_model::command::MemAlloc {
                    allocation_name: options.name,
                    device_ptr: base_addr + addr,
                    fill_l2: options.fill_l2,
                    num_bytes,
                },
            ));

        DevicePtr {
            inner: value,
            mem_space: options.mem_space,
            memory: self.clone(),
            offset: base_addr + addr,
            bypass_l1: false,
            bypass_l2: false,
        }
    }

    async fn commands<'a>(self: &'a Arc<Self>) -> Vec<trace_model::Command> {
        self.commands.lock().unwrap().clone()
    }

    async fn trace_kernel<G, B, K>(
        self: &Arc<Self>,
        grid: G,
        block_size: B,
        kernel: &mut K,
        options: &Options,
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
        let mut trace = Vec::new();

        let grid = grid.into();
        let block_size = block_size.into();

        let cfg_iter = self
            .trace_control_flow_graphs(&grid, &block_size, kernel, kernel_launch_id)
            .await?;

        for (warp_id, warp_cfg, thread_cfgs) in cfg_iter {
            let WarpId {
                block_id,
                warp_id_in_block,
            } = warp_id;
            let warp_instruction = trace_model::MemAccessTraceEntry {
                cuda_ctx: 0,
                device_id: 0,
                sm_id: 0,
                kernel_id: 0,
                block_id: block_id.clone().into(),
                warp_id_in_sm: warp_id_in_block as u32,
                warp_id_in_block: warp_id_in_block as u32,
                warp_size: trace_model::WARP_SIZE as u32,
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
                active_mask: ActiveMask::ZERO,
                addrs: [0; 32],
                thread_indices: [(0, 0, 0); 32],
            };

            let mut pc = 0;

            let warp_cfg_root_node_idx = warp_cfg
                .find_node(&cfg::WarpNode::Branch {
                    id: 0,
                    branch_id: 0,
                })
                .unwrap();

            let iter = cfg::visit::DominatedDfs::new(&warp_cfg, warp_cfg_root_node_idx);

            for (edge_idx, node_idx) in iter {
                let edge = warp_cfg[edge_idx];
                log::trace!(
                    "trace assembly: node={} took branch={}",
                    warp_cfg[node_idx],
                    edge.took_branch()
                );

                // useful for debugging
                // for node_idx in thread_graphs[0].node_indices() {
                //     let node = &thread_graphs[0][node_idx];
                //     if node.instructions().is_empty() {
                //         continue;
                //     }
                //     for inst in node.instructions() {
                //         log::trace!("node {} instructions: {}", node, inst);
                //     }
                // }

                // add the instructions
                let active_threads: Vec<_> =
                    active_threads(&thread_cfgs, &warp_cfg[node_idx], edge.took_branch()).collect();

                // find longest branch
                // the length can differ if we have loops with different number of repetitions
                let (_, longest_thread_trace) = active_threads
                    .iter()
                    .max_by_key(|(_, instructions)| instructions.len())
                    .copied()
                    .unwrap_or_default();

                let mut branch_trace: Vec<_> = longest_thread_trace
                    .iter()
                    .enumerate()
                    .map(|(instr_idx, instr)| {
                        match instr {
                            Instruction::Memory(access) => {
                                let is_load = access.kind == model::MemAccessKind::Load;
                                let is_store = access.kind == model::MemAccessKind::Store;
                                let mut instr_opcode = match access.mem_space {
                                    model::MemorySpace::Local if is_load => "LDL.E".to_string(),
                                    model::MemorySpace::Global if is_load => "LDG.E".to_string(),
                                    model::MemorySpace::Shared if is_load => "LDS.E".to_string(),
                                    // MemorySpace::Texture if is_load => "LDG".to_string(),
                                    model::MemorySpace::Constant if is_load => "LDC.E".to_string(),
                                    model::MemorySpace::Local if is_store => "STL.E".to_string(),
                                    model::MemorySpace::Global if is_store => "STG.E".to_string(),
                                    model::MemorySpace::Shared if is_store => "STS.E".to_string(),
                                    // MemorySpace::Texture if is_store => "LDG".to_string(),
                                    model::MemorySpace::Constant if is_store => {
                                        todo!("constant store")
                                    }
                                    other => panic!("unknown memory space {other:?}"),
                                };

                                if access.bypass_l1 {
                                    instr_opcode += ".CG";
                                }

                                trace_model::MemAccessTraceEntry {
                                    instr_opcode: instr_opcode.to_string(),
                                    instr_is_mem: true,
                                    instr_is_store: is_store,
                                    instr_is_load: is_load,
                                    instr_is_extended: false,
                                    instr_mem_space: access.mem_space.into(),
                                    instr_idx: instr_idx as u32,
                                    ..warp_instruction.clone()
                                }
                            }
                            Instruction::Barrier => trace_model::MemAccessTraceEntry {
                                instr_opcode: "MEMBAR".to_string(),
                                instr_is_mem: false,
                                instr_is_store: false,
                                instr_is_load: false,
                                instr_is_extended: false,
                                instr_mem_space: trace_model::MemorySpace::None,
                                instr_idx: instr_idx as u32,
                                ..warp_instruction.clone()
                            },
                        }
                    })
                    .collect();

                // push the instructions for this branch
                for (instr_idx, instr) in branch_trace.iter_mut().enumerate() {
                    for (tid, instructions) in &active_threads {
                        if instructions.get(instr_idx).is_some() {
                            instr.active_mask.set(*tid, true);
                        }
                        instr.instr_offset = pc;
                        match instructions.get(instr_idx) {
                            Some(Instruction::Memory(access)) => {
                                instr.addrs[*tid] = access.addr;
                                // TODO: find a way to add the thread idx here
                                // instr.thread_indices[*tid] = access.addr;

                                if options.no_data_dependency {
                                    instr.set_source_registers([]);
                                    instr.set_dest_registers([]);
                                } else {
                                    // We assume memory instructions are all data dependant.
                                    //
                                    // This assumption does not always hold, but one could
                                    // argue that it makes sense if compute instructions are
                                    // skipped.
                                    if instr.instr_is_load {
                                        // read address R1 and store to R1.
                                        instr.set_source_registers([1]);
                                        instr.set_dest_registers([1]);
                                    } else if instr.instr_is_store {
                                        // store data R1 to R2 (no destination register)
                                        instr.set_source_registers([1, 2]);
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    pc += 1;
                }

                // here we have the warp_trace ready to be added into the global trace
                for inst in fmt::simplify_warp_trace(&branch_trace, true) {
                    log::trace!("{}", inst);
                }

                trace.extend(branch_trace.into_iter());

                let mut active_mask = ActiveMask::ZERO;
                for (tid, _) in &active_threads {
                    active_mask.set(*tid, true);
                }
            }

            // end of warp: add EXIT instruction
            trace.push(trace_model::MemAccessTraceEntry {
                instr_opcode: "EXIT".to_string(),
                instr_idx: trace.len() as u32,
                instr_offset: pc,
                active_mask: ActiveMask::all_ones(),
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
            shared_mem_addr_limit: 0,
            local_mem_base_addr: 0,
            local_mem_addr_limit: 0,
            nvbit_version: "none".to_string(),
            device_properties: trace_model::DeviceProperties::default(),
        };
        self.commands
            .lock()
            .unwrap()
            .push(trace_model::command::Command::KernelLaunch(
                launch_config.clone(),
            ));
        Ok((launch_config, trace))
    }
}

pub mod util {
    #[macro_export]
    macro_rules! mem_inst {
        ($kind:ident[$space:ident]@$addr:expr, $size:expr) => {{
            let mem_space = $crate::model::MemorySpace::$space;
            let base_addr = mem_space.base_addr();
            $crate::model::MemInstruction {
                kind: $crate::model::MemAccessKind::$kind,
                size: $size,
                addr: base_addr + $addr,
                bypass_l1: false,
                bypass_l2: false,
                mem_space,
            }
        }};
    }
    pub use mem_inst;
}

pub mod fmt {
    use itertools::Itertools;
    use trace_model::WARP_SIZE;

    #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
    pub enum Addresses {
        None,
        PerThread([u64; WARP_SIZE]),
        BaseStride { base: i128, stride: i128 },
        Uniform(u64),
    }

    impl std::fmt::Display for Addresses {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::None => write!(f, "None"),
                Self::Uniform(addr) => write!(f, "Uniform({:>3})", addr),
                Self::BaseStride { base, stride } => {
                    write!(f, "BaseStride({:>3}, stride={:<3})", base, stride)
                }
                Self::PerThread(addresses) => write!(f, "{:?}", addresses),
            }
        }
    }

    #[derive(Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub struct SimplifiedTraceInstruction {
        opcode: String,
        addresses: Addresses,
        active_mask: String,
        pc: u32,
    }

    impl std::fmt::Display for SimplifiedTraceInstruction {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "{:<10}{:<16}\t active={} \tpc={}",
                self.opcode,
                self.addresses,
                // self.addresses
                //     .as_ref()
                //     .map(ToString::to_string)
                //     .as_deref()
                //     .unwrap_or(""),
                self.active_mask,
                self.pc,
            )
        }
    }

    impl From<(usize, (&str, Addresses, &str, u32))> for SimplifiedTraceInstruction {
        fn from(value: (usize, (&str, Addresses, &str, u32))) -> Self {
            let (_idx, (opcode, addresses, active_mask, pc)) = value;
            assert_eq!(active_mask.len(), 32, "invalid active mask {active_mask:?}");
            Self {
                opcode: opcode.to_string(),
                addresses,
                active_mask: active_mask.to_string(),
                pc,
            }
        }
    }

    pub fn simplify_warp_instruction(
        warp_instr: &trace_model::MemAccessTraceEntry,
        relative: bool,
    ) -> SimplifiedTraceInstruction {
        let human_readable_active_mask = warp_instr
            .active_mask
            .to_string()
            .chars()
            .rev()
            .collect::<String>();

        let valid_addresses: Vec<(usize, u64)> = warp_instr
            .addrs
            .iter()
            .copied()
            .enumerate()
            .filter(|(ti, addr)| warp_instr.active_mask[*ti] && *addr > 0)
            .map(|(tid, valid_addr)| {
                let valid_addr = if relative {
                    let base_addr = warp_instr.instr_mem_space.base_addr();
                    valid_addr.checked_sub(base_addr).unwrap_or(valid_addr)
                } else {
                    valid_addr
                };
                (tid, valid_addr)
            })
            .collect();

        let addresses = if valid_addresses.is_empty() {
            Addresses::None
        } else if !valid_addresses.is_empty()
            && valid_addresses.iter().map(|(_, addr)| addr).all_equal()
        {
            let (_, addr) = valid_addresses.first().unwrap();
            Addresses::Uniform(*addr)
        } else {
            let strides = valid_addresses
                .windows(2)
                .map(|w| w[1].1 as i128 - w[0].1 as i128)
                .collect::<Vec<_>>();

            if !strides.is_empty() && strides.iter().all_equal() {
                Addresses::BaseStride {
                    base: valid_addresses[0].1 as i128,
                    stride: strides[0] as i128,
                }
            } else {
                let mut addresses = [0; WARP_SIZE];
                for (ti, valid_addr) in valid_addresses {
                    addresses[ti] = valid_addr;
                }
                Addresses::PerThread(addresses)
            }
        };

        SimplifiedTraceInstruction {
            opcode: warp_instr.instr_opcode.clone(),
            addresses,
            // first_addr: warp_instr
            //     .addrs
            //     .iter()
            //     .enumerate()
            //     .filter(|(ti, addr)| warp_instr.active_mask[*ti] && **addr > 0)
            //     .next()
            //     .map(|(_, addr)| {
            //         if relative {
            //             let base_addr = warp_instr.instr_mem_space.base_addr();
            //             addr.checked_sub(base_addr).unwrap_or(*addr)
            //         } else {
            //             *addr
            //         }
            //     }),
            active_mask: human_readable_active_mask,
            pc: warp_instr.instr_offset,
        }
    }

    pub fn simplify_warp_trace(
        warp_trace: &[trace_model::MemAccessTraceEntry],
        relative: bool,
    ) -> impl Iterator<Item = SimplifiedTraceInstruction> + '_ {
        warp_trace
            .iter()
            .map(move |inst| simplify_warp_instruction(inst, relative))
    }
}

#[cfg(test)]
mod tests {
    use super::fmt::{self, Addresses, SimplifiedTraceInstruction};
    use super::util::mem_inst;
    use super::{alloc, DevicePtr, ThreadBlock, ThreadIndex, TraceGenerator};
    use crate::model::MemorySpace;
    use color_eyre::eyre;
    use rand::Rng;
    use std::path::PathBuf;
    use tokio::sync::Mutex;
    use trace_model::Dim;
    use utils::diff;

    const EPSILON: f32 = 0.0001;
    use ndarray::Array2;

    fn get_reference_warp_traces(kernel_name: &str) -> eyre::Result<trace_model::WarpTraces> {
        let manifest = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
        let commands_file_path =
            manifest.join("../test-apps/microbenches/trace-reconstruction/traces/commands.json");

        let reader = utils::fs::open_readable(&commands_file_path)?;
        let commands: Vec<trace_model::Command> = serde_json::from_reader(reader)?;

        let have: Vec<String> = commands
            .iter()
            .filter_map(|cmd| match cmd {
                trace_model::Command::KernelLaunch(launch) => Some(launch.mangled_name.clone()),
                _ => None,
            })
            .collect();

        let kernel_launch = commands
            .into_iter()
            .find_map(|cmd| match cmd {
                trace_model::Command::KernelLaunch(launch)
                    if launch.mangled_name.contains(kernel_name) =>
                {
                    Some(launch)
                }
                _ => None,
            })
            .ok_or_else(|| {
                eyre::eyre!("no kernel with name {:?} (have {:?})", kernel_name, have)
            })?;

        dbg!(&kernel_launch);

        let kernel_trace_path = commands_file_path
            .parent()
            .unwrap()
            .join(kernel_launch.trace_file);
        dbg!(&kernel_trace_path);

        let mut reader = utils::fs::open_readable(&kernel_trace_path)?;
        let trace: trace_model::MemAccessTrace = rmp_serde::from_read(&mut reader)?;
        Ok(trace.to_warp_traces())
    }

    // #[macro_export]
    // macro_rules! function_name {
    //     () => {{
    //         // Okay, this is ugly, I get it. However, this is the best we can get on a stable rust.
    //         fn f() {}
    //         fn type_name_of<T>(_: T) -> &'static str {
    //             std::any::type_name::<T>()
    //         }
    //         let name = type_name_of(f);
    //         // 3 is the length of the "::f" suffix
    //         &name[..name.len() - 3]
    //     }};
    // }
    // let test = PathBuf::from(file!()).parent().unwrap().to_path_buf();

    pub fn testing_dir() -> PathBuf {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let testing_dir = manifest_dir.join("testing");
        testing_dir
    }

    pub fn render_graphs(
        cfg_iter: impl Iterator<Item = super::WarpTrace>,
        name: &str,
    ) -> Result<(), std::io::Error> {
        #[cfg(feature = "render")]
        {
            use crate::cfg::render::Render;
            let graphs_dir = testing_dir().join(name);
            std::fs::create_dir_all(&graphs_dir).ok();
            for (warp_id, warp_cfg, thread_cfgs) in cfg_iter.take(2) {
                // dbg!(&warp_id, &warp_cfg, &thread_cfgs);
                let super::WarpId {
                    block_id,
                    warp_id_in_block,
                } = warp_id;
                let name = format!(
                    "block_{:_>3}_{:_>3}_{:_>3}_warp_{:_>2}",
                    block_id.x, block_id.y, block_id.z, warp_id_in_block
                );
                let graph_path = graphs_dir.join(format!("{}.svg", name));
                dbg!(&graph_path);
                warp_cfg.render_to(&graph_path)?;

                for (tid, thread_cfg) in thread_cfgs.iter().enumerate() {
                    let graph_path = graphs_dir.join(format!("{}_thread_{:_>2}.svg", name, tid));
                    thread_cfg.render_to(&graph_path)?;
                }
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_single_for_loop_kernel() -> eyre::Result<()> {
        crate::tests::init_test();

        struct SingleForLoopKernel {}
        #[async_trait::async_trait]
        impl super::Kernel for SingleForLoopKernel {
            type Error = std::convert::Infallible;
            #[crate::instrument_control_flow]
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
        let options = super::Options::default();
        let (_launch_config, trace) = tracer
            .trace_kernel(1, 32, &mut SingleForLoopKernel {}, &options)
            .await?;
        let warp_traces = trace.clone().to_warp_traces();
        let first_warp = &warp_traces[&(Dim::ZERO, 0)];
        for inst in fmt::simplify_warp_trace(first_warp, true) {
            println!("{}", inst);
        }

        let ref_warp_traces = get_reference_warp_traces("single_for_loop")?;
        let ref_first_warp = &ref_warp_traces[&(Dim::ZERO, 0)];
        for inst in fmt::simplify_warp_trace(ref_first_warp, true) {
            println!("{}", inst);
        }

        let have: Vec<_> = fmt::simplify_warp_trace(first_warp, true).collect();
        let want: Vec<_> = [
            (
                "LDG.E",
                Addresses::Uniform(1),
                "11111111111111111111111111111111",
                0,
            ),
            (
                "LDG.E",
                Addresses::Uniform(1),
                "10101010101010101010101010101010",
                1,
            ),
            (
                "LDG.E",
                Addresses::Uniform(1),
                "10101010101010101010101010101010",
                2,
            ),
            (
                "LDG.E",
                Addresses::Uniform(100),
                "11111111111111111111111111111111",
                3,
            ),
            (
                "EXIT",
                Addresses::None,
                "11111111111111111111111111111111",
                4,
            ),
        ]
        .into_iter()
        .enumerate()
        .map(SimplifiedTraceInstruction::from)
        .collect();

        dbg!(&have);
        diff::assert_eq!(have: have, want: want);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_single_if_kernel() -> eyre::Result<()> {
        crate::tests::init_test();

        struct SingleIfKernel {}
        #[async_trait::async_trait]
        impl super::Kernel for SingleIfKernel {
            type Error = std::convert::Infallible;
            #[crate::instrument_control_flow]
            async fn run(&self, block: &ThreadBlock, tid: &ThreadIndex) -> Result<(), Self::Error> {
                block
                    .memory
                    .push_thread_instruction(tid, mem_inst!(Load[Global]@1, 4).into());
                if tid.thread_idx.x < 16 {
                    block
                        .memory
                        .push_thread_instruction(tid, mem_inst!(Load[Global]@2, 4).into());
                } else {
                    block
                        .memory
                        .push_thread_instruction(tid, mem_inst!(Load[Global]@3, 4).into());
                }
                block
                    .memory
                    .push_thread_instruction(tid, mem_inst!(Load[Global]@4, 4).into());
                Ok(())
            }
        }

        let tracer = super::Tracer::new();
        let options = super::Options::default();
        let (_launch_config, trace) = tracer
            .trace_kernel(1, 32, &mut SingleIfKernel {}, &options)
            .await?;
        let warp_traces = trace.clone().to_warp_traces();
        let first_warp = &warp_traces[&(Dim::ZERO, 0)];
        for inst in fmt::simplify_warp_trace(first_warp, true) {
            println!("{}", inst);
        }

        let ref_warp_traces = get_reference_warp_traces("single_if")?;
        let ref_first_warp = &ref_warp_traces[&(Dim::ZERO, 0)];
        for inst in fmt::simplify_warp_trace(ref_first_warp, true) {
            println!("{}", inst);
        }

        let have: Vec<_> = fmt::simplify_warp_trace(first_warp, true).collect();
        let want: Vec<_> = [
            (
                "LDG.E",
                Addresses::Uniform(1),
                "11111111111111111111111111111111",
                0,
            ),
            (
                "LDG.E",
                Addresses::Uniform(2),
                "11111111111111110000000000000000",
                1,
            ),
            (
                "LDG.E",
                Addresses::Uniform(3),
                "00000000000000001111111111111111",
                2,
            ),
            (
                "LDG.E",
                Addresses::Uniform(4),
                "11111111111111111111111111111111",
                3,
            ),
            (
                "EXIT",
                Addresses::None,
                "11111111111111111111111111111111",
                4,
            ),
        ]
        .into_iter()
        .enumerate()
        .map(SimplifiedTraceInstruction::from)
        .collect();

        dbg!(&have);
        diff::assert_eq!(have: have, want: want);
        Ok(())
    }

    // #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    // async fn test_full_imbalance() -> eyre::Result<()> {
    //     crate::tests::init_test();
    //
    //     struct FullImbalanceKernel {}
    //     #[async_trait::async_trait]
    //     impl super::Kernel for FullImbalanceKernel {
    //         type Error = std::convert::Infallible;
    //         #[crate::inject_reconvergence_points]
    //         async fn run(&self, block: &ThreadBlock, tid: &ThreadIndex) -> Result<(), Self::Error> {
    //             if tid.thread_idx.x < 16 {
    //                 if tid.thread_idx.x < 8 {
    //                     let inst = mem_inst!(Load[Global]@1, 4);
    //                     block.memory.push_thread_instruction(tid, inst.into());
    //                 }
    //                 let inst = mem_inst!(Load[Global]@2, 4);
    //                 block.memory.push_thread_instruction(tid, inst.into());
    //             } else {
    //                 if tid.thread_idx.x < 24 {
    //                     let inst = mem_inst!(Load[Global]@10, 4);
    //                     block.memory.push_thread_instruction(tid, inst.into());
    //                 }
    //                 let inst = mem_inst!(Load[Global]@11, 4);
    //                 block.memory.push_thread_instruction(tid, inst.into());
    //             }
    //             let inst = mem_inst!(Load[Global]@100, 4);
    //             block.memory.push_thread_instruction(tid, inst.into());
    //             Ok(())
    //         }
    //     }
    //
    //     let tracer = super::Tracer::new();
    //     let (_launch_config, trace) = tracer.trace_kernel(1, 32, FullImbalanceKernel {}).await?;
    //     let warp_traces = trace.clone().to_warp_traces();
    //     let first_warp = &warp_traces[&(Dim::ZERO, 0)];
    //     for inst in testing::simplify_warp_trace(first_warp) {
    //         println!("{}", inst);
    //     }
    //
    //     let ref_warp_traces = get_reference_warp_traces("full_imbalance")?;
    //     let ref_first_warp = &ref_warp_traces[&(Dim::ZERO, 0)];
    //     for inst in testing::simplify_warp_trace(ref_first_warp) {
    //         println!("{}", inst);
    //     }
    //
    //     diff::assert_eq!(
    //         have: testing::simplify_warp_trace(first_warp).collect::<Vec<_>>(),
    //         want: [
    //             ("LDG.E", 1, "11111111000000000000000000000000", 0),
    //             ("LDG.E", 2, "11111111111111110000000000000000", 0),
    //             ("LDG.E", 10, "00000000000000001111111100000000", 0),
    //             ("LDG.E", 11, "00000000000000001111111111111111", 0),
    //             ("LDG.E", 100, "11111111111111111111111111111111", 0),
    //             ("EXIT", 0, "11111111111111111111111111111111", 0),
    //         ].into_iter().enumerate().map(SimplifiedTraceInstruction::from).collect::<Vec<_>>()
    //     );
    //     Ok(())
    // }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_two_level_nested_for_loops_kernel() -> eyre::Result<()> {
        crate::tests::init_test();

        struct TwoLevelNestedForLoops {}
        #[async_trait::async_trait]
        impl super::Kernel for TwoLevelNestedForLoops {
            type Error = std::convert::Infallible;
            #[crate::instrument_control_flow]
            async fn run(&self, block: &ThreadBlock, tid: &ThreadIndex) -> Result<(), Self::Error> {
                // fully parallel
                let inst = mem_inst!(Load[Global]@100, 4);
                block.memory.push_thread_instruction(tid, inst.into());

                for j in 0..2 {
                    for i in 0..2 {
                        let addr = j * 2 + i;
                        let inst = mem_inst!(Store[Global]@addr, 4);
                        block.memory.push_thread_instruction(tid, inst.into());
                    }
                }

                // for j in 0..4 {
                //     for i in 0..8 {
                //         let inst = mem_inst!(Store[Global]@(j*8 + i), 4);
                //         block.memory.push_thread_instruction(tid, inst.into());
                //     }
                // }

                // have reconverged to fully parallel
                let inst = mem_inst!(Load[Global]@100, 4);
                block.memory.push_thread_instruction(tid, inst.into());
                Ok(())
            }
        }

        let tracer = super::Tracer::new();
        let options = super::Options::default();
        let (_launch_config, trace) = tracer
            .trace_kernel(1, 32, &mut TwoLevelNestedForLoops {}, &options)
            .await?;
        let warp_traces = trace.clone().to_warp_traces();
        let first_warp = &warp_traces[&(Dim::ZERO, 0)];
        for inst in fmt::simplify_warp_trace(first_warp, true) {
            println!("{}", inst);
        }

        let ref_warp_traces = get_reference_warp_traces("two_level_nested_if_balanced")?;
        let ref_first_warp = &ref_warp_traces[&(Dim::ZERO, 0)];
        for inst in fmt::simplify_warp_trace(ref_first_warp, true) {
            println!("{}", inst);
        }

        let have: Vec<_> = fmt::simplify_warp_trace(first_warp, true).collect();
        let want: Vec<_> = [
            (
                "LDG.E",
                Addresses::Uniform(100),
                "11111111111111111111111111111111",
                0,
            ),
            (
                "STG.E",
                Addresses::Uniform(0),
                "11111111111111111111111111111111",
                1,
            ),
            (
                "STG.E",
                Addresses::Uniform(1),
                "11111111111111111111111111111111",
                2,
            ),
            (
                "STG.E",
                Addresses::Uniform(2),
                "11111111111111111111111111111111",
                3,
            ),
            (
                "STG.E",
                Addresses::Uniform(3),
                "11111111111111111111111111111111",
                4,
            ),
            (
                "LDG.E",
                Addresses::Uniform(100),
                "11111111111111111111111111111111",
                5,
            ),
            (
                "EXIT",
                Addresses::None,
                "11111111111111111111111111111111",
                6,
            ),
        ]
        .into_iter()
        .enumerate()
        .map(SimplifiedTraceInstruction::from)
        .collect::<Vec<_>>();

        dbg!(&have);
        diff::assert_eq!(have: have, want: want);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_two_level_nested_multiple_serial_if_kernel() -> eyre::Result<()> {
        crate::tests::init_test();

        struct TwoLevelNestedMultipleSerialIf {}
        #[async_trait::async_trait]
        impl super::Kernel for TwoLevelNestedMultipleSerialIf {
            type Error = std::convert::Infallible;
            #[crate::instrument_control_flow]
            async fn run(&self, block: &ThreadBlock, tid: &ThreadIndex) -> Result<(), Self::Error> {
                // fully parallel
                let inst = mem_inst!(Load[Global]@100, 4);
                block.memory.push_thread_instruction(tid, inst.into());

                if tid.thread_idx.x < 16 {
                    if tid.thread_idx.x < 8 {
                        let inst = mem_inst!(Load[Global]@10, 4);
                        block.memory.push_thread_instruction(tid, inst.into());
                    }
                    if tid.thread_idx.x >= 8 {
                        let inst = mem_inst!(Load[Global]@11, 4);
                        block.memory.push_thread_instruction(tid, inst.into());
                    }
                } else {
                    // let inst = mem_inst!(Load[Global]@20, 4);
                    // block.memory.push_thread_instruction(tid, inst.into());
                    if tid.thread_idx.x < 24 {
                        let inst = mem_inst!(Load[Global]@20, 4);
                        block.memory.push_thread_instruction(tid, inst.into());
                    }
                    if tid.thread_idx.x >= 24 {
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
        let options = super::Options::default();
        let (_launch_config, trace) = tracer
            .trace_kernel(1, 32, &mut TwoLevelNestedMultipleSerialIf {}, &options)
            .await?;
        let warp_traces = trace.clone().to_warp_traces();
        let first_warp = &warp_traces[&(Dim::ZERO, 0)];
        for inst in fmt::simplify_warp_trace(first_warp, true) {
            println!("{}", inst);
        }

        let ref_warp_traces = get_reference_warp_traces("two_level_nested_if_balanced")?;
        let ref_first_warp = &ref_warp_traces[&(Dim::ZERO, 0)];
        for inst in fmt::simplify_warp_trace(ref_first_warp, true) {
            println!("{}", inst);
        }

        let have: Vec<_> = fmt::simplify_warp_trace(first_warp, true).collect();
        let want: Vec<_> = [
            (
                "LDG.E",
                Addresses::Uniform(100),
                "11111111111111111111111111111111",
                0,
            ),
            (
                "LDG.E",
                Addresses::Uniform(10),
                "11111111000000000000000000000000",
                1,
            ),
            (
                "LDG.E",
                Addresses::Uniform(11),
                "00000000111111110000000000000000",
                2,
            ),
            (
                "LDG.E",
                Addresses::Uniform(20),
                "00000000000000001111111100000000",
                3,
            ),
            (
                "LDG.E",
                Addresses::Uniform(21),
                "00000000000000000000000011111111",
                4,
            ),
            (
                "LDG.E",
                Addresses::Uniform(100),
                "11111111111111111111111111111111",
                5,
            ),
            (
                "EXIT",
                Addresses::None,
                "11111111111111111111111111111111",
                6,
            ),
        ]
        .into_iter()
        .enumerate()
        .map(SimplifiedTraceInstruction::from)
        .collect();

        dbg!(&have);
        diff::assert_eq!(have: have, want: want);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_two_level_nested_if_balanced_kernel() -> eyre::Result<()> {
        crate::tests::init_test();

        struct Balanced {}
        #[async_trait::async_trait]
        impl super::Kernel for Balanced {
            type Error = std::convert::Infallible;
            #[crate::instrument_control_flow]
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
                    if tid.thread_idx.x >= 24 {
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
        let options = &super::Options::default();
        let (_launch_config, trace) = tracer
            .trace_kernel(1, 32, &mut Balanced {}, &options)
            .await?;
        let warp_traces = trace.clone().to_warp_traces();
        let first_warp = &warp_traces[&(Dim::ZERO, 0)];
        for inst in fmt::simplify_warp_trace(first_warp, true) {
            println!("{}", inst);
        }

        let ref_warp_traces = get_reference_warp_traces("two_level_nested_if_balanced")?;
        let ref_first_warp = &ref_warp_traces[&(Dim::ZERO, 0)];
        for inst in fmt::simplify_warp_trace(ref_first_warp, true) {
            println!("{}", inst);
        }

        let have: Vec<_> = fmt::simplify_warp_trace(first_warp, true).collect();
        let want: Vec<_> = [
            (
                "LDG.E",
                Addresses::Uniform(100),
                "11111111111111111111111111111111",
                0,
            ),
            (
                "LDG.E",
                Addresses::Uniform(10),
                "11111111111111110000000000000000",
                1,
            ),
            (
                "LDG.E",
                Addresses::Uniform(11),
                "11111111000000000000000000000000",
                2,
            ),
            (
                "LDG.E",
                Addresses::Uniform(20),
                "00000000000000001111111111111111",
                3,
            ),
            (
                "LDG.E",
                Addresses::Uniform(21),
                "00000000000000000000000011111111",
                4,
            ),
            (
                "LDG.E",
                Addresses::Uniform(100),
                "11111111111111111111111111111111",
                5,
            ),
            (
                "EXIT",
                Addresses::None,
                "11111111111111111111111111111111",
                6,
            ),
        ]
        .into_iter()
        .enumerate()
        .map(SimplifiedTraceInstruction::from)
        .collect();

        dbg!(&have);
        diff::assert_eq!(have: have, want: want);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_two_level_nested_if_imbalanced_kernel() -> eyre::Result<()> {
        crate::tests::init_test();

        struct Imbalanced {}
        #[async_trait::async_trait]
        impl super::Kernel for Imbalanced {
            type Error = std::convert::Infallible;
            #[crate::instrument_control_flow]
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

        let tracer = super::Tracer::new();
        let options = &super::Options::default();
        let (_launch_config, trace) = tracer
            .trace_kernel(1, 32, &mut Imbalanced {}, &options)
            .await?;
        let warp_traces = trace.clone().to_warp_traces();
        let first_warp = &warp_traces[&(Dim::ZERO, 0)];
        for inst in fmt::simplify_warp_trace(first_warp, true) {
            println!("{}", inst);
        }

        let ref_warp_traces = get_reference_warp_traces("two_level_nested_if_imbalanced")?;
        let ref_first_warp = &ref_warp_traces[&(Dim::ZERO, 0)];
        for inst in fmt::simplify_warp_trace(ref_first_warp, true) {
            println!("{}", inst);
        }

        let have: Vec<_> = fmt::simplify_warp_trace(first_warp, true).collect();
        let want: Vec<_> = [
            (
                "LDG.E",
                Addresses::Uniform(100),
                "11111111111111111111111111111111",
                0,
            ),
            (
                "LDG.E",
                Addresses::Uniform(1),
                "11111111111111110000000000000000",
                1,
            ),
            (
                "LDG.E",
                Addresses::Uniform(2),
                "11111111000000000000000000000000",
                2,
            ),
            (
                "LDG.E",
                Addresses::Uniform(100),
                "11111111111111111111111111111111",
                3,
            ),
            (
                "EXIT",
                Addresses::None,
                "11111111111111111111111111111111",
                4,
            ),
        ]
        .into_iter()
        .enumerate()
        .map(SimplifiedTraceInstruction::from)
        .collect();

        dbg!(&have);
        diff::assert_eq!(have: have, want: want);
        Ok(())
    }

    #[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_vectoradd() -> eyre::Result<()> {
        crate::tests::init_test();

        fn reference_vectoradd(a: &[f32], b: &[f32], result: &mut [f32]) {
            for (i, sum) in result.iter_mut().enumerate() {
                *sum = a[i] + b[i];
            }
        }

        #[derive(Debug)]
        struct VecAdd<'a> {
            dev_a: Mutex<DevicePtr<&'a mut Vec<f32>>>,
            dev_b: Mutex<DevicePtr<&'a mut Vec<f32>>>,
            dev_result: Mutex<DevicePtr<&'a mut Vec<f32>>>,
            n: usize,
        }

        #[async_trait::async_trait]
        impl<'a> super::Kernel for VecAdd<'a> {
            type Error = std::convert::Infallible;

            #[crate::instrument_control_flow]
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
                }
                block.synchronize_threads().await;
                Ok(())
            }
        }

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

        let dev_a = tracer
            .allocate(
                &mut a,
                Some(alloc::Options {
                    mem_space: MemorySpace::Global,
                    name: Some("a".to_string()),
                    ..alloc::Options::default()
                }),
            )
            .await;
        let dev_b = tracer
            .allocate(
                &mut b,
                Some(alloc::Options {
                    mem_space: MemorySpace::Global,
                    name: Some("b".to_string()),
                    ..alloc::Options::default()
                }),
            )
            .await;
        let dev_result = tracer
            .allocate(
                &mut result,
                Some(alloc::Options {
                    mem_space: MemorySpace::Global,
                    name: Some("result".to_string()),
                    ..alloc::Options::default()
                }),
            )
            .await;

        let mut kernel = VecAdd {
            dev_a: Mutex::new(dev_a),
            dev_b: Mutex::new(dev_b),
            dev_result: Mutex::new(dev_result),
            n,
        };
        let grid_size = (n as f64 / f64::from(block_size)).ceil() as u32;
        let options = super::Options::default();

        let grid: Dim = grid_size.into();
        let block_size: Dim = block_size.into();

        let cfg_iter = tracer
            .trace_control_flow_graphs(&grid, &block_size, &mut kernel, 0)
            .await?;
        render_graphs(cfg_iter, "vectoradd")?;

        let (_launch_config, trace) = tracer
            .trace_kernel(grid, block_size, &mut kernel, &options)
            .await?;

        let warp_traces = trace.clone().to_warp_traces();
        let first_warp = &warp_traces[&(Dim::ZERO, 0)];

        reference_vectoradd(&a, &b, &mut ref_result);
        diff::assert_eq!(have: result, want: ref_result);
        for inst in fmt::simplify_warp_trace(first_warp, true) {
            println!("{}", inst);
        }

        let have: Vec<_> = fmt::simplify_warp_trace(first_warp, true).collect();
        let want: Vec<_> = [
            (
                "MEMBAR",
                Addresses::None,
                "11111111111111111111111111111111",
                0,
            ),
            (
                "LDG.E",
                Addresses::BaseStride { base: 0, stride: 4 },
                "11111111111111111111000000000000",
                1,
            ),
            (
                "LDG.E",
                Addresses::BaseStride {
                    base: 256,
                    stride: 4,
                },
                "11111111111111111111000000000000",
                2,
            ),
            (
                "STG.E",
                Addresses::BaseStride {
                    base: 512,
                    stride: 4,
                },
                "11111111111111111111000000000000",
                3,
            ),
            (
                "MEMBAR",
                Addresses::None,
                "11111111111111111111111111111111",
                4,
            ),
            (
                "EXIT",
                Addresses::None,
                "11111111111111111111111111111111",
                5,
            ),
        ]
        .into_iter()
        .enumerate()
        .map(SimplifiedTraceInstruction::from)
        .collect();

        dbg!(&have);
        diff::assert_eq!(have: have, want: want);
        Ok(())
    }

    #[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_transpose_coalesced() -> eyre::Result<()> {
        crate::tests::init_test();

        fn reference_transpose(mat: &[f32], result: &mut [f32], rows: usize, cols: usize) {
            assert_eq!(mat.len(), result.len());
            for y in 0..rows {
                for x in 0..cols {
                    result[(x * rows) + y] = mat[(y * cols) + x];
                }
            }
        }

        #[derive(Debug)]
        struct TransposeCoalesced<'a> {
            pub dev_mat: Mutex<alloc::DevicePtr<&'a Vec<f32>>>,
            pub dev_result: Mutex<alloc::DevicePtr<&'a mut Vec<f32>>>,
            pub rows: usize,
            pub cols: usize,

            /// Shared memory array used to store the tiles
            pub shared_mem_tiles: Mutex<alloc::DevicePtr<Vec<f32>>>,
        }

        const TILE_DIM: u32 = 16;
        const BLOCK_ROWS: u32 = 16;

        #[async_trait::async_trait]
        impl<'a> super::Kernel for TransposeCoalesced<'a> {
            type Error = std::convert::Infallible;

            #[crate::instrument_control_flow]
            async fn run(&self, block: &ThreadBlock, tid: &ThreadIndex) -> Result<(), Self::Error> {
                let x_index = (tid.block_idx.x * TILE_DIM + tid.thread_idx.x) as usize;
                let y_index = (tid.block_idx.y * TILE_DIM + tid.thread_idx.y) as usize;
                let index_in = x_index + y_index * self.cols;

                let x_index = (tid.block_idx.y * TILE_DIM + tid.thread_idx.x) as usize;
                let y_index = (tid.block_idx.x * TILE_DIM + tid.thread_idx.y) as usize;
                let index_out = x_index + y_index * self.rows;

                // load into shared memory
                let mut i = 0;
                while i < TILE_DIM {
                    let dev_mat = self.dev_mat.lock().await;
                    let mut tiles = self.shared_mem_tiles.lock().await;

                    let tile_idx = (tid.thread_idx.y + i) * TILE_DIM + tid.thread_idx.x;
                    let mat_idx = index_in + i as usize * self.cols;
                    // eprintln!(
                    //     "block {:>15} warp {:>3} thread {:>3} {:>15} => load {:>3}",
                    //     tid.block_idx.to_string(),
                    //     tid.warp_id_in_block,
                    //     tid.thread_id_in_warp,
                    //     tid.thread_idx.to_string(),
                    //     mat_idx
                    // );
                    tiles[(tid, tile_idx as usize)] = dev_mat[(tid, mat_idx)];
                    i += BLOCK_ROWS;
                }

                block.synchronize_threads().await;

                let mut i = 0;
                while i < TILE_DIM {
                    let mut dev_result = self.dev_result.lock().await;
                    let tiles = self.shared_mem_tiles.lock().await;
                    let tile_idx = tid.thread_idx.x * TILE_DIM + tid.thread_idx.y + i;

                    let result_idx = index_out + i as usize * self.rows;
                    dev_result[(tid, result_idx)] = tiles[(tid, tile_idx as usize)];
                    i += BLOCK_ROWS;
                }
                Ok(())
            }
        }

        let dim = 16;

        let tracer = super::Tracer::new();

        let mut mat: Vec<f32> = vec![0.0; dim * dim];
        let mut result: Vec<f32> = vec![0.0; dim * dim];
        let mut ref_result: Vec<f32> = vec![0.0; dim * dim];

        for (i, v) in mat.iter_mut().enumerate() {
            *v = i as f32;
        }

        let dev_mat = tracer
            .allocate(
                &mat,
                Some(alloc::Options {
                    mem_space: MemorySpace::Global,
                    name: Some("matrix".to_string()),
                    ..alloc::Options::default()
                }),
            )
            .await;

        let dev_result = tracer
            .allocate(
                &mut result,
                Some(alloc::Options {
                    mem_space: MemorySpace::Global,
                    name: Some("result".to_string()),
                    ..alloc::Options::default()
                }),
            )
            .await;

        // shared memory
        let shared_mem_tiles = vec![0.0; (TILE_DIM * TILE_DIM) as usize];
        let shared_mem_tiles = tracer
            .allocate(
                shared_mem_tiles,
                Some(alloc::Options {
                    mem_space: MemorySpace::Shared,
                    name: Some("shared_mem_tiles".to_string()),
                    ..alloc::Options::default()
                }),
            )
            .await;

        let mut kernel = TransposeCoalesced {
            dev_mat: Mutex::new(dev_mat),
            dev_result: Mutex::new(dev_result),
            shared_mem_tiles: Mutex::new(shared_mem_tiles),
            rows: dim,
            cols: dim,
        };

        let block_size: Dim = (TILE_DIM, BLOCK_ROWS).into();
        let grid_x = dim / TILE_DIM as usize;
        let grid_y = dim / TILE_DIM as usize;
        let grid: Dim = (grid_x as u32, grid_y as u32).into();
        println!("grid dim:  {grid}");
        println!("block dim: {block_size}");

        assert!(grid.x > 0);
        assert!(grid.y > 0);
        assert!(grid.z > 0);

        let options = super::Options {
            no_data_dependency: false,
        };

        let cfg_iter = tracer
            .trace_control_flow_graphs(&grid, &block_size, &mut kernel, 0)
            .await?;
        render_graphs(cfg_iter, "transpose_coalesced")?;

        let (_launch_config, trace) = tracer
            .trace_kernel(grid, block_size, &mut kernel, &options)
            .await?;

        let warp_traces = trace.clone().to_warp_traces();
        let first_warp = &warp_traces[&(trace_model::Dim::ZERO, 0)];

        let have: Vec<_> = fmt::simplify_warp_trace(first_warp, true).collect();
        let want: Vec<_> = [
            (
                "LDG.E",
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
                "MEMBAR",
                Addresses::None,
                "11111111111111111111111111111111",
                2,
            ),
            (
                "LDS.E",
                Addresses::PerThread([
                    0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 4,
                    68, 132, 196, 260, 324, 388, 452, 516, 580, 644, 708, 772, 836, 900, 964,
                ]),
                "11111111111111111111111111111111",
                3,
            ),
            (
                "STG.E",
                Addresses::BaseStride {
                    base: 1024,
                    stride: 4,
                },
                "11111111111111111111111111111111",
                4,
            ),
            (
                "EXIT",
                Addresses::None,
                "11111111111111111111111111111111",
                5,
            ),
        ]
        .into_iter()
        .enumerate()
        .map(SimplifiedTraceInstruction::from)
        .collect();

        dbg!(&have);
        diff::assert_eq!(have: have, want: want);

        // check functional correctness
        reference_transpose(&mat, &mut ref_result, dim, dim);
        let ref_result = Array2::from_shape_vec((dim, dim), ref_result)?;
        let result = Array2::from_shape_vec((dim, dim), result)?;
        dbg!(&ref_result);
        dbg!(&result);
        if !approx::abs_diff_eq!(result, ref_result, epsilon = EPSILON) {
            diff::assert_eq!(have: result, want: ref_result);
        }

        Ok(())
    }

    #[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_tiled_matrixmul() -> eyre::Result<()> {
        crate::tests::init_test();

        #[derive(Debug)]
        struct TiledMatrixul<'a> {
            dev_a: Mutex<alloc::DevicePtr<&'a Vec<f32>>>,
            dev_b: Mutex<alloc::DevicePtr<&'a Vec<f32>>>,
            dev_result: Mutex<alloc::DevicePtr<&'a mut Vec<f32>>>,
            num_rows: usize,
            /// Shared memory array used to store the sub-matrix of A
            shared_mem_a: Mutex<alloc::DevicePtr<Vec<f32>>>,
            /// Shared memory array used to store the sub-matrix of B
            shared_mem_b: Mutex<alloc::DevicePtr<Vec<f32>>>,
        }

        const BLOCK_SIZE: usize = 4;

        #[async_trait::async_trait]
        impl<'a> super::Kernel for TiledMatrixul<'a> {
            type Error = std::convert::Infallible;

            #[crate::instrument_control_flow]
            async fn run(&self, block: &ThreadBlock, tid: &ThreadIndex) -> Result<(), Self::Error> {
                let bx = tid.block_idx.x as usize;
                let by = tid.block_idx.y as usize;

                let tx = tid.thread_idx.x as usize;
                let ty = tid.thread_idx.y as usize;

                // index of the first sub-matrix of A processed by the block
                let a_begin = self.num_rows * BLOCK_SIZE * by;

                // index of the last sub-matrix of A processed by the block
                let a_end = a_begin + self.num_rows - 1;

                // step size used to iterate through the sub-matrices of A
                let a_step = BLOCK_SIZE;

                // index of the first sub-matrix of B processed by the block
                let b_begin = BLOCK_SIZE * bx;

                // step size used to iterate through the sub-matrices of B
                let b_step = BLOCK_SIZE * self.num_rows;

                // c_sub is used to store the element of the block sub-matrix
                // that is computed by the thread
                let mut c_sub = 0.0;

                // Loop over all the sub-matrices of A and B
                // required to compute the block sub-matrix

                let mut ai = a_begin;
                let mut bi = b_begin;
                while ai <= a_end {
                    // dbg!(&tid.block_idx, tid.thread_id_in_warp, ai, bi);
                    {
                        // load the matrices from device memory to shared memory
                        // each thread loads one element of each matrix

                        // As[ty][tx] = A[a + wA * ty + tx];
                        let a = self.dev_a.lock().await;
                        let mut shared_a = self.shared_mem_a.lock().await;
                        shared_a[(tid, ty * BLOCK_SIZE + tx)] =
                            a[(tid, ai + self.num_rows * ty + tx)];

                        // Bs[ty][tx] = B[b + wB * ty + tx];
                        let b = self.dev_b.lock().await;
                        let mut shared_b = self.shared_mem_b.lock().await;
                        shared_b[(tid, ty * BLOCK_SIZE + tx)] =
                            b[(tid, bi + self.num_rows * ty + tx)];
                    }

                    block.synchronize_threads().await;

                    for k in 0..BLOCK_SIZE {
                        let shared_a = self.shared_mem_a.lock().await;
                        let shared_b = self.shared_mem_b.lock().await;
                        c_sub += shared_a[(tid, ty * BLOCK_SIZE + k)]
                            * shared_b[(tid, k * BLOCK_SIZE + tx)];
                    }

                    // Synchronize to make sure that the preceding
                    // computation is done before loading two new
                    // sub-matrices of A and B in the next iteration
                    block.synchronize_threads().await;

                    ai += a_step;
                    bi += b_step;
                }

                // Write the block sub-matrix to device memory;
                // each thread writes one element
                let c = self.num_rows * BLOCK_SIZE * by + BLOCK_SIZE * bx;
                let mut result = self.dev_result.lock().await;
                result[(tid, c + self.num_rows * ty + tx)] = c_sub;

                Ok(())
            }
        }

        let num_rows = 4;
        let matrix_shape = (num_rows, num_rows);
        let matrix_size = num_rows * num_rows;

        let tracer = super::Tracer::new();

        let mut a: Vec<f32> = vec![0.0; matrix_size];
        let mut b: Vec<f32> = vec![0.0; matrix_size];
        let mut result: Vec<f32> = vec![0.0; matrix_size];

        // initialize vectors
        let mut rng = rand::thread_rng();
        for i in 0..matrix_size {
            a[i] = 1.0 + rng.gen_range(0.0..1.0);
            b[i] = 1.0 + rng.gen_range(0.0..1.0);
        }

        let ndarray_result = {
            let ref_a = Array2::from_shape_vec(matrix_shape, a.clone())?;
            let ref_b = Array2::from_shape_vec(matrix_shape, b.clone())?;
            ref_a.dot(&ref_b)
        };

        let dev_a = tracer
            .allocate(
                &a,
                Some(alloc::Options {
                    mem_space: MemorySpace::Global,
                    name: Some("a".to_string()),
                    ..alloc::Options::default()
                }),
            )
            .await;

        let dev_b = tracer
            .allocate(
                &b,
                Some(alloc::Options {
                    mem_space: MemorySpace::Global,
                    name: Some("a".to_string()),
                    ..alloc::Options::default()
                }),
            )
            .await;

        let dev_result = tracer
            .allocate(
                &mut result,
                Some(alloc::Options {
                    mem_space: MemorySpace::Global,
                    name: Some("result".to_string()),
                    ..alloc::Options::default()
                }),
            )
            .await;

        // shared memory
        let shared_mem_a = vec![0.0; (BLOCK_SIZE * BLOCK_SIZE) as usize];
        let shared_mem_a = tracer
            .allocate(
                shared_mem_a,
                Some(alloc::Options {
                    mem_space: MemorySpace::Shared,
                    name: Some("shared_a".to_string()),
                    ..alloc::Options::default()
                }),
            )
            .await;

        let shared_mem_b = vec![0.0; (BLOCK_SIZE * BLOCK_SIZE) as usize];
        let shared_mem_b = tracer
            .allocate(
                shared_mem_b,
                Some(alloc::Options {
                    mem_space: MemorySpace::Shared,
                    name: Some("shared_b".to_string()),
                    ..alloc::Options::default()
                }),
            )
            .await;

        // number of thread blocks in grid
        let block_size: Dim = (BLOCK_SIZE as u32, BLOCK_SIZE as u32).into();
        let grid_size = (num_rows + (BLOCK_SIZE - 1)) / BLOCK_SIZE;
        let grid: Dim = (grid_size as u32, grid_size as u32).into();
        println!("grid dim:  {grid}");
        println!("block dim: {block_size}");

        assert!(grid.x > 0);
        assert!(grid.y > 0);
        assert!(grid.z > 0);

        let mut kernel = TiledMatrixul {
            dev_a: Mutex::new(dev_a),
            dev_b: Mutex::new(dev_b),
            dev_result: Mutex::new(dev_result),
            shared_mem_a: Mutex::new(shared_mem_a),
            shared_mem_b: Mutex::new(shared_mem_b),
            num_rows,
        };
        let options = super::Options::default();

        let cfg_iter = tracer
            .trace_control_flow_graphs(&grid, &block_size, &mut kernel, 0)
            .await?;
        render_graphs(cfg_iter, "tiled_matrixmul")?;

        let (_launch_config, trace) = tracer
            .trace_kernel(grid, block_size, &mut kernel, &options)
            .await?;

        let warp_traces = trace.clone().to_warp_traces();
        let first_warp = &warp_traces[&(trace_model::Dim::ZERO, 0)];

        let have: Vec<_> = fmt::simplify_warp_trace(first_warp, true).collect();
        let want: Vec<_> = [
            (
                // load row of sub a
                "LDG.E",
                Addresses::BaseStride { base: 0, stride: 4 },
                "11111111111111110000000000000000",
                0,
            ),
            (
                // store row of sub a
                "STS.E",
                Addresses::BaseStride { base: 0, stride: 4 },
                "11111111111111110000000000000000",
                1,
            ),
            (
                // load row of sub b
                "LDG.E",
                Addresses::BaseStride {
                    base: 256,
                    stride: 4,
                },
                "11111111111111110000000000000000",
                2,
            ),
            (
                // store row of sub b
                "STS.E",
                Addresses::BaseStride {
                    base: 256,
                    stride: 4,
                },
                "11111111111111110000000000000000",
                3,
            ),
            (
                "MEMBAR",
                Addresses::None,
                "11111111111111110000000000000000",
                4,
            ),
            (
                "LDS.E",
                Addresses::PerThread([
                    0, 0, 0, 0, 16, 16, 16, 16, 32, 32, 32, 32, 48, 48, 48, 48, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ]),
                "11111111111111110000000000000000",
                5,
            ),
            (
                "LDS.E",
                Addresses::PerThread([
                    256, 260, 264, 268, 256, 260, 264, 268, 256, 260, 264, 268, 256, 260, 264, 268,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ]),
                "11111111111111110000000000000000",
                6,
            ),
            (
                "LDS.E",
                Addresses::PerThread([
                    4, 4, 4, 4, 20, 20, 20, 20, 36, 36, 36, 36, 52, 52, 52, 52, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ]),
                "11111111111111110000000000000000",
                7,
            ),
            (
                "LDS.E",
                Addresses::PerThread([
                    272, 276, 280, 284, 272, 276, 280, 284, 272, 276, 280, 284, 272, 276, 280, 284,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ]),
                "11111111111111110000000000000000",
                8,
            ),
            (
                "LDS.E",
                Addresses::PerThread([
                    8, 8, 8, 8, 24, 24, 24, 24, 40, 40, 40, 40, 56, 56, 56, 56, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ]),
                "11111111111111110000000000000000",
                9,
            ),
            (
                "LDS.E",
                Addresses::PerThread([
                    288, 292, 296, 300, 288, 292, 296, 300, 288, 292, 296, 300, 288, 292, 296, 300,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ]),
                "11111111111111110000000000000000",
                10,
            ),
            (
                "LDS.E",
                Addresses::PerThread([
                    12, 12, 12, 12, 28, 28, 28, 28, 44, 44, 44, 44, 60, 60, 60, 60, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ]),
                "11111111111111110000000000000000",
                11,
            ),
            (
                "LDS.E",
                Addresses::PerThread([
                    304, 308, 312, 316, 304, 308, 312, 316, 304, 308, 312, 316, 304, 308, 312, 316,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ]),
                "11111111111111110000000000000000",
                12,
            ),
            (
                "MEMBAR",
                Addresses::None,
                "11111111111111110000000000000000",
                13,
            ),
            (
                "STG.E",
                Addresses::BaseStride {
                    base: 512,
                    stride: 4,
                },
                "11111111111111110000000000000000",
                14,
            ),
            (
                "STG.E",
                Addresses::BaseStride {
                    base: 512,
                    stride: 4,
                },
                "11111111111111110000000000000000",
                15,
            ),
            (
                "EXIT",
                Addresses::None,
                "11111111111111111111111111111111",
                16,
            ),
        ]
        .into_iter()
        .enumerate()
        .map(SimplifiedTraceInstruction::from)
        .collect();

        dbg!(&have);
        diff::assert_eq!(have: have, want: want);

        // check functional correctness
        let result = Array2::from_shape_vec(matrix_shape, result)?;
        dbg!(&ndarray_result);
        dbg!(&result);
        if !approx::abs_diff_eq!(result, ndarray_result, epsilon = EPSILON) {
            diff::assert_eq!(have: result, want: ndarray_result);
        }

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
