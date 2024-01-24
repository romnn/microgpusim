use super::alloc::{self, Allocatable, DevicePtr};
use super::cfg::{self, UniqueGraph};
use super::kernel::{Kernel, ThreadBlock, ThreadIndex};
use super::model::{self, MemInstruction, ThreadInstruction};
use futures::StreamExt;
use itertools::Itertools;
use std::collections::HashMap;
use std::sync::{atomic, Arc};
use std::time::Instant;
use tokio::sync::Mutex;
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

#[async_trait::async_trait]
pub trait TraceGenerator {
    type Error;

    /// Trace kernel.
    async fn trace_kernel<G, B, K>(
        self: &Arc<Self>,
        grid: G,
        block_size: B,
        kernel: &mut K,
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
}

/// Texture allocation byte alignment (cudaDeviceProp::textureAlignment)
///
/// Guaranteed to be 256B or greater.
/// On Pascal GTX1080, its 512B.
pub const ALIGNMENT_BYTES: u64 = 256;

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

    #[allow(clippy::too_many_lines)]
    async fn trace_kernel<G, B, K>(
        self: &Arc<Self>,
        grid: G,
        block_size: B,
        kernel: &mut K,
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

        for (
            WarpId {
                block_id,
                warp_id_in_block,
            },
            per_thread_instructions,
        ) in traced_instructions.into_iter()
        {
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

            let mut super_cfg = cfg::CFG::new();
            let super_cfg_root_node_idx = super_cfg.add_unique_node(cfg::Node::Branch {
                id: 0,
                branch_id: 0,
            });

            let start = Instant::now();
            let mut thread_graphs = [(); WARP_SIZE as usize].map(|_| cfg::ThreadCFG::default());
            for (ti, thread_instructions) in per_thread_instructions.iter().enumerate() {
                let (thread_cfg, (thread_cfg_root_node_idx, thread_cfg_sink_node_idx)) =
                    cfg::build_control_flow_graph(thread_instructions, &mut super_cfg);

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
                        ti,
                        cfg::format_control_flow_path(&thread_cfg, &paths[0]).join(" ")
                    );
                }

                thread_graphs[ti] = thread_cfg;
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
            // cfg::add_missing_control_flow_edges(&mut super_cfg);

            let mut unique_control_flow_path_count: Option<usize> = None;
            #[cfg(debug_assertions)]
            if false {
                let super_cfg_sink_node_idx = super_cfg
                    .find_node(&cfg::Node::Reconverge {
                        id: 0,
                        branch_id: 0,
                    })
                    .unwrap();

                unique_control_flow_path_count = Some(
                    super::cfg::all_simple_paths::<Vec<_>, _>(
                        &super_cfg,
                        super_cfg_root_node_idx,
                        super_cfg_sink_node_idx,
                    )
                    .count(),
                );
            };

            log::debug!(
                "super CFG: {} nodes, {} edges, {} edge weights, {} unique control flow paths",
                super_cfg.node_count(),
                super_cfg.edge_count(),
                super_cfg.edge_weights().count(),
                unique_control_flow_path_count
                    .as_ref()
                    .map(ToString::to_string)
                    .unwrap_or("?".to_string()),
            );

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
                active_mask: trace_model::ActiveMask::ZERO,
                addrs: [0; 32],
                thread_indices: [(0, 0, 0); 32],
            };

            let mut pc = 0;

            let iter = cfg::visit::DominatedDfs::new(&super_cfg, super_cfg_root_node_idx);

            for (edge_idx, node_idx) in iter {
                let took_branch = super_cfg[edge_idx];
                log::trace!(
                    "trace assembly: node={} took branch={}",
                    super_cfg[node_idx],
                    took_branch
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
                    active_threads(&thread_graphs, &super_cfg[node_idx], took_branch).collect();

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
                    .map(|(instr_idx, access)| {
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
                            instr_is_extended: true,
                            instr_mem_space: access.mem_space.into(),
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
                            // TODO: find a way to add the thread idx here
                            // instr.thread_indices[*tid] = access.addr;
                            instr.instr_offset = pc;

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
                    pc += 1;
                }

                // here we have the warp_trace ready to be added into the global trace
                for inst in fmt::simplify_warp_trace(&branch_trace, true) {
                    log::trace!("{}", inst);
                }

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
                instr_offset: pc,
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
    #[derive(Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub struct SimplifiedTraceInstruction {
        opcode: String,
        first_addr: Option<u64>,
        active_mask: String,
        pc: u32,
    }

    impl std::fmt::Display for SimplifiedTraceInstruction {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "{:<10}{:<16}\t active={} \tpc={}",
                self.opcode,
                self.first_addr
                    .as_ref()
                    .map(ToString::to_string)
                    .as_deref()
                    .unwrap_or(""),
                self.active_mask,
                self.pc,
            )
        }
    }

    impl From<(usize, (&str, Option<u64>, &str, u32))> for SimplifiedTraceInstruction {
        fn from(value: (usize, (&str, Option<u64>, &str, u32))) -> Self {
            let (_idx, (opcode, first_addr, active_mask, pc)) = value;
            assert_eq!(active_mask.len(), 32, "invalid active mask {active_mask:?}");
            Self {
                opcode: opcode.to_string(),
                first_addr,
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

        SimplifiedTraceInstruction {
            opcode: warp_instr.instr_opcode.clone(),
            first_addr: warp_instr
                .addrs
                .iter()
                .enumerate()
                .filter(|(ti, addr)| warp_instr.active_mask[*ti] && **addr > 0)
                .next()
                .map(|(_, addr)| {
                    if relative {
                        let base_addr = warp_instr.instr_mem_space.base_addr();
                        addr.checked_sub(base_addr).unwrap_or(*addr)
                    } else {
                        *addr
                    }
                }),
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
    use super::fmt::{self, SimplifiedTraceInstruction};
    use super::util::mem_inst;
    use super::{alloc, DevicePtr, ThreadBlock, ThreadIndex, TraceGenerator};
    use crate::model::MemorySpace;
    use color_eyre::eyre;
    use num_traits::Float;
    use tokio::sync::Mutex;
    use utils::diff;

    fn get_reference_warp_traces(kernel_name: &str) -> eyre::Result<trace_model::WarpTraces> {
        use std::path::PathBuf;

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
        let (_launch_config, trace) = tracer
            .trace_kernel(1, 32, &mut SingleForLoopKernel {})
            .await?;
        let warp_traces = trace.clone().to_warp_traces();
        let first_warp = &warp_traces[&(trace_model::Dim::ZERO, 0)];
        for inst in fmt::simplify_warp_trace(first_warp, true) {
            println!("{}", inst);
        }

        let ref_warp_traces = get_reference_warp_traces("single_for_loop")?;
        let ref_first_warp = &ref_warp_traces[&(trace_model::Dim::ZERO, 0)];
        for inst in fmt::simplify_warp_trace(ref_first_warp, true) {
            println!("{}", inst);
        }

        diff::assert_eq!(
            have: fmt::simplify_warp_trace(first_warp, true).collect::<Vec<_>>(),
            want: [
                ("LDG.E", Some(1), "11111111111111111111111111111111", 0),
                ("LDG.E", Some(1), "10101010101010101010101010101010", 1),
                ("LDG.E", Some(1), "10101010101010101010101010101010", 2),
                ("LDG.E", Some(100), "11111111111111111111111111111111", 3),
                ("EXIT", None, "11111111111111111111111111111111", 4),
            ].into_iter().enumerate().map(SimplifiedTraceInstruction::from).collect::<Vec<_>>()
        );
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
        let (_launch_config, trace) = tracer.trace_kernel(1, 32, &mut SingleIfKernel {}).await?;
        let warp_traces = trace.clone().to_warp_traces();
        let first_warp = &warp_traces[&(trace_model::Dim::ZERO, 0)];
        for inst in fmt::simplify_warp_trace(first_warp, true) {
            println!("{}", inst);
        }

        let ref_warp_traces = get_reference_warp_traces("single_if")?;
        let ref_first_warp = &ref_warp_traces[&(trace_model::Dim::ZERO, 0)];
        for inst in fmt::simplify_warp_trace(ref_first_warp, true) {
            println!("{}", inst);
        }

        diff::assert_eq!(
            have: fmt::simplify_warp_trace(first_warp, true).collect::<Vec<_>>(),
            want: [
                ("LDG.E", Some(1), "11111111111111111111111111111111", 0),
                ("LDG.E", Some(2), "11111111111111110000000000000000", 1),
                ("LDG.E", Some(3), "00000000000000001111111111111111", 2),
                ("LDG.E", Some(4), "11111111111111111111111111111111", 3),
                ("EXIT", None, "11111111111111111111111111111111", 4),
            ].into_iter().enumerate().map(SimplifiedTraceInstruction::from).collect::<Vec<_>>()
        );
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
    //     let first_warp = &warp_traces[&(trace_model::Dim::ZERO, 0)];
    //     for inst in testing::simplify_warp_trace(first_warp) {
    //         println!("{}", inst);
    //     }
    //
    //     let ref_warp_traces = get_reference_warp_traces("full_imbalance")?;
    //     let ref_first_warp = &ref_warp_traces[&(trace_model::Dim::ZERO, 0)];
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
        let (_launch_config, trace) = tracer
            .trace_kernel(1, 32, &mut TwoLevelNestedForLoops {})
            .await?;
        let warp_traces = trace.clone().to_warp_traces();
        let first_warp = &warp_traces[&(trace_model::Dim::ZERO, 0)];
        for inst in fmt::simplify_warp_trace(first_warp, true) {
            println!("{}", inst);
        }

        // let ref_warp_traces = get_reference_warp_traces("two_level_nested_if_balanced")?;
        // let ref_first_warp = &ref_warp_traces[&(trace_model::Dim::ZERO, 0)];
        // for inst in testing::simplify_warp_trace(ref_first_warp) {
        //     println!("{}", inst);
        // }

        diff::assert_eq!(
            have: fmt::simplify_warp_trace(first_warp, true).collect::<Vec<_>>(),
            want: [
                ("LDG.E", Some(100), "11111111111111111111111111111111", 0),
                ("STG.E", Some(0), "11111111111111111111111111111111", 1),
                ("STG.E", Some(1), "11111111111111111111111111111111", 2),
                ("STG.E", Some(2), "11111111111111111111111111111111", 3),
                ("STG.E", Some(3), "11111111111111111111111111111111", 4),
                ("LDG.E", Some(100), "11111111111111111111111111111111", 5),
                ("EXIT", None, "11111111111111111111111111111111", 6),
            ].into_iter().enumerate().map(SimplifiedTraceInstruction::from).collect::<Vec<_>>()
        );
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
        let (_launch_config, trace) = tracer
            .trace_kernel(1, 32, &mut TwoLevelNestedMultipleSerialIf {})
            .await?;
        let warp_traces = trace.clone().to_warp_traces();
        let first_warp = &warp_traces[&(trace_model::Dim::ZERO, 0)];
        for inst in fmt::simplify_warp_trace(first_warp, true) {
            println!("{}", inst);
        }

        // let ref_warp_traces = get_reference_warp_traces("two_level_nested_if_balanced")?;
        // let ref_first_warp = &ref_warp_traces[&(trace_model::Dim::ZERO, 0)];
        // for inst in testing::simplify_warp_trace(ref_first_warp) {
        //     println!("{}", inst);
        // }

        diff::assert_eq!(
            have: fmt::simplify_warp_trace(first_warp, true).collect::<Vec<_>>(),
            want: [
                ("LDG.E", Some(100), "11111111111111111111111111111111", 0),
                ("LDG.E", Some(10), "11111111000000000000000000000000", 1),
                ("LDG.E", Some(11), "00000000111111110000000000000000", 2),
                ("LDG.E", Some(20), "00000000000000001111111100000000", 3),
                ("LDG.E", Some(21), "00000000000000000000000011111111", 4),
                ("LDG.E", Some(100), "11111111111111111111111111111111", 5),
                ("EXIT", None, "11111111111111111111111111111111", 6),
            ].into_iter().enumerate().map(SimplifiedTraceInstruction::from).collect::<Vec<_>>()
        );
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
        let (_launch_config, trace) = tracer.trace_kernel(1, 32, &mut Balanced {}).await?;
        let warp_traces = trace.clone().to_warp_traces();
        let first_warp = &warp_traces[&(trace_model::Dim::ZERO, 0)];
        for inst in fmt::simplify_warp_trace(first_warp, true) {
            println!("{}", inst);
        }

        let ref_warp_traces = get_reference_warp_traces("two_level_nested_if_balanced")?;
        let ref_first_warp = &ref_warp_traces[&(trace_model::Dim::ZERO, 0)];
        for inst in fmt::simplify_warp_trace(ref_first_warp, true) {
            println!("{}", inst);
        }

        diff::assert_eq!(
            have: fmt::simplify_warp_trace(first_warp, true).collect::<Vec<_>>(),
            want: [
                ("LDG.E", Some(100), "11111111111111111111111111111111", 0),
                ("LDG.E", Some(10), "11111111111111110000000000000000", 1),
                ("LDG.E", Some(11), "11111111000000000000000000000000", 2),
                ("LDG.E", Some(20), "00000000000000001111111111111111", 3),
                ("LDG.E", Some(21), "00000000000000000000000011111111", 4),
                ("LDG.E", Some(100), "11111111111111111111111111111111", 5),
                ("EXIT", None, "11111111111111111111111111111111", 6),
            ].into_iter().enumerate().map(SimplifiedTraceInstruction::from).collect::<Vec<_>>()
        );
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
        let (_launch_config, trace) = tracer.trace_kernel(1, 32, &mut Imbalanced {}).await?;
        let warp_traces = trace.clone().to_warp_traces();
        let first_warp = &warp_traces[&(trace_model::Dim::ZERO, 0)];
        for inst in fmt::simplify_warp_trace(first_warp, true) {
            println!("{}", inst);
        }

        let ref_warp_traces = get_reference_warp_traces("two_level_nested_if_imbalanced")?;
        let ref_first_warp = &ref_warp_traces[&(trace_model::Dim::ZERO, 0)];
        for inst in fmt::simplify_warp_trace(ref_first_warp, true) {
            println!("{}", inst);
        }

        diff::assert_eq!(
            have: fmt::simplify_warp_trace(first_warp, true).collect::<Vec<_>>(),
            want: [
                ("LDG.E", Some(100), "11111111111111111111111111111111", 0),
                ("LDG.E", Some(1), "11111111111111110000000000000000", 1),
                ("LDG.E", Some(2), "11111111000000000000000000000000", 2),
                ("LDG.E", Some(100), "11111111111111111111111111111111", 3),
                ("EXIT", None, "11111111111111111111111111111111", 4),
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

        let mut kernel: VecAdd<f32> = VecAdd {
            dev_a: Mutex::new(dev_a),
            dev_b: Mutex::new(dev_b),
            dev_result: Mutex::new(dev_result),
            n,
        };
        let grid_size = (n as f64 / f64::from(block_size)).ceil() as u32;
        let (_launch_config, trace) = tracer
            .trace_kernel(grid_size, block_size, &mut kernel)
            .await?;
        let warp_traces = trace.clone().to_warp_traces();
        let first_warp = &warp_traces[&(trace_model::Dim::ZERO, 0)];

        reference_vectoradd(&a, &b, &mut ref_result);
        diff::assert_eq!(have: result, want: ref_result);
        for inst in fmt::simplify_warp_trace(first_warp, true) {
            println!("{}", inst);
        }
        diff::assert_eq!(
            have: fmt::simplify_warp_trace(first_warp, true).collect::<Vec<_>>(),
            want: [
                ("LDG.E", Some(0), "11111111111111111111000000000000", 0),
                ("LDG.E", Some(512), "11111111111111111111000000000000", 1),
                ("STG.E", Some(1024), "11111111111111111111000000000000", 2),
                ("EXIT", None, "11111111111111111111111111111111", 3),
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
