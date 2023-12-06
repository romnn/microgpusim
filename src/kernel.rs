use crate::sync::{Mutex, RwLock};
use crate::{config, instruction, opcodes, warp};
use color_eyre::{
    eyre::{self},
    Help,
};
use model::command::KernelLaunch;
use std::collections::HashSet;
use std::path::Path;

use trace_model as model;

pub const TRACE_BUF_SIZE: usize = 1_000_000;

// pub fn read_trace(path: impl AsRef<Path>) -> eyre::Result<model::MemAccessTrace> {
//     use serde::Deserializer;
//
//     let reader = utils::fs::open_readable(path.as_ref())?;
//     let mut reader = rmp_serde::Deserializer::new(reader);
//     let mut trace = vec![];
//     let decoder = nvbit_io::Decoder::new(|access: model::MemAccessTraceEntry| {
//         trace.push(access);
//     });
//     reader.deserialize_seq(decoder).suggestion("maybe the traces does not match the most recent binary trace format, try re-generating the traces.")?;
//     Ok(model::MemAccessTrace(trace))
// }

/// Kernel represents a kernel.
///
/// This includes its launch configuration,
/// as well as its state of execution.
// #[derive(Debug)]
pub struct Kernel<T>
where
    T: Iterator<Item = model::MemAccessTraceEntry>,
{
    pub opcodes: &'static opcodes::OpcodeMap,
    pub config: KernelLaunch,
    pub memory_only: bool,
    // pub num_cores_running: usize,
    pub start_cycle: Mutex<Option<u64>>,
    pub completed_cycle: Mutex<Option<u64>>,
    pub start_time: Mutex<Option<std::time::Instant>>,
    pub completed_time: Mutex<Option<std::time::Instant>>,

    // trace_rx: crossbeam::channel::Receiver<model::MemAccessTraceEntry>,
    // trace: T,
    trace: RwLock<std::iter::Peekable<T>>,
    next_block: RwLock<Option<model::Dim>>,
    current_block: RwLock<Option<model::Dim>>,
    running_blocks: RwLock<usize>,
    // let mut trace_iter = self.trace.try_write();
    // let mut trace = trace_iter.by_ref().peekable();

    // trace: std::iter::Peekable<T>,
    // trace_pos: RwLock<usize>,
    // trace: model::MemAccessTrace,
}

// impl PartialEq for Kernel {
impl<T> PartialEq for Kernel<T>
where
    T: Iterator<Item = model::MemAccessTraceEntry>,
    // where
    //     T: Sync + Send + 'static,
{
    fn eq(&self, other: &Self) -> bool {
        self.config.id == other.config.id
    }
}

impl<T> std::fmt::Debug for Kernel<T>
where
    T: Iterator<Item = model::MemAccessTraceEntry>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // workaround: this is required because we dont want to impose debug on trace stream yet
        f.debug_struct("Kernel")
            .field("name", &self.config.name())
            .field("id", &self.config.id)
            .finish()
    }
}

// impl std::fmt::Display for Kernel {
impl<T> std::fmt::Display for Kernel<T>
where
    T: Iterator<Item = model::MemAccessTraceEntry>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kernel")
            .field("name", &self.config.name())
            .field("id", &self.config.id)
            .finish()
    }
}

pub trait KernelTrait: std::fmt::Debug + std::fmt::Display + Send + Sync + 'static {
    fn name(&self) -> &str;
    fn id(&self) -> u64;
    fn config(&self) -> &KernelLaunch;

    fn set_completed(&self, cycle: u64);
    fn set_started(&self, cycle: u64);
    fn elapsed_cycles(&self) -> Option<u64>;
    fn elapsed_time(&self) -> Option<std::time::Duration>;
    fn launched(&self) -> bool;
    fn completed(&self) -> bool;

    fn increment_running_blocks(&self);
    fn decrement_running_blocks(&self);

    fn opcode(&self, opcode: &str) -> Option<&opcodes::Opcode>;

    fn next_block(&self) -> Option<model::Point>;
    fn current_block(&self) -> Option<model::Point>;

    fn next_threadblock_traces(&self, warps: &mut [warp::Ref], config: &config::GPU) -> bool;

    fn num_running_blocks(&self) -> usize;

    fn running(&self) -> bool {
        self.num_running_blocks() > 0
        // *self.running_blocks.try_read() == 0
        // self.current_block().is_some()
    }

    // #[inline]
    fn no_more_blocks_to_run(&self) -> bool {
        // dbg!(&self.current_block());
        // assert!(self.current_block().is_some());
        // !self.current_block().is_some()
        self.next_block().is_none()
    }

    fn done(&self) -> bool {
        self.no_more_blocks_to_run() && !self.running()
    }
}

impl<T> KernelTrait for Kernel<T>
where
    T: Sync + Send + 'static,
    T: Iterator<Item = model::MemAccessTraceEntry>,
{
    fn name(&self) -> &str {
        self.config.name()
    }

    fn id(&self) -> u64 {
        self.config.id
    }

    fn config(&self) -> &KernelLaunch {
        &self.config
    }

    fn opcode(&self, opcode: &str) -> Option<&opcodes::Opcode> {
        self.opcodes.get(opcode)
    }

    fn set_started(&self, cycle: u64) {
        *self.start_time.lock() = Some(std::time::Instant::now());
        *self.start_cycle.lock() = Some(cycle);
    }

    fn set_completed(&self, cycle: u64) {
        *self.completed_time.lock() = Some(std::time::Instant::now());
        *self.completed_cycle.lock() = Some(cycle);
    }

    fn elapsed_cycles(&self) -> Option<u64> {
        let start_cycle = self.start_cycle.lock();
        let completed_cycle = self.completed_cycle.lock();
        match (*start_cycle, *completed_cycle) {
            (Some(start_cycle), Some(completed_cycle)) => Some(completed_cycle - start_cycle),
            _ => None,
        }
    }

    fn elapsed_time(&self) -> Option<std::time::Duration> {
        let start_time = self.start_time.lock();
        let completed_time = self.completed_time.lock();
        match (*start_time, *completed_time) {
            (Some(start_time), Some(completed_time)) => Some(completed_time - start_time),
            _ => None,
        }
    }

    // #[inline]
    fn launched(&self) -> bool {
        self.start_cycle.lock().is_some()
    }

    // #[inline]
    fn completed(&self) -> bool {
        self.completed_cycle.lock().is_some()
    }

    fn increment_running_blocks(&self) {
        *self.running_blocks.try_write() += 1;
    }

    fn decrement_running_blocks(&self) {
        *self.running_blocks.try_write() -= 1;
    }

    fn num_running_blocks(&self) -> usize {
        *self.running_blocks.try_read()
    }

    // #[inline]
    fn current_block(&self) -> Option<model::Point> {
        let current_block = self.current_block.try_read().clone()?;
        Some(model::Point::new(current_block, self.config.grid.clone()))
    }

    fn next_block(&self) -> Option<model::Point> {
        // self.current_block.try_read().clone()
        // *self.current_block.try_write() =
        let next_block = self.next_block.try_read().clone()?;
        Some(model::Point::new(next_block, self.config.grid.clone()))

        // let mut trace_iter = self.trace.try_read();
        // let mut trace = trace_iter.by_ref().peekable();
        // match trace.peek().map(|entry| entry.block_id) {
        //     Some(current_block) => Some(model::Point::new(
        //         trace.block_id.clone(),
        //         self.config.grid.clone(),
        //     )),
        //     None => None,
        // }
        // let traces_pos = self.trace_pos.try_read();
        // let trace = self.trace.get(*traces_pos)?;
        // Some(model::Point::new(
        //     trace.block_id.clone(),
        //     self.config.grid.clone(),
        // ))
    }

    // fn running(&self) -> bool {
    //     self.num_cores_running > 0
    // }

    fn next_threadblock_traces(&self, warps: &mut [warp::Ref], config: &config::GPU) -> bool {
        let mut instructions = 0;
        let mut trace = self.trace.try_write();
        // let mut trace_iter = self.trace.try_write();
        // let mut trace = trace_iter.by_ref().peekable();

        // dbg!(&trace.peek());
        let current_block = trace.peek().map(|entry| entry.block_id.clone());
        *self.current_block.try_write() = current_block.clone();
        // dbg!(&current_block);

        let Some(current_block) = current_block else {
            // no more threadblocks
            log::info!("blocks done: no more threadblock traces");
            return false;
        };

        log::info!(
            "{} ({}) issue block {}/{}",
            self.name(),
            self.id(),
            current_block,
            self.config.grid,
        );

        loop {
            let Some(entry) = &trace.peek() else {
                break;
            };
            if entry.block_id != current_block {
                // println!("stopping with peek={:#?}", entry);
                break;
            }

            let warp_id = entry.warp_id_in_block as usize;
            let instr = instruction::WarpInstruction::from_trace(self, entry, config);

            if !self.memory_only || instr.is_memory_instruction() {
                let warp = warps.get_mut(warp_id).unwrap();
                let mut warp = warp.try_lock();
                log::trace!(
                    "block {}: adding {} to warp {}",
                    current_block,
                    instr,
                    warp.warp_id
                );
                warp.push_trace_instruction(instr);
            }

            instructions += 1;
            trace.next();
        }

        let next_block = trace.peek().map(|entry| entry.block_id.clone());
        *self.next_block.try_write() = next_block.clone();
        // dbg!(&next_block);

        // let mut trace_pos = self.trace_pos.write();
        //
        // let mut instructions = 0;
        // let trace_size = self.trace.len();
        //
        // if *trace_pos + 1 >= trace_size || trace_size == 0 {
        //     // no more threadblocks
        //     log::info!("blocks done: no more threadblock traces");
        //     return false;
        // }
        // let next_block = &self.trace[*trace_pos + 1].block_id;
        // log::info!("{} ({}) issue block {next_block}", self.name(), self.id());
        //
        // while *trace_pos < trace_size {
        //     let entry = &self.trace[*trace_pos];
        //     if entry.block_id != *next_block {
        //         // get instructions until new block
        //         break;
        //     }
        //
        //     let warp_id = entry.warp_id_in_block as usize;
        //     let instr = instruction::WarpInstruction::from_trace(self, entry, config);
        //
        //     if !self.memory_only || instr.is_memory_instruction() {
        //         let warp = warps.get_mut(warp_id).unwrap();
        //         let mut warp = warp.try_lock();
        //         warp.push_trace_instruction(instr);
        //     }
        //
        //     instructions += 1;
        //     *trace_pos += 1;
        // }

        log::debug!(
            "added {instructions} instructions ({} per warp) for block {current_block}",
            instructions / warps.len()
        );
        debug_assert!(instructions > 0);

        debug_assert!(
            warps
                .iter()
                .all(|w| !w.try_lock().trace_instructions.is_empty()),
            "all warps have at least one instruction (need at least an EXIT)"
        );
        true
    }
}

// impl Kernel {
// impl<T> Kernel<T> {
//     pub fn new(
//         config: model::command::KernelLaunch,
//         // trace: model::MemAccessTrace,
//         // trace_rx: crossbeam::channel::Receiver<model::MemAccessTraceEntry>,
//         trace_rx: crossbeam::channel::Receiver<model::MemAccessTraceEntry>,
//         // trace_rx: crossbeam::channel::Receiver<model::MemAccessTraceEntry>,
//         // memory_only: bool,
//     ) -> Self {
//         // sanity check
//         // trace.check_valid().expect("valid trace");
//
//         // check if grid size is equal to the number of unique blocks in the trace
//         // let all_blocks: HashSet<_> = trace.iter().map(|t| &t.block_id).collect();
//         // log::info!(
//         //     "parsed kernel trace for {:?}: {}/{} blocks",
//         //     config.unmangled_name,
//         //     all_blocks.len(),
//         //     config.grid.size(),
//         // );
//         // assert_eq!(config.grid.size(), all_blocks.len() as u64);
//
//         let opcodes = opcodes::get_opcode_map(&config).unwrap();
//         Self {
//             opcodes,
//             config,
//             num_cores_running: 0,
//             memory_only: false,
//             start_cycle: Mutex::new(None),
//             start_time: Mutex::new(None),
//             completed_cycle: Mutex::new(None),
//             completed_time: Mutex::new(None),
//             trace_rx,
//             // trace,
//             // trace_pos: RwLock::new(0),
//         }
//     }
// }

pub type TraceIter = crossbeam::channel::IntoIter<model::MemAccessTraceEntry>;

impl Kernel<TraceIter>
// impl Kernel<std::iter::Peekable<TraceIter>>
// impl<T> Kernel<T>
// where
//     T: Send + Sync + 'static,
{
    // pub fn launched(&self) -> bool {
    //     *self.launched.try_lock()
    // }
    // fn id(&self) -> u64 {
    //     self.config.id
    // }

    // pub fn set_completed(&self, cycle: u64) -> bool {
    //     *self.completed_time.lock() = Some(completion_time);
    //     *self.completed_cycle.lock() = Some(cycle);
    // }

    pub fn from_trace(config: model::command::KernelLaunch, traces_dir: impl AsRef<Path>) -> Self {
        log::info!(
            "parsing kernel for launch {:#?} from {}",
            &config,
            &config.trace_file
        );
        let trace_path = traces_dir
            .as_ref()
            .join(&config.trace_file)
            .with_extension("msgpack");

        let (trace_tx, trace_rx) = crossbeam::channel::bounded(TRACE_BUF_SIZE);

        // spawn a decoder thread
        let reader = utils::fs::open_readable(trace_path).unwrap();
        std::thread::spawn(move || {
            use serde::Deserializer;
            let mut reader = rmp_serde::Deserializer::new(reader);
            let decoder: nvbit_io::Decoder<model::MemAccessTraceEntry, _> =
                nvbit_io::Decoder::new(|access: model::MemAccessTraceEntry| {
                    trace_tx.send(access).unwrap();
                });

            reader.deserialize_seq(decoder).suggestion("maybe the traces does not match the most recent binary trace format, try re-generating the traces.").unwrap();
        });

        // let mut trace = vec![];

        // reader.deserialize_seq(decoder).suggestion("maybe the traces does not match the most recent binary trace format, try re-generating the traces.")?;
        //     Ok(model::MemAccessTrace(trace))

        // let mut trace = crate::timeit!("read trace", read_trace(trace_path).unwrap());
        // TODO: temp hotfix
        // for inst in trace.0.iter_mut() {
        //     // does this break simulation?
        //     inst.instr_offset = 0;
        //     inst.instr_idx = 0;
        //
        //     inst.line_num = 0;
        //     inst.warp_id_in_sm = 0;
        //     inst.instr_data_width = 0;
        //     inst.sm_id = 0;
        //     inst.cuda_ctx = 0;
        //
        //     inst.src_regs.fill(0);
        //     inst.num_src_regs = 0;
        //
        //     inst.dest_regs.fill(0);
        //     inst.num_dest_regs = 0;
        // }

        // Self::new(config, trace_rx.into_iter().peekable())
        let trace = trace_rx.into_iter().peekable();
        // let trace = trace_rx.into_iter();
        let opcodes = opcodes::get_opcode_map(&config).unwrap();
        Self {
            opcodes,
            config,
            // num_cores_running: 0,
            memory_only: false,
            start_cycle: Mutex::new(None),
            start_time: Mutex::new(None),
            completed_cycle: Mutex::new(None),
            completed_time: Mutex::new(None),
            trace: RwLock::new(trace),
            // next_block: RwLock::new(None),
            current_block: RwLock::new(None),
            next_block: RwLock::new(Some(0.into())),
            running_blocks: RwLock::new(0),
            // current_block: RwLock::new(Some(0.into())),
            // current_block: RwLock::new(Some(model::Point::new(0.into(), 0.into()))),
            // trace,
            // trace_pos: RwLock::new(0),
        }
    }

    // pub fn shared_memory_bytes_human_readable(&self) -> String {
    //     human_bytes::human_bytes(f64::from(self.config.shared_mem_bytes))
    // }

    // pub fn set_launched(&self) {
    //     *self.launched.try_lock() = true;
    // }

    // #[inline]
    // pub fn id(&self) -> u64 {
    //     self.config.id
    // }

    // #[inline]
    // pub fn name(&self) -> &str {
    //     &self.config.unmangled_name
    // }

    // #[inline]

    // #[inline]
    // pub fn num_blocks(&self) -> usize {
    //     let grid = &self.config.grid;
    //     grid.x as usize * grid.y as usize * grid.z as usize
    // }

    // #[inline]
    // pub fn threads_per_block(&self) -> usize {
    //     let block = &self.config.block;
    //     block.x as usize * block.y as usize * block.z as usize
    // }
}

// impl<T> Kernel<T>
// where
//     T: Send + Sync + 'static,
// {
//     // #[inline]
//     pub fn done(&self) -> bool {
//         self.no_more_blocks_to_run() && !self.running()
//     }
// }
