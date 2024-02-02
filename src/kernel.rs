use crate::sync::Mutex;
use crate::{config, opcodes, warp};

pub trait Kernel: std::fmt::Debug + std::fmt::Display + Send + Sync + 'static {
    // type Test: crate::trace::ReadWarpsForBlock;

    fn name(&self) -> &str;
    fn id(&self) -> u64;
    fn config(&self) -> &trace_model::command::KernelLaunch;

    fn set_completed(&self, cycle: u64);
    fn set_started(&self, cycle: u64);
    fn elapsed_cycles(&self) -> Option<u64>;
    fn elapsed_time(&self) -> Option<std::time::Duration>;
    fn launched(&self) -> bool;
    // fn completed(&self) -> bool;

    fn increment_running_blocks(&self);
    fn decrement_running_blocks(&self);

    // todo: aggregate with set_completed?
    fn set_done(&self);

    // fn opcode(&self, opcode: &str) -> Option<&opcodes::Opcode>;

    // fn next_block(&self) -> Option<trace_model::Point>;
    // fn current_block(&self) -> Option<trace_model::Point>;

    //
    // fn next_threadblock_traces(&self, warps: &mut [warp::Warp], config: &config::GPU) -> bool;

    // fn next_block_reader(&self) -> &Mutex<Self::Test>;
    // fn next_block_reader(&self) -> &Mutex<dyn crate::trace::ReadWarpsForBlock>;
    fn next_block_reader(&self) -> &Box<Mutex<dyn crate::trace::ReadWarpsForBlock>>;
    // fn next_block_reader(&self) -> &Mutex<&dyn crate::trace::ReadWarpsForBlock>;
    // fn next_block_reader(&self) -> &mut dyn crate::trace::ReadWarpsForBlock;

    fn num_running_blocks(&self) -> usize;

    fn running(&self) -> bool {
        self.num_running_blocks() > 0
    }

    fn no_more_blocks_to_run(&self) -> bool;

    // fn no_more_blocks_to_run(&self) -> bool {
    //     self.current_block().is_none()
    //     // self.next_block().is_none()
    // }

    fn done(&self) -> bool {
        self.no_more_blocks_to_run() && !self.running()
    }
}

pub mod trace {
    use crate::sync::{Mutex, RwLock};
    use crate::trace::KernelTraceReader;
    use crate::{config, instruction, opcodes, warp};
    use color_eyre::Help;
    use std::path::Path;
    use trace_model::command::KernelLaunch;

    pub const TRACE_BUF_SIZE: usize = 1_000_000;

    /// Kernel represents a kernel.
    ///
    /// This includes its launch configuration,
    /// as well as its state of execution.
    // #[derive(Debug)]
    pub struct KernelTrace<T>
    where
        T: crate::trace::ReadWarpsForBlock,
        // T: Iterator<Item = trace_model::MemAccessTraceEntry>,
    {
        config: KernelLaunch,
        start_cycle: Mutex<Option<u64>>,
        completed_cycle: Mutex<Option<u64>>,
        start_time: Mutex<Option<std::time::Instant>>,
        completed_time: Mutex<Option<std::time::Instant>>,
        running_blocks: RwLock<usize>,

        done: RwLock<bool>,

        // reader: Mutex<T>,
        // reader: Box<Mutex<T>>,
        // reader: Box<Mutex<dyn crate::trace::ReadWarpsForBlock>>,
        reader: Box<Mutex<dyn crate::trace::ReadWarpsForBlock>>,
        // reader: Mutex<dyn crate::trace::ReadWarpsForBlock>,
        phantom: std::marker::PhantomData<T>,

        // reader: Mutex<crate::trace::KernelTraceReader<T>>,
        // reader: crate::trace::KernelTraceReader<T>,
        current_block: RwLock<Option<trace_model::Dim>>,
        next_block: RwLock<Option<trace_model::Dim>>,
        // pub opcodes: &'static opcodes::OpcodeMap,
        // pub config: KernelLaunch,
        // pub memory_only: bool,

        //
        // trace: RwLock<std::iter::Peekable<T>>,
        // next_block: RwLock<Option<trace_model::Dim>>,
        // current_block: RwLock<Option<trace_model::Dim>>,
    }

    impl<T> PartialEq for KernelTrace<T>
    where
        T: crate::trace::ReadWarpsForBlock,
        // T: Iterator<Item = trace_model::MemAccessTraceEntry>,
    {
        fn eq(&self, other: &Self) -> bool {
            self.config.id == other.config.id
        }
    }

    impl<T> std::fmt::Debug for KernelTrace<T>
    where
        T: crate::trace::ReadWarpsForBlock,
        // T: Iterator<Item = trace_model::MemAccessTraceEntry>,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            // workaround: this is required because we dont want to impose debug on trace stream yet
            f.debug_struct("Kernel")
                .field("name", &self.config.name())
                .field("id", &self.config.id)
                .finish()
        }
    }

    impl<T> std::fmt::Display for KernelTrace<T>
    where
        T: crate::trace::ReadWarpsForBlock,
        // T: Iterator<Item = trace_model::MemAccessTraceEntry>,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Kernel")
                .field("name", &self.config.name())
                .field("id", &self.config.id)
                .finish()
        }
    }

    impl super::Kernel for KernelTrace<crate::trace::KernelTraceReader<crate::trace::TraceIter>>
    // impl<T> super::Kernel for KernelTrace<T>
    // where
    //     T: Sync + Send + 'static,
    // T: crate::trace::ReadWarpsForBlock,
    // T: Iterator<Item = trace_model::MemAccessTraceEntry>,
    {
        // type Test = crate::trace::KernelTraceReader<crate::trace::TraceIter>;

        fn name(&self) -> &str {
            self.config.name()
        }

        fn id(&self) -> u64 {
            self.config.id
        }

        fn config(&self) -> &KernelLaunch {
            &self.config
        }

        // fn opcode(&self, opcode: &str) -> Option<&opcodes::Opcode> {
        //     self.opcodes.get(opcode)
        // }

        // cold function
        fn set_done(&self) {
            *self.done.write() = true;
        }

        // cold function
        fn set_started(&self, cycle: u64) {
            *self.start_time.lock() = Some(std::time::Instant::now());
            *self.start_cycle.lock() = Some(cycle);
        }

        // cold function
        fn set_completed(&self, cycle: u64) {
            *self.completed_time.lock() = Some(std::time::Instant::now());
            *self.completed_cycle.lock() = Some(cycle);
        }

        // cold function
        fn elapsed_cycles(&self) -> Option<u64> {
            let start_cycle = self.start_cycle.lock();
            let completed_cycle = self.completed_cycle.lock();
            match (*start_cycle, *completed_cycle) {
                (Some(start_cycle), Some(completed_cycle)) => Some(completed_cycle - start_cycle),
                _ => None,
            }
        }

        // cold function
        fn elapsed_time(&self) -> Option<std::time::Duration> {
            let start_time = self.start_time.lock();
            let completed_time = self.completed_time.lock();
            match (*start_time, *completed_time) {
                (Some(start_time), Some(completed_time)) => Some(completed_time - start_time),
                _ => None,
            }
        }

        // cold function, only during launch and get finished
        fn launched(&self) -> bool {
            self.start_cycle.lock().is_some()
        }

        // #[inline]
        // fn completed(&self) -> bool {
        //     self.completed_cycle.lock().is_some()
        // }

        // hot
        fn increment_running_blocks(&self) {
            *self.running_blocks.write() += 1;
        }

        // hot
        fn decrement_running_blocks(&self) {
            *self.running_blocks.write() -= 1;
        }

        // hot
        fn num_running_blocks(&self) -> usize {
            *self.running_blocks.read()
        }

        // hot
        fn no_more_blocks_to_run(&self) -> bool {
            *self.done.read()
        }

        // NOTE: this must not be absolutely consistent
        // #[inline]
        // fn current_block(&self) -> Option<trace_model::Point> {
        //     // cannot be try_read because we have cores competing for blocks
        //     let current_block = self.current_block.read();
        //     // self.reader.read().current_block()
        //     // let current_block = self.current_block.try_read().clone()?;
        //     Some(trace_model::Point::new(
        //         current_block.as_ref()?.clone(),
        //         self.config.grid.clone(),
        //     ))
        // }

        // fn next_block(&self) -> Option<trace_model::Point> {
        //     // cannot be try_read because we have cores competing for blocks
        //     // self.next_block.read()
        //     // self.reader.next_block()
        //     // self.reader.read().next_block()
        //     let next_block = self.next_block.read();
        //     Some(trace_model::Point::new(
        //         next_block.as_ref()?.clone(),
        //         self.config.grid.clone(),
        //     ))
        // }

        // fn next_block_reader(&self) -> &mut dyn crate::trace::ReadWarpsForBlock {
        // fn next_block_reader(&self) -> &Mutex<&dyn crate::trace::ReadWarpsForBlock> {
        // fn next_block_reader(&self) -> &Mutex<T> {
        // fn next_block_reader(&self) -> &Mutex<&dyn crate::trace::ReadWarpsForBlock> {
        // fn next_block_reader(&self) -> &Mutex<Self::Test> {

        // cold
        fn next_block_reader(&self) -> &Box<Mutex<dyn crate::trace::ReadWarpsForBlock>> {
            &self.reader
            // todo!()
            // Box::new(&self.reader as Mutex<dyn crate::trace::ReadWarpsForBlock>)
            // let test: Mutex<&dyn crate::trace::ReadWarpsForBlock> = &self.reader;
            // &self.reader as &Mutex<&dyn crate::trace::ReadWarpsForBlock>
            // &mut *self.reader.lock()
        }

        // fn request_block(&self) -> &mut KernelTraceReader<T> {
        //     self.reader.lock().as_mut()
        // }

        // fn next_threadblock_traces(&self, warps: &mut [warp::Warp], config: &config::GPU) -> bool {
        //     // cannot be try_write because we have cores competing for blocks
        //
        //     let (current_block, next_block) = self.reader.next_threadblock_traces(warps, config);
        //     let done = current_block.is_none();
        //     *self.current_block.write() = current_block;
        //     *self.next_block.write() = next_block;
        //     done
        //
        //     // self.reader.write().next_threadblock_traces(warps, config)
        //     // let mut instructions = 0;
        //     // // let mut trace = self.trace.try_write();
        //     // // try write wont work here because
        //     // let mut trace = self.trace.write();
        //     //
        //     // let current_block = trace.peek().map(|entry| entry.block_id.clone());
        //     // *self.current_block.try_write() = current_block.clone();
        //     //
        //     // let Some(current_block) = current_block else {
        //     //     // no more threadblocks
        //     //     log::info!("blocks done: no more threadblock traces");
        //     //     return false;
        //     // };
        //     //
        //     // log::info!(
        //     //     "{} ({}) issue block {}/{}",
        //     //     self.name(),
        //     //     self.id(),
        //     //     current_block,
        //     //     self.config.grid,
        //     // );
        //     //
        //     // loop {
        //     //     let Some(entry) = &trace.peek() else {
        //     //     break;
        //     // };
        //     //     if entry.block_id != current_block {
        //     //         // println!("stopping with peek={:#?}", entry);
        //     //         break;
        //     //     }
        //     //
        //     //     let warp_id = entry.warp_id_in_block as usize;
        //     //     let instr = instruction::WarpInstruction::from_trace(
        //     //         entry,
        //     //         &self.config,
        //     //         &self.opcodes,
        //     //         config,
        //     //     );
        //     //     // let instr = instruction::WarpInstruction::from_trace(self, entry, config);
        //     //
        //     //     if !self.memory_only || instr.is_memory_instruction() {
        //     //         let warp = warps.get_mut(warp_id).unwrap();
        //     //         log::trace!(
        //     //             "block {}: adding {} to warp {}",
        //     //             current_block,
        //     //             instr,
        //     //             warp.warp_id
        //     //         );
        //     //         warp.push_trace_instruction(instr);
        //     //     }
        //     //
        //     //     instructions += 1;
        //     //     trace.next();
        //     // }
        //     //
        //     // let next_block = trace.peek().map(|entry| entry.block_id.clone());
        //     // *self.next_block.try_write() = next_block.clone();
        //     //
        //     // log::debug!(
        //     //     "added {instructions} instructions ({} per warp) for block {current_block}",
        //     //     instructions / warps.len()
        //     // );
        //     // debug_assert!(instructions > 0);
        //     //
        //     // debug_assert!(
        //     //     warps.iter().all(|w| { !w.trace_instructions.is_empty() }),
        //     //     "all warps have at least one instruction (need at least an EXIT)"
        //     // );
        //     // true
        // }
    }

    // impl KernelTrace<crate::trace::TraceIter>
    impl KernelTrace<crate::trace::KernelTraceReader<crate::trace::TraceIter>>
    // where
    //     T: Iterator<Item = trace_model::MemAccessTraceEntry>,
    {
        pub fn new(
            config: trace_model::command::KernelLaunch,
            traces_dir: impl AsRef<Path>,
            memory_only: bool,
        ) -> Self {
            let reader =
                crate::trace::KernelTraceReader::new(config.clone(), traces_dir, memory_only);
            let reader = Mutex::new(reader);
            let reader = Box::new(reader);
            Self {
                config,
                start_cycle: Mutex::new(None),
                start_time: Mutex::new(None),
                completed_cycle: Mutex::new(None),
                completed_time: Mutex::new(None),
                reader,
                phantom: std::marker::PhantomData,
                // reader: RwLock::new(reader),
                done: RwLock::new(false),
                current_block: RwLock::new(None),
                next_block: RwLock::new(Some(0.into())),
                running_blocks: RwLock::new(0),
            }
        }
    }
}

// pub type TraceIter = crossbeam::channel::IntoIter<trace_model::MemAccessTraceEntry>;
//
// impl KernelTrace<TraceIter> {
//     pub fn new(
//         config: trace_model::command::KernelLaunch,
//         traces_dir: impl AsRef<Path>,
//     ) -> Self {
//         log::info!(
//             "parsing kernel for launch {:#?} from {}",
//             &config,
//             &config.trace_file
//         );
//         let trace_path = traces_dir
//             .as_ref()
//             .join(&config.trace_file)
//             .with_extension("msgpack");
//
//         let buffer_memory =
//             TRACE_BUF_SIZE * std::mem::size_of::<trace_model::MemAccessTraceEntry>();
//         assert!(buffer_memory as u64 <= 1 * crate::config::GB);
//         let (trace_tx, trace_rx) = crossbeam::channel::bounded(TRACE_BUF_SIZE);
//
//         // spawn a decoder thread
//         let reader = utils::fs::open_readable(trace_path).unwrap();
//         std::thread::spawn(move || {
//             use serde::Deserializer;
//             let mut reader = rmp_serde::Deserializer::new(reader);
//             let decoder: nvbit_io::Decoder<trace_model::MemAccessTraceEntry, _> =
//                 nvbit_io::Decoder::new(|access: trace_model::MemAccessTraceEntry| {
//                     trace_tx.send(access).unwrap();
//                 });
//
//             reader.deserialize_seq(decoder).suggestion("maybe the traces does not match the most recent binary trace format, try re-generating the traces.").unwrap();
//         });
//
//         let trace = trace_rx.into_iter().peekable();
//         let opcodes = opcodes::get_opcode_map(&config).unwrap();
//         Self {
//             opcodes,
//             config,
//             memory_only: false,
//             start_cycle: Mutex::new(None),
//             start_time: Mutex::new(None),
//             completed_cycle: Mutex::new(None),
//             completed_time: Mutex::new(None),
//             trace: RwLock::new(trace),
//             current_block: RwLock::new(None),
//             next_block: RwLock::new(Some(0.into())),
//             running_blocks: RwLock::new(0),
//         }
//     }
// }
