use crate::{config, opcodes, warp};
use trace_model as model;

pub trait Kernel: std::fmt::Debug + std::fmt::Display + Send + Sync + 'static {
    fn name(&self) -> &str;
    fn id(&self) -> u64;
    fn config(&self) -> &model::command::KernelLaunch;

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
    }

    fn no_more_blocks_to_run(&self) -> bool {
        self.next_block().is_none()
    }

    fn done(&self) -> bool {
        self.no_more_blocks_to_run() && !self.running()
    }
}

pub mod trace {
    use crate::sync::{Mutex, RwLock};
    use crate::{config, instruction, opcodes, warp};
    use color_eyre::Help;
    use model::command::KernelLaunch;
    use std::path::Path;
    use trace_model as model;

    pub const TRACE_BUF_SIZE: usize = 1_000_000;

    /// Kernel represents a kernel.
    ///
    /// This includes its launch configuration,
    /// as well as its state of execution.
    // #[derive(Debug)]
    pub struct KernelTrace<T>
    where
        T: Iterator<Item = model::MemAccessTraceEntry>,
    {
        pub opcodes: &'static opcodes::OpcodeMap,
        pub config: KernelLaunch,
        pub memory_only: bool,
        pub start_cycle: Mutex<Option<u64>>,
        pub completed_cycle: Mutex<Option<u64>>,
        pub start_time: Mutex<Option<std::time::Instant>>,
        pub completed_time: Mutex<Option<std::time::Instant>>,

        trace: RwLock<std::iter::Peekable<T>>,
        next_block: RwLock<Option<model::Dim>>,
        current_block: RwLock<Option<model::Dim>>,
        running_blocks: RwLock<usize>,
    }

    impl<T> PartialEq for KernelTrace<T>
    where
        T: Iterator<Item = model::MemAccessTraceEntry>,
    {
        fn eq(&self, other: &Self) -> bool {
            self.config.id == other.config.id
        }
    }

    impl<T> std::fmt::Debug for KernelTrace<T>
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

    impl<T> std::fmt::Display for KernelTrace<T>
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

    impl<T> super::Kernel for KernelTrace<T>
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
            *self.running_blocks.write() += 1;
        }

        fn decrement_running_blocks(&self) {
            *self.running_blocks.write() -= 1;
        }

        fn num_running_blocks(&self) -> usize {
            *self.running_blocks.read()
        }

        // #[inline]
        fn current_block(&self) -> Option<model::Point> {
            let current_block = self.current_block.try_read().clone()?;
            Some(model::Point::new(current_block, self.config.grid.clone()))
        }

        fn next_block(&self) -> Option<model::Point> {
            let next_block = self.next_block.try_read().clone()?;
            Some(model::Point::new(next_block, self.config.grid.clone()))
        }

        fn next_threadblock_traces(&self, warps: &mut [warp::Ref], config: &config::GPU) -> bool {
            let mut instructions = 0;
            let mut trace = self.trace.try_write();

            let current_block = trace.peek().map(|entry| entry.block_id.clone());
            *self.current_block.try_write() = current_block.clone();

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

    pub type TraceIter = crossbeam::channel::IntoIter<model::MemAccessTraceEntry>;

    impl KernelTrace<TraceIter> {
        pub fn new(config: model::command::KernelLaunch, traces_dir: impl AsRef<Path>) -> Self {
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

            let trace = trace_rx.into_iter().peekable();
            let opcodes = opcodes::get_opcode_map(&config).unwrap();
            Self {
                opcodes,
                config,
                memory_only: false,
                start_cycle: Mutex::new(None),
                start_time: Mutex::new(None),
                completed_cycle: Mutex::new(None),
                completed_time: Mutex::new(None),
                trace: RwLock::new(trace),
                current_block: RwLock::new(None),
                next_block: RwLock::new(Some(0.into())),
                running_blocks: RwLock::new(0),
            }
        }
    }
}
