use crate::sync::Arc;
use crate::{config, instruction::WarpInstruction, Kernel};
use color_eyre::{eyre, Help};
use std::path::{Path, PathBuf};
use trace_model::Command;

pub fn parse_commands(path: &Path) -> eyre::Result<Vec<Command>> {
    let reader = utils::fs::open_readable(path)?;
    let commands = serde_json::from_reader(reader)?;
    Ok(commands)
}

pub struct Trace {
    pub traces_dir: Option<PathBuf>,

    pub commands: Vec<Command>,
    command_idx: usize,
}

impl Trace {
    pub fn new(_config: Arc<config::GPU>) -> Self {
        Self {
            traces_dir: None,
            commands: Vec::new(),
            command_idx: 0,
        }
    }

    pub fn add_commands(
        &mut self,
        commands_path: impl AsRef<Path>,
        traces_dir: impl Into<PathBuf>,
    ) -> eyre::Result<()> {
        self.commands
            .extend(parse_commands(commands_path.as_ref())?);
        self.traces_dir = Some(traces_dir.into());
        Ok(())
    }

    pub fn commands_left(&self) -> bool {
        self.command_idx < self.commands.len()
    }

    pub fn next_command(&mut self) -> Option<&Command> {
        let command = self.commands.get(self.command_idx);
        self.command_idx += 1;
        command
    }
}

pub struct KernelTraceReader<T>
where
    T: Iterator<Item = trace_model::MemAccessTraceEntry>,
{
    pub opcodes: &'static crate::opcodes::OpcodeMap,
    pub config: trace_model::command::KernelLaunch,
    pub memory_only: bool,

    trace: std::iter::Peekable<T>,
    current_block: Option<trace_model::Dim>,
}

impl<T> std::fmt::Debug for KernelTraceReader<T>
where
    T: Iterator<Item = trace_model::MemAccessTraceEntry>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("KernelTraceReader")
            .field("config", &self.config)
            .field("current_block", &self.current_block)
            .field("memory_only", &self.memory_only)
            .finish_non_exhaustive()
    }
}

pub const TRACE_BUF_SIZE: usize = 1_000_000;

pub type TraceIter = crossbeam::channel::IntoIter<trace_model::MemAccessTraceEntry>;

impl KernelTraceReader<TraceIter> {
    pub fn new(
        config: trace_model::command::KernelLaunch,
        traces_dir: impl AsRef<Path>,
        memory_only: bool,
    ) -> Self {
        log::info!(
            "parsing kernel for launch {:#?} from {}",
            &config,
            &config.trace_file
        );
        let trace_path = traces_dir
            .as_ref()
            .join(&config.trace_file)
            .with_extension("msgpack");

        let buffer_memory =
            TRACE_BUF_SIZE * std::mem::size_of::<trace_model::MemAccessTraceEntry>();
        assert!(buffer_memory as u64 <= 1 * crate::config::GB);
        let (trace_tx, trace_rx) = crossbeam::channel::bounded(TRACE_BUF_SIZE);

        // spawn a decoder thread
        let reader = utils::fs::open_readable(trace_path).unwrap();
        std::thread::spawn(move || {
            use serde::Deserializer;
            let mut reader = rmp_serde::Deserializer::new(reader);
            let decoder: nvbit_io::Decoder<trace_model::MemAccessTraceEntry, _> =
                nvbit_io::Decoder::new(|access: trace_model::MemAccessTraceEntry| {
                    trace_tx.send(access).unwrap();
                });

            reader.deserialize_seq(decoder).suggestion("maybe the traces does not match the most recent binary trace format, try re-generating the traces.").unwrap();
        });

        let opcodes = crate::opcodes::get_opcode_map(&config).unwrap();

        let trace = trace_rx.into_iter().peekable();
        let current_block = None;
        Self {
            opcodes,
            config,
            memory_only,
            trace,
            current_block,
        }
    }
}

pub trait ReadWarpsForBlock: std::fmt::Debug + Send + Sync + 'static {
    fn current_block(&mut self) -> Option<trace_model::Point>;

    fn read_warps_for_block(
        &mut self,
        warps: &mut [crate::warp::Warp],
        kernel: &dyn Kernel,
        config: &config::GPU,
    ) -> (Option<trace_model::Dim>, Option<trace_model::Dim>);
}

impl<T> ReadWarpsForBlock for KernelTraceReader<T>
where
    T: Send + Sync + 'static,
    T: Iterator<Item = trace_model::MemAccessTraceEntry>,
{
    fn current_block(&mut self) -> Option<trace_model::Point> {
        let current_block = self.trace.peek().map(|entry| entry.block_id.clone())?;
        Some(trace_model::Point::new(
            current_block,
            self.config.grid.clone(),
        ))
    }

    fn read_warps_for_block(
        &mut self,
        warps: &mut [crate::warp::Warp],
        kernel: &dyn Kernel,
        config: &config::GPU,
    ) -> (Option<trace_model::Dim>, Option<trace_model::Dim>) {
        let current_block_lock = &mut self.current_block;
        let trace_lock = &mut self.trace;

        *current_block_lock = trace_lock.peek().map(|entry| entry.block_id.clone());

        let Some(ref current_block) = *current_block_lock else {
            // no more threadblocks
            log::info!("blocks done: no more threadblock traces");
            return (None, None);
        };

        log::info!(
            "{} ({}) issue block {}/{}",
            self.config.name(),
            self.config.id,
            current_block,
            self.config.grid,
        );

        let mut instructions = 0;
        loop {
            let Some(entry) = &trace_lock.peek() else {
                break;
            };
            if entry.block_id != *current_block {
                break;
            }

            let warp_id = entry.warp_id_in_block as usize;
            let instr = WarpInstruction::from_trace(entry, &self.config, &self.opcodes, config);

            if !self.memory_only || instr.is_memory_instruction() {
                let warp = warps.get_mut(warp_id).unwrap();
                log::trace!(
                    "block {}: adding {} to warp {}",
                    current_block,
                    instr,
                    warp.warp_id
                );
                warp.push_trace_instruction(instr);
            }

            instructions += 1;
            trace_lock.next();
        }

        let next_block = trace_lock.peek().map(|entry| entry.block_id.clone());
        if next_block.is_none() {
            kernel.set_done();
        }

        log::debug!(
            "added {instructions} instructions ({} per warp) for block {current_block}",
            instructions / warps.len()
        );
        debug_assert!(instructions > 0);

        debug_assert!(
            warps.iter().all(|w| { !w.trace_instructions.is_empty() }),
            "all warps have at least one instruction (need at least an EXIT)"
        );
        (current_block_lock.clone(), next_block)
    }
}
