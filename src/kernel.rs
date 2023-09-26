use crate::sync::{Mutex, RwLock};
use crate::{config, instruction, opcodes, warp};
use color_eyre::{
    eyre::{self},
    Help,
};
use std::collections::HashSet;
use std::path::Path;

use trace_model as model;

pub fn read_trace(path: impl AsRef<Path>) -> eyre::Result<model::MemAccessTrace> {
    use serde::Deserializer;

    let reader = utils::fs::open_readable(path.as_ref())?;
    let mut reader = rmp_serde::Deserializer::new(reader);
    let mut trace = vec![];
    let decoder = nvbit_io::Decoder::new(|access: model::MemAccessTraceEntry| {
        trace.push(access);
    });
    reader.deserialize_seq(decoder).suggestion("maybe the traces does not match the most recent binary trace format, try re-generating the traces.")?;
    Ok(model::MemAccessTrace(trace))
}

/// Kernel represents a kernel.
///
/// This includes its launch configuration,
/// as well as its state of execution.
#[derive(Debug)]
pub struct Kernel {
    pub opcodes: &'static opcodes::OpcodeMap,
    pub config: model::command::KernelLaunch,
    pub memory_only: bool,
    pub num_cores_running: usize,

    pub start_cycle: Mutex<Option<u64>>,
    pub completed_cycle: Mutex<Option<u64>>,
    pub start_time: Mutex<Option<std::time::Instant>>,
    pub completed_time: Mutex<Option<std::time::Instant>>,

    trace_pos: RwLock<usize>,
    trace: model::MemAccessTrace,
}

impl PartialEq for Kernel {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl std::fmt::Display for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kernel")
            .field("name", &self.name())
            .field("id", &self.id())
            .finish()
    }
}

impl Kernel {
    pub fn new(
        config: model::command::KernelLaunch,
        trace: model::MemAccessTrace,
        // memory_only: bool,
    ) -> Self {
        // sanity check
        trace.check_valid().expect("valid trace");

        // check if grid size is equal to the number of unique blocks in the trace
        let all_blocks: HashSet<_> = trace.iter().map(|t| &t.block_id).collect();
        log::info!(
            "parsed kernel trace for {:?}: {}/{} blocks",
            config.unmangled_name,
            all_blocks.len(),
            config.grid.size(),
        );
        assert_eq!(config.grid.size(), all_blocks.len() as u64);

        let opcodes = opcodes::get_opcode_map(&config).unwrap();
        Self {
            opcodes,
            config,
            num_cores_running: 0,
            memory_only: false,
            start_cycle: Mutex::new(None),
            start_time: Mutex::new(None),
            completed_cycle: Mutex::new(None),
            completed_time: Mutex::new(None),
            trace,
            trace_pos: RwLock::new(0),
        }
    }

    // pub fn launched(&self) -> bool {
    //     *self.launched.try_lock()
    // }

    #[inline]
    pub fn launched(&self) -> bool {
        self.start_cycle.lock().is_some()
    }

    #[inline]
    pub fn completed(&self) -> bool {
        self.completed_cycle.lock().is_some()
    }

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

        let trace = crate::timeit!("read trace", read_trace(trace_path).unwrap());
        Self::new(config, trace)
    }

    // pub fn shared_memory_bytes_human_readable(&self) -> String {
    //     human_bytes::human_bytes(f64::from(self.config.shared_mem_bytes))
    // }

    // pub fn set_launched(&self) {
    //     *self.launched.try_lock() = true;
    // }

    #[inline]
    pub fn id(&self) -> u64 {
        self.config.id
    }

    pub fn next_threadblock_traces(&self, warps: &mut [warp::Ref], config: &config::GPU) -> bool {
        let mut trace_pos = self.trace_pos.write();

        let mut instructions = 0;
        let trace_size = self.trace.len();

        if *trace_pos + 1 >= trace_size || trace_size == 0 {
            // no more threadblocks
            log::info!("blocks done: no more threadblock traces");
            return false;
        }
        let next_block = &self.trace[*trace_pos + 1].block_id;

        while *trace_pos < trace_size {
            let entry = &self.trace[*trace_pos];
            if entry.block_id != *next_block {
                // get instructions until new block
                break;
            }

            let warp_id = entry.warp_id_in_block as usize;
            let instr = instruction::WarpInstruction::from_trace(self, entry, config);

            // if instr.active_mask.not_any() {
            log::error!(
                "instruction #{}: {:<30} {}",
                *trace_pos,
                instr.to_string(),
                instr.active_mask
            );
            // } else {
            //     log::warn!(
            //         "instruction #{}: {:<30} {}",
            //         *trace_pos,
            //         instr.to_string(),
            //         instr.active_mask
            //     );
            // }
            // log::error!(
            //     "instruction #{}: {:#?} {}",
            //     *trace_pos,
            //     instr,
            //     instr.active_mask
            // );

            // assert!(instr.is_memory_instruction());
            if !self.memory_only || instr.is_memory_instruction() {
                let warp = warps.get_mut(warp_id).unwrap();
                let mut warp = warp.try_lock();
                warp.push_trace_instruction(instr);
            } else {
                log::error!("SKIP non memory instruction {}", instr);
                // panic!("skipped instruction");
            }

            instructions += 1;
            *trace_pos += 1;
        }

        log::debug!(
            "added {instructions} instructions ({} per warp) for block {next_block}",
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

    // pub fn inc_running(&mut self) {
    //     self.num_cores_running += 1;
    // }

    #[inline]
    pub fn name(&self) -> &str {
        &self.config.unmangled_name
    }

    // pub fn was_launched(&self) -> bool {
    //     *self.launched.try_lock()
    // }

    #[inline]
    pub fn running(&self) -> bool {
        self.num_cores_running > 0
    }

    #[inline]
    pub fn current_block(&self) -> Option<model::Point> {
        let traces_pos = self.trace_pos.try_read();
        let trace = self.trace.get(*traces_pos)?;
        Some(model::Point::new(
            trace.block_id.clone(),
            self.config.grid.clone(),
        ))
    }

    #[inline]
    pub fn done(&self) -> bool {
        self.no_more_blocks_to_run() && !self.running()
    }

    #[inline]
    pub fn num_blocks(&self) -> usize {
        let grid = &self.config.grid;
        grid.x as usize * grid.y as usize * grid.z as usize
    }

    #[inline]
    pub fn threads_per_block(&self) -> usize {
        let block = &self.config.block;
        block.x as usize * block.y as usize * block.z as usize
    }

    #[inline]
    pub fn no_more_blocks_to_run(&self) -> bool {
        self.current_block().is_none()
    }
}
