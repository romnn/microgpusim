use super::{instruction, opcodes, scheduler as sched};
use color_eyre::{
    eyre::{self},
    Help,
};
use std::collections::HashSet;
use std::path::Path;
use std::sync::{Mutex, RwLock};
use std::time::Instant;
use trace_model::{KernelLaunch, MemAccessTraceEntry, Point};

pub fn read_trace(path: impl AsRef<Path>) -> eyre::Result<Vec<MemAccessTraceEntry>> {
    use serde::Deserializer;

    let reader = utils::fs::open_readable(path.as_ref())?;
    let mut reader = rmp_serde::Deserializer::new(reader);
    let mut trace = vec![];
    let decoder = nvbit_io::Decoder::new(|access: MemAccessTraceEntry| {
        trace.push(access);
    });
    reader.deserialize_seq(decoder).suggestion("maybe the traces does not match the most recent binary trace format, try re-generating the traces.")?;
    Ok(trace)
}

/// Kernel represents a kernel.
///
/// This includes its launch configuration,
/// as well as its state of execution.
#[derive(Debug)]
pub struct Kernel {
    pub opcodes: &'static opcodes::OpcodeMap,
    pub config: KernelLaunch,
    trace: Vec<MemAccessTraceEntry>,
    trace_pos: RwLock<usize>,
    launched: Mutex<bool>,
    num_cores_running: usize,
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
    pub fn from_trace(traces_dir: impl AsRef<Path>, config: KernelLaunch) -> Self {
        let start = Instant::now();
        log::info!(
            "parsing kernel for launch {:?} from {}",
            &config,
            &config.trace_file
        );
        let trace_path = traces_dir
            .as_ref()
            .join(&config.trace_file)
            .with_extension("msgpack");

        let trace = read_trace(trace_path).unwrap();

        // sanity check
        assert!(trace_model::is_valid_trace(&trace));

        // check if grid size is equal to the number of unique blocks in the trace
        let all_blocks: HashSet<_> = trace.iter().map(|t| &t.block_id).collect();
        log::info!(
            "parsed kernel trace for {:?}: {}/{} blocks in {:?}",
            config.name,
            all_blocks.len(),
            config.grid.size(),
            start.elapsed()
        );
        assert_eq!(config.grid.size(), all_blocks.len() as u64);

        let opcodes = opcodes::get_opcode_map(&config).unwrap();

        Self {
            config,
            trace,
            trace_pos: RwLock::new(0),
            opcodes,
            launched: Mutex::new(false),
            num_cores_running: 0,
        }
    }

    pub fn shared_memory_bytes_human_readable(&self) -> String {
        human_bytes::human_bytes(f64::from(self.config.shared_mem_bytes))
    }

    pub fn set_launched(&self) {
        *self.launched.lock().unwrap() = true;
    }

    pub fn launched(&self) -> bool {
        *self.launched.lock().unwrap()
    }

    pub fn id(&self) -> u64 {
        self.config.id
    }

    pub fn next_threadblock_traces(&self, warps: &mut [sched::WarpRef]) {
        let mut trace_pos = self.trace_pos.write().unwrap();

        let mut instructions = 0;
        let trace_size = self.trace.len();

        if *trace_pos + 1 >= trace_size || trace_size == 0 {
            // no more threadblocks
            log::info!("blocks done: no more threadblock traces");
            return;
        }
        let next_block = &self.trace[*trace_pos + 1].block_id;

        while *trace_pos < trace_size {
            let entry = &self.trace[*trace_pos];
            if entry.block_id != *next_block {
                // get instructions until new block
                break;
            }

            let warp_id = entry.warp_id_in_block as usize;
            let instr = instruction::WarpInstruction::from_trace(self, entry.clone());
            let warp = warps.get_mut(warp_id).unwrap();
            // let mut warp = warp.try_borrow_mut().unwrap();
            let mut warp = warp.try_lock().unwrap();
            warp.push_trace_instruction(instr);

            instructions += 1;
            *trace_pos += 1;
        }

        log::debug!(
            "added {instructions} instructions ({} per warp) for block {next_block}",
            instructions / warps.len()
        );
        debug_assert!(instructions > 0);
        // debug_assert!(instructions % 32 == 0);
        // dbg!(warps
        //     .iter()
        //     .map(|w| w.try_borrow().unwrap().trace_instructions.len())
        //     .collect::<Vec<_>>());
        // debug_assert!(
        //     warps
        //         .iter()
        //         .map(|w| w.try_borrow().unwrap().trace_instructions.len())
        //         .collect::<HashSet<_>>()
        //         .len()
        //         == 1,
        //     "all warps have the same number of instructions"
        // );
        // dbg!(warps
        //     .iter()
        //     .map(|w| w.try_borrow().unwrap().trace_instructions.len())
        //     .collect::<Vec<_>>());

        debug_assert!(
            warps
                .iter()
                .all(|w| !w.try_lock().unwrap().trace_instructions.is_empty()),
            // .all(|w| !w.try_borrow().unwrap().trace_instructions.is_empty()),
            "all warps have at least one instruction (need at least an EXIT)"
        );
    }

    pub fn inc_running(&mut self) {
        self.num_cores_running += 1;
    }

    pub fn name(&self) -> &str {
        &self.config.name
    }

    pub fn was_launched(&self) -> bool {
        *self.launched.lock().unwrap()
    }

    pub fn running(&self) -> bool {
        self.num_cores_running > 0
    }

    pub fn current_block(&self) -> Option<Point> {
        let traces_pos = self.trace_pos.read().unwrap();
        let trace = self.trace.get(*traces_pos)?;
        Some(Point::new(trace.block_id.clone(), self.config.grid.clone()))
    }

    pub fn done(&self) -> bool {
        self.no_more_blocks_to_run() && !self.running()
    }

    pub fn num_blocks(&self) -> usize {
        let grid = &self.config.grid;
        grid.x as usize * grid.y as usize * grid.z as usize
    }

    pub fn threads_per_block(&self) -> usize {
        let block = &self.config.block;
        block.x as usize * block.y as usize * block.z as usize
    }

    pub fn no_more_blocks_to_run(&self) -> bool {
        self.current_block().is_none()
    }
}
