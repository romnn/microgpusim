use accelsim::tracegen;
use clap::Parser;
use color_eyre::eyre;
use std::path::{Path, PathBuf};
use utils::diff;

#[derive(Parser, Debug, Clone)]
pub enum Command {
    Info,
    Compare {
        #[clap(long = "print", help = "print matching traces")]
        print: bool,
    },
}

#[derive(Parser, Debug, Clone)]
pub struct Options {
    #[clap(short = 'f', long = "file", help = "path to trace file")]
    pub traces: Vec<PathBuf>,

    #[clap(subcommand)]
    pub command: Command,
}

fn parse_accelsim_traces(
    trace_dir: &Path,
    commands: &Path,
    mem_only: bool,
) -> eyre::Result<tracegen::reader::CommandTraces> {
    let mut command_traces =
        tracegen::reader::read_traces_for_commands(trace_dir, commands, mem_only)?;

    // note: accelsim kernel launch ids start at index 1
    for (cmd, _) in &mut command_traces {
        if let trace_model::Command::KernelLaunch(kernel) = cmd {
            kernel.id = kernel
                .id
                .checked_sub(1)
                .expect("accelsim kernel launch ids start at index 1");
        }
    }

    Ok(command_traces)
}

fn parse_box_traces(
    trace_dir: &Path,
    commands: &Path,
) -> eyre::Result<tracegen::reader::CommandTraces> {
    use itertools::Itertools;

    let commands: Vec<trace_model::Command> = {
        let reader = utils::fs::open_readable(commands)?;

        serde_json::from_reader(reader)?
    };

    let command_traces = commands
        .into_iter()
        .map(|cmd| match cmd {
            trace_model::Command::KernelLaunch(ref kernel) => {
                let kernel_trace_path = trace_dir.join(&kernel.trace_file);
                let mut reader = utils::fs::open_readable(kernel_trace_path)?;
                let trace: trace_model::MemAccessTrace = rmp_serde::from_read(&mut reader)?;
                Ok::<_, eyre::Report>((cmd, Some(trace)))
            }
            _ => Ok((cmd, None)),
        })
        .try_collect()?;
    Ok(command_traces)
}

type CommandTraces = Vec<(TraceCommand, Option<WarpTraces>)>;

fn parse_trace(
    trace_dir: &Path,
    commands_path: &Path,
    mem_only: bool,
) -> eyre::Result<CommandTraces> {
    let commands_traces = parse_box_traces(trace_dir, commands_path)
        .or_else(|_err| parse_accelsim_traces(trace_dir, commands_path, mem_only))?;

    let commands_traces = commands_traces
        .into_iter()
        .map(|(cmd, trace)| {
            (
                TraceCommand(cmd),
                trace
                    .map(trace_model::MemAccessTrace::to_warp_traces)
                    .map(WarpTraces::from),
            )
        })
        .collect();

    Ok(commands_traces)
}

#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum TraceCommandKey {
    MemcpyHtoD {},
    MemAlloc {},
    KernelLaunch { id: u64 },
}

#[allow(clippy::module_name_repetitions)]
#[derive(Debug)]
pub struct TraceCommand(trace_model::Command);

impl TraceCommand {
    pub fn key(&self) -> TraceCommandKey {
        match &self.0 {
            trace_model::Command::MemcpyHtoD(_) => TraceCommandKey::MemcpyHtoD {},
            trace_model::Command::MemAlloc(_) => TraceCommandKey::MemAlloc {},
            trace_model::Command::KernelLaunch(k) => TraceCommandKey::KernelLaunch { id: k.id },
        }
    }
}
impl std::cmp::Eq for TraceCommand {}

impl std::cmp::PartialEq for TraceCommand {
    fn eq(&self, other: &Self) -> bool {
        self.key().eq(&other.key())
    }
}

impl std::hash::Hash for TraceCommand {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.key().hash(state);
    }
}

#[allow(clippy::module_name_repetitions)]
#[derive(Clone)]
pub struct TraceInstruction(trace_model::MemAccessTraceEntry);

impl TraceInstruction {
    pub fn active_mask(&self) -> gpucachesim::warp::ActiveMask {
        use bitvec::field::BitField;
        let mut active_mask = gpucachesim::warp::ActiveMask::ZERO;
        active_mask.store(self.0.active_mask);
        active_mask
    }

    fn id(
        &self,
    ) -> (
        &trace_model::Dim,
        u32,
        u32,
        u32,
        &String,
        gpucachesim::warp::ActiveMask,
    ) {
        (
            &self.0.block_id,
            self.0.warp_id_in_block,
            self.0.instr_idx,
            self.0.instr_offset,
            &self.0.instr_opcode,
            self.active_mask(),
        )
    }
}

impl std::cmp::Eq for TraceInstruction {}

impl std::cmp::PartialEq for TraceInstruction {
    fn eq(&self, other: &Self) -> bool {
        self.id().eq(&other.id())
    }
}

impl std::fmt::Display for TraceInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use trace_model::ToBitString;
        write!(
            f,
            "     [ block {} warp{:>3} ]\t inst_idx={:<4}  offset={:<4}\t {:<20}\t\t active={}",
            self.0.block_id,
            self.0.warp_id_in_block,
            self.0.instr_idx,
            self.0.instr_offset,
            self.0.instr_opcode,
            self.active_mask().to_bit_string(),
        )
    }
}

impl std::fmt::Debug for TraceInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct WarpTraces(pub indexmap::IndexMap<(trace_model::Dim, u32), Vec<TraceInstruction>>);

impl From<trace_model::WarpTraces> for WarpTraces {
    fn from(trace: trace_model::WarpTraces) -> Self {
        Self(
            trace
                .into_iter()
                .map(|(k, v)| {
                    let v = v.into_iter().map(TraceInstruction).collect();
                    (k, v)
                })
                .collect(),
        )
    }
}

fn print_trace(warp_traces: &WarpTraces) {
    for ((block_id, warp_id), trace) in &warp_traces.0 {
        println!(
            "#### block={:<10} warp={:<2}",
            block_id.to_string(),
            warp_id
        );
        for (_trace_idx, entry) in trace.iter().enumerate() {
            println!("{entry}");
        }
    }
}

fn trace_info(commands: &Path) -> eyre::Result<()> {
    let trace_dir = commands.parent().unwrap();
    let mem_only = false;
    let command_traces = parse_trace(trace_dir, commands, mem_only)?;

    for (i, (cmd, warp_traces)) in command_traces.iter().enumerate() {
        println!("command {i}: {cmd:?}");
        let Some(warp_traces) = warp_traces else {
            continue;
        };

        print_trace(&warp_traces);
    }
    Ok(())
}

fn compare_traces(
    left_commands_path: &Path,
    right_commands_path: &Path,
    print_traces: bool,
) -> eyre::Result<()> {
    use itertools::Itertools;
    use std::collections::{HashMap, HashSet};

    let mem_only = false;
    let left_command_traces: HashMap<_, _> = {
        let trace_dir = left_commands_path.parent().unwrap();
        let command_traces = parse_trace(trace_dir, left_commands_path, mem_only)?;
        command_traces.into_iter().collect()
    };
    let right_command_traces: HashMap<_, _> = {
        let trace_dir = right_commands_path.parent().unwrap();
        let command_traces = parse_trace(trace_dir, right_commands_path, mem_only)?;
        command_traces.into_iter().collect()
    };

    let left_commands: HashSet<_> = left_command_traces.keys().collect();
    let right_commands: HashSet<_> = right_command_traces.keys().collect();
    let all_commands: Vec<_> = left_commands
        .union(&right_commands)
        .sorted_by_key(|cmd| cmd.key())
        .collect();

    for (_cmd_idx, cmd) in all_commands.iter().enumerate() {
        if matches!(
            cmd.0,
            trace_model::Command::MemcpyHtoD(_) | trace_model::Command::MemAlloc(_)
        ) {
            continue;
        }
        println!("===> command {cmd:?}");
        println!("left: {}", left_commands_path.display());
        println!("right: {}", right_commands_path.display());
        match (
            left_command_traces.contains_key(cmd),
            right_command_traces.contains_key(cmd),
        ) {
            (true, true) => {}
            (false, false) => unreachable!(),
            (false, true) => diff::diff!(left: None::<trace_model::Command>, right: Some(cmd)),
            (true, false) => diff::diff!(left: Some(cmd), right: None::<trace_model::Command>),
        }
        let left_trace = left_command_traces
            .get(cmd)
            .and_then(Option::as_ref)
            .cloned()
            .unwrap_or_default();
        let right_trace = right_command_traces
            .get(cmd)
            .and_then(Option::as_ref)
            .cloned()
            .unwrap_or_default();

        if left_trace != right_trace {
            utils::diff::diff!(left: &left_trace, right: &right_trace);
        } else if print_traces {
            // print matching trace
            print_trace(&left_trace);
        }
        println!();
    }

    Ok(())
}

pub fn run(options: &Options) -> eyre::Result<()> {
    match options.command {
        Command::Info => {
            for trace in &options.traces {
                trace_info(trace)?;
            }
        }
        Command::Compare { print } => {
            if options.traces.len() != 2 {
                eyre::bail!(
                    "can only compare exactly 2 trace files, got {}",
                    options.traces.len()
                );
            }
            let left_trace = &options.traces[0];
            let right_trace = &options.traces[1];
            compare_traces(left_trace, right_trace, print)?;
        }
    }

    Ok(())
}
