use accelsim::tracegen;
use clap::Parser;
use color_eyre::eyre;
use console::style;
use itertools::Itertools;
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

    #[clap(
        short = 'm',
        long = "mem-only",
        help = "filter trace for memory instructions"
    )]
    pub memory_only: bool,

    #[clap(short = 'v', long = "verbose", help = "verbose")]
    pub verbose: bool,

    #[clap(subcommand)]
    pub command: Command,
}

fn parse_accelsim_traces(
    trace_dir: &Path,
    commands: &Path,
    mem_only: bool,
) -> eyre::Result<tracegen::reader::CommandTraces> {
    let mut command_traces = tracegen::reader::read_command_traces(trace_dir, commands, mem_only)?;

    // note: accelsim kernel launch ids start at index 1
    for (cmd, _) in &mut command_traces {
        if let Some(trace_model::Command::KernelLaunch(kernel)) = cmd {
            kernel.id = kernel
                .id
                .checked_sub(1)
                .expect("accelsim kernel launch ids start at index 1");
        }
    }

    Ok(command_traces)
}

fn parse_box_kernel_trace(
    kernel_trace_path: &Path,
    _mem_only: bool,
) -> eyre::Result<trace_model::MemAccessTrace> {
    let mut reader = utils::fs::open_readable(kernel_trace_path)?;
    let trace: trace_model::MemAccessTrace = rmp_serde::from_read(&mut reader)?;
    Ok(trace)
}

fn parse_box_traces(
    trace_dir: &Path,
    commands_or_trace: &Path,
    mem_only: bool,
) -> eyre::Result<tracegen::reader::CommandTraces> {
    if commands_or_trace
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        == Some(".json")
    {
        let commands: Vec<trace_model::Command> = {
            let reader = utils::fs::open_readable(commands_or_trace)?;

            serde_json::from_reader(reader)?
        };

        let command_traces: tracegen::reader::CommandTraces = commands
            .into_iter()
            .map(|cmd| match cmd {
                trace_model::Command::KernelLaunch(ref kernel) => {
                    let kernel_trace_path = trace_dir.join(&kernel.trace_file);
                    let trace = parse_box_kernel_trace(&kernel_trace_path, mem_only)?;
                    Ok::<_, eyre::Report>((Some(cmd), Some(trace)))
                }
                _ => Ok((Some(cmd), None)),
            })
            .try_collect()?;

        Ok(command_traces)
    } else {
        let trace = parse_box_kernel_trace(commands_or_trace, mem_only)?;
        Ok(vec![(None, Some(trace))])
    }
}

type CommandTraces = Vec<(Option<TraceCommand>, Option<WarpTraces>)>;

fn parse_trace(
    trace_dir: &Path,
    commands_or_trace: &Path,
    mem_only: bool,
) -> eyre::Result<CommandTraces> {
    let box_trace = parse_box_traces(trace_dir, commands_or_trace, mem_only);
    let accelsim_trace = parse_accelsim_traces(trace_dir, commands_or_trace, mem_only);
    let traces = match (box_trace, accelsim_trace) {
        (Ok(box_trace), Err(_)) => Ok(box_trace),
        (Err(_), Ok(accelsim_trace)) => Ok(accelsim_trace),
        (Err(box_err), Err(accelsim_err)) => Err(eyre::Report::from(box_err)
            .wrap_err(accelsim_err)
            .wrap_err(eyre::eyre!(
                "trace {} is neither a valid accelsim or box trace",
                commands_or_trace.display()
            ))),
        (Ok(_), Ok(_)) => {
            unreachable!(
                "trace {} is both a valid accelsim and box trace",
                commands_or_trace.display()
            )
        }
    }?;

    let traces = traces
        .into_iter()
        .map(|(cmd, trace)| {
            (
                cmd.map(TraceCommand),
                trace
                    .map(trace_model::MemAccessTrace::to_warp_traces)
                    .map(WarpTraces::from),
            )
        })
        .collect();

    Ok(traces)
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

    pub fn is_kernel_launch(&self) -> bool {
        matches!(self.0, trace_model::Command::KernelLaunch(_))
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

#[derive(Clone)]
#[repr(transparent)]
pub struct InstructionComparator(trace_model::MemAccessTraceEntry);

impl InstructionComparator {
    pub fn id(
        &self,
    ) -> (
        &trace_model::Dim,
        u32,
        u32,
        u32,
        &String,
        &gpucachesim::warp::ActiveMask,
    ) {
        let inst = &self.0;
        (
            &inst.block_id,
            inst.warp_id_in_block,
            inst.instr_idx,
            inst.instr_offset,
            &inst.instr_opcode,
            &inst.active_mask,
        )
    }
}

impl std::cmp::Eq for InstructionComparator {}

impl std::cmp::PartialEq for InstructionComparator {
    fn eq(&self, other: &Self) -> bool {
        self.id().eq(&other.id())
    }
}

impl std::fmt::Display for InstructionComparator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inst = &self.0;
        write!(
            f,
            "     [ block {} warp{:>3} ]\t inst_idx={:<4}  offset={:<4}\t {:<20}\t\t active={}",
            inst.block_id,
            inst.warp_id_in_block,
            inst.instr_idx,
            inst.instr_offset,
            inst.instr_opcode,
            inst.active_mask,
        )
    }
}

impl std::fmt::Debug for InstructionComparator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceInstruction {
    Valid(InstructionComparator),
    Filtered(InstructionComparator),
}

impl AsRef<InstructionComparator> for TraceInstruction {
    fn as_ref(&self) -> &InstructionComparator {
        match self {
            Self::Valid(inst) => &inst,
            Self::Filtered(inst) => &inst,
        }
    }
}

impl TraceInstruction {
    pub fn into_inner(self) -> InstructionComparator {
        match self {
            Self::Valid(inst) => inst,
            Self::Filtered(inst) => inst,
        }
    }

    pub fn is_filtered(&self) -> bool {
        matches!(self, Self::Filtered(_))
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
                    let v = v
                        .into_iter()
                        .map(InstructionComparator)
                        .map(TraceInstruction::Valid)
                        .collect();
                    (k, v)
                })
                .collect(),
        )
    }
}

fn print_trace(warp_traces: &WarpTraces, verbose: bool) {
    for ((block_id, warp_id), trace) in &warp_traces.0 {
        println!(
            "#### block={:<10} warp={:<2}",
            block_id.to_string(),
            warp_id
        );
        for (_trace_idx, entry) in trace.iter().enumerate() {
            if verbose {
                println!("{:#?}", &entry.as_ref());
            } else {
                println!("{}", &entry.as_ref());
            }
        }
    }
}

fn trace_info(commands_or_trace: &Path, mem_only: bool, verbose: bool) -> eyre::Result<()> {
    let trace_dir = commands_or_trace.parent().unwrap();
    let command_traces = parse_trace(trace_dir, commands_or_trace, mem_only)?;

    for (i, (cmd, warp_traces)) in command_traces.iter().enumerate() {
        println!("command {i}: {cmd:?}");
        let Some(warp_traces) = warp_traces else {
            continue;
        };

        print_trace(warp_traces, verbose);
    }
    Ok(())
}

fn compare_trace(
    left: Option<&WarpTraces>,
    left_path: &Path,
    right: Option<&WarpTraces>,
    right_path: &Path,
    mem_only: bool,
) {
    let (left_label, right_label) = match common_path::common_path(left_path, right_path) {
        Some(ref common_prefix) => (
            left_path
                .strip_prefix(common_prefix)
                .unwrap_or(left_path)
                .display(),
            right_path
                .strip_prefix(common_prefix)
                .unwrap_or(right_path)
                .display(),
        ),
        None => (left_path.display(), right_path.display()),
    };

    match (left, right) {
        (Some(left), Some(right)) => {
            let left_warps: Vec<_> = left
                .0
                .keys()
                .sorted_by_key(|(warp_id, _)| warp_id.clone())
                .collect();
            let right_warps: Vec<_> = right
                .0
                .keys()
                .sorted_by_key(|(warp_id, _)| warp_id.clone())
                .collect();

            if left_warps != right_warps {
                diff!(left: left_warps, right: right_warps);
            }
            let all_warps: Vec<_> = left_warps
                .iter()
                .chain(right_warps.iter())
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .sorted()
                .collect();

            for warp_id in all_warps.iter() {
                let (block_id, warp_id_in_block) = warp_id;
                let left_warp_trace: Vec<_> = left
                    .0
                    .get(*warp_id)
                    .unwrap_or(&vec![])
                    .iter()
                    .cloned()
                    .map(|inst| {
                        if mem_only && !inst.as_ref().0.is_memory_instruction() {
                            TraceInstruction::Filtered(inst.into_inner())
                        } else {
                            inst
                        }
                    })
                    .collect();

                let right_warp_trace: Vec<_> = right
                    .0
                    .get(*warp_id)
                    .unwrap_or(&vec![])
                    .iter()
                    .cloned()
                    .map(|inst| {
                        if mem_only && !inst.as_ref().0.is_memory_instruction() {
                            TraceInstruction::Filtered(inst.into_inner())
                        } else {
                            inst
                        }
                    })
                    .collect();

                // if mem_only {
                //     for inst in left_warp_trace.iter_mut() {
                //         // .filter(Option::is_some) {
                //         // match inst {
                //         //     Some(inst) if inst.0.is_memory_instruction() => {
                //         //     },
                //         //     _ =>
                //         // }
                //         if !inst.0.is_memory_instruction() {
                //             *inst = None;
                //         }
                //         // trace.0.retain(|t| t.is_memory_instruction());
                //     }
                // }
                // if let Some(left_warp_trace) = left_warp_trace {
                //     if mem_only {
                //         trace.0.retain(|t| t.is_memory_instruction());
                //     }
                // }

                // let right_warp_trace = right.0.get(*warp_id);

                // for inst in &left_warp_trace {
                //     println!("{:?}", &inst);
                // }
                // for inst in &right_warp_trace {
                //     println!("{:?}", &inst);
                // }

                // let filtered
                let (left_valid, left_filtered): (Vec<_>, Vec<_>) =
                    partition_instructions(left_warp_trace);
                let (right_valid, right_filtered): (Vec<_>, Vec<_>) =
                    partition_instructions(right_warp_trace);

                println!(
                    "\t=> block {} warp {:<3} filtered: {} (left) {} (right)",
                    block_id,
                    warp_id_in_block,
                    left_filtered.len(),
                    right_filtered.len(),
                );
                // diff!(left: left_valid, right: right_valid);
                diff!(left_label, left_valid, right_label, right_valid);
            }
        }
        (Some(left), None) => {
            println!(
                "{}",
                style(format!(
                    "=> have trace with {} warps in left ({}) but not in right ({})",
                    left.0.len(),
                    left_path.display(),
                    right_path.display(),
                ))
                .red()
            );
        }
        (None, Some(right)) => {
            println!(
                "{}",
                style(format!(
                    "have trace with {} warps in right ({}) but not in left ({})",
                    right.0.len(),
                    right_path.display(),
                    left_path.display()
                ))
                .red()
            );
        }
        (None, None) => {}
    }
}

fn compare_traces(
    left_commands_or_trace: &Path,
    right_commands_or_trace: &Path,
    print_traces: bool,
    mem_only: bool,
    verbose: bool,
) -> eyre::Result<()> {
    // use itertools::Itertools;
    use indexmap::{IndexMap, IndexSet};
    // use std::collections::{HashMap, HashSet};

    let left_command_traces: IndexMap<_, _> = {
        let trace_dir = left_commands_or_trace.parent().unwrap();
        let command_traces = parse_trace(trace_dir, left_commands_or_trace, mem_only)?;
        command_traces.into_iter().collect()
    };
    let right_command_traces: IndexMap<_, _> = {
        let trace_dir = right_commands_or_trace.parent().unwrap();
        let command_traces = parse_trace(trace_dir, right_commands_or_trace, mem_only)?;
        command_traces.into_iter().collect()
    };

    // either we are comparing exactly two traces, or we have kernel launch commands to perform the
    // mapping
    // let comparing_exactly_two_kernel_traces =
    //     left_command_traces.len() <= 1 && right_command_traces.len() <= 1;
    // let have_kernel_launch_information = left_command_traces.keys().all(Option::is_some)
    //     && right_command_traces.keys().all(Option::is_some);

    // if !(comparing_exactly_two_kernel_traces || have_kernel_launch_information) {
    //     eyre::bail!(
    //         "cannot compare {} items from {} with {} entries from {}",
    //         left_command_traces.len(),
    //         left_commands_or_trace.display(),
    //         right_command_traces.len(),
    //         right_commands_or_trace.display()
    //     );
    // }

    // compare kernel launches first
    let left_kernel_launches: IndexMap<_, _> = left_command_traces
        .iter()
        .filter(|(cmd, traces)| {
            cmd.as_ref().is_some_and(TraceCommand::is_kernel_launch) || traces.is_some()
        })
        .collect();
    let right_kernel_launches: IndexMap<_, _> = right_command_traces
        .iter()
        .filter(|(cmd, traces)| {
            cmd.as_ref().is_some_and(TraceCommand::is_kernel_launch) || traces.is_some()
        })
        .collect();
    for kernel_launches in [&left_kernel_launches, &right_kernel_launches] {
        dbg!(kernel_launches.len());
        dbg!(kernel_launches
            .values()
            .next()
            .unwrap()
            .as_ref()
            .unwrap()
            .0
            .len());
    }

    let kernel_launch_commands: IndexSet<_> = left_kernel_launches
        .keys()
        .chain(right_kernel_launches.keys())
        .copied()
        .filter_map(Option::as_ref)
        .collect();

    if kernel_launch_commands.is_empty() {
        let num_kernel_traces = left_kernel_launches.len().max(right_kernel_launches.len());
        let left_traces = left_kernel_launches
            .values()
            .copied()
            .chain(std::iter::repeat(&None));
        let right_traces = right_kernel_launches
            .values()
            .copied()
            .chain(std::iter::repeat(&None));

        // if left_command_traces.len() <= 1 && right_command_traces.len() <= 1

        for ((i, left), right) in (0..num_kernel_traces)
            .into_iter()
            .zip(left_traces)
            .zip(right_traces)
        {
            dbg!(i, left.is_some(), right.is_some());
            compare_trace(
                left.as_ref(),
                left_commands_or_trace,
                right.as_ref(),
                right_commands_or_trace,
                mem_only,
            );
        }
    } else {
        todo!()
    }

    // for launch_command in kernel_launch_commands {}
    // .sorted_by_key(|&cmd| cmd.as_ref().map(TraceCommand::key))
    // .collect();

    // let left_commands: HashSet<_> = left_command_traces.keys().collect();
    // let right_commands: HashSet<_> = right_command_traces.keys().collect();
    // let all_commands: Vec<_> = left_commands
    //     .union(&right_commands)
    //     .copied()
    //     .sorted_by_key(|&cmd| cmd.as_ref().map(TraceCommand::key))
    //     .collect();
    //
    // for (_cmd_idx, cmd) in all_commands.into_iter().enumerate() {
    //     if matches!(
    //         cmd,
    //         Some(TraceCommand(
    //             trace_model::Command::MemcpyHtoD(_) | trace_model::Command::MemAlloc(_)
    //         ))
    //     ) {
    //         continue;
    //     }
    //     println!("===> command {cmd:?}");
    //     println!("left: {}", left_commands_or_trace.display());
    //     println!("right: {}", right_commands_or_trace.display());
    //     match (
    //         left_command_traces.contains_key(cmd),
    //         right_command_traces.contains_key(cmd),
    //     ) {
    //         (true, true) => {}
    //         (false, false) => unreachable!(),
    //         (false, true) => diff::diff!(left: None::<trace_model::Command>, right: Some(cmd)),
    //         (true, false) => diff::diff!(left: Some(cmd), right: None::<trace_model::Command>),
    //     }
    //     let left_trace = left_command_traces
    //         .get(cmd)
    //         .and_then(Option::as_ref)
    //         .cloned()
    //         .unwrap_or_default();
    //     let right_trace = right_command_traces
    //         .get(cmd)
    //         .and_then(Option::as_ref)
    //         .cloned()
    //         .unwrap_or_default();
    //
    //     if left_trace != right_trace {
    //         utils::diff::diff!(left: &left_trace, right: &right_trace);
    //     } else if print_traces {
    //         // print matching trace
    //         print_trace(&left_trace, verbose);
    //     }
    //     println!();
    // }

    Ok(())
}

pub fn partition_instructions<C, FC>(
    instructions: impl IntoIterator<Item = TraceInstruction>,
) -> (C, FC)
where
    C: std::iter::FromIterator<InstructionComparator>,
    FC: std::iter::FromIterator<InstructionComparator>,
{
    let (filtered, valid): (Vec<_>, Vec<_>) = instructions
        .into_iter()
        .partition(TraceInstruction::is_filtered);
    let valid: C = valid
        .into_iter()
        .map(TraceInstruction::into_inner)
        .collect();
    let filtered: FC = filtered
        .into_iter()
        .map(TraceInstruction::into_inner)
        .collect();
    (valid, filtered)
}

pub fn run(options: &Options) -> eyre::Result<()> {
    match options.command {
        Command::Info => {
            for trace in &options.traces {
                trace_info(trace, options.memory_only, options.verbose)?;
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
            compare_traces(
                left_trace,
                right_trace,
                print,
                options.memory_only,
                options.verbose,
            )?;
        }
    }

    Ok(())
}
