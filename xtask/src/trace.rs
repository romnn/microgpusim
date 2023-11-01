use accelsim::tracegen;
use clap::Parser;
use color_eyre::{eyre, Section, SectionExt};
use console::style;
use gpucachesim::allocation::Allocations;
use indexmap::{IndexMap, IndexSet};
use itertools::Itertools;
use once_cell::sync::Lazy;
use regex::Regex;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use trace_model::ToBitString;
use utils::diff;

#[derive(Parser, Debug, Clone)]
pub enum Command {
    Info {
        #[clap(long = "block", help = "get trace for specific block only")]
        block_id: Option<trace_model::Dim>,
        #[clap(
            long = "warp-id",
            aliases = ["warp"],
            help = "get trace for specific warp only"
        )]
        warp_id: Option<usize>,
        #[clap(
            long = "instruction-limit",
            help = "limit the number of instructions printed"
        )]
        instruction_limit: Option<usize>,
        #[clap(long = "summary", help = "summarize traces")]
        summary: bool,
    },
    Compare {
        #[clap(long = "print", help = "print matching traces")]
        print: bool,
        #[clap(long = "summary", help = "print trace summary")]
        summary: bool,
        #[clap(long = "block", help = "get trace for specific block only")]
        block_id: Option<trace_model::Dim>,
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

static ACCELSIM_KERNEL_TRACE_FILE_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"kernel-(\d+).*").unwrap());

fn parse_accelsim_traces(
    trace_dir: &Path,
    commands_or_trace: &Path,
    mem_only: bool,
) -> eyre::Result<tracegen::reader::CommandTraces> {
    use accelsim::tracegen::reader::Command as AccelsimCommand;
    let mut command_traces = if commands_or_trace
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        == Some("traceg")
    {
        let extract_kernel_launch_id = || {
            // extract kernel launch id from file name similar to "kernel-1.traceg"
            let file_name = commands_or_trace.file_stem().map(OsStr::to_str).flatten()?;
            let captures = ACCELSIM_KERNEL_TRACE_FILE_REGEX.captures(file_name)?;
            captures
                .get(0)
                .as_ref()
                .map(regex::Match::as_str)
                .map(str::parse)
                .map(Result::ok)
                .flatten()
        };

        let find_kernel_launch = |kernel_launch_id: u64| {
            let reader = utils::fs::open_readable(trace_dir.join("kernelslist.g"))?;
            let accelsim_commands = accelsim::tracegen::reader::read_commands(trace_dir, reader)?;

            // find matching launch
            let kernel_launch = accelsim_commands.into_iter().find_map(|cmd| match cmd {
                AccelsimCommand::KernelLaunch((kernel, metadata))
                    if kernel.id == kernel_launch_id =>
                {
                    Some((kernel, metadata))
                }
                _ => None,
            });
            Ok::<_, eyre::Report>(kernel_launch)
        };

        let kernel_launch = extract_kernel_launch_id()
            .and_then(|launch_id| find_kernel_launch(launch_id).ok())
            .flatten();

        let reader = utils::fs::open_readable(commands_or_trace)?;
        let trace_version = kernel_launch
            .as_ref()
            .map(|(_, metadata)| metadata.trace_version)
            .unwrap_or(4);
        let line_info = kernel_launch
            .as_ref()
            .map(|(_, metadata)| metadata.line_info)
            .unwrap_or(false);

        let trace = accelsim::tracegen::reader::read_trace_instructions(
            reader,
            trace_version,
            line_info,
            mem_only,
            kernel_launch.as_ref().map(|(kernel, _)| kernel),
        )?;

        Ok::<_, eyre::Report>(vec![(
            kernel_launch
                .map(|(kernel, _)| kernel)
                .map(trace_model::Command::KernelLaunch),
            Some(trace_model::MemAccessTrace(trace)),
        )])
    } else {
        tracegen::reader::read_command_traces(trace_dir, commands_or_trace, mem_only)
    }?;

    if command_traces.is_empty() {
        return Err(eyre::eyre!("command list is empty"));
    }

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

fn parse_box_kernel_trace(kernel_trace_path: &Path) -> eyre::Result<trace_model::MemAccessTrace> {
    let mut reader = utils::fs::open_readable(kernel_trace_path)?;
    let trace: trace_model::MemAccessTrace = rmp_serde::from_read(&mut reader)?;
    Ok(trace)
}

fn get_box_allocations(commands: &[trace_model::Command]) -> eyre::Result<Allocations> {
    let mut allocations = Allocations::default();
    for cmd in commands.iter() {
        if let trace_model::Command::MemAlloc(trace_model::command::MemAlloc {
            allocation_name,
            device_ptr,
            num_bytes,
        }) = cmd
        {
            let alloc_range = *device_ptr..(*device_ptr + num_bytes);
            allocations.insert(alloc_range, allocation_name.clone());
        }
    }

    Ok(allocations)
}

fn parse_box_commands(commands_path: &Path) -> eyre::Result<Vec<trace_model::Command>> {
    let reader = utils::fs::open_readable(commands_path)?;
    let commands = serde_json::from_reader(reader)?;
    Ok(commands)
}

fn parse_box_traces(
    trace_dir: &Path,
    commands_or_trace: &Path,
) -> eyre::Result<(tracegen::reader::CommandTraces, Option<Allocations>)> {
    if commands_or_trace
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        == Some("json")
    {
        let commands = parse_box_commands(commands_or_trace)?;
        let allocations = get_box_allocations(&commands).ok();
        let command_traces: tracegen::reader::CommandTraces = commands
            .into_iter()
            .map(|cmd| match cmd {
                trace_model::Command::KernelLaunch(ref kernel) => {
                    let kernel_trace_path = trace_dir.join(&kernel.trace_file);
                    let trace = parse_box_kernel_trace(&kernel_trace_path)?;
                    Ok::<_, eyre::Report>((Some(cmd), Some(trace)))
                }
                _ => Ok((Some(cmd), None)),
            })
            .try_collect()?;

        Ok((command_traces, allocations))
    } else {
        let commands_path_guess = commands_or_trace
            .with_file_name("commands.json")
            .with_extension("json");
        let allocations = parse_box_commands(&commands_path_guess)
            .ok()
            .and_then(|commands| get_box_allocations(&commands).ok());
        let trace = parse_box_kernel_trace(commands_or_trace)?;
        Ok((vec![(None, Some(trace))], allocations))
    }
}

use std::rc::Rc;

type CommandTraces = Vec<(Option<TraceCommand>, Option<WarpTraces>)>;
// type CommandTraces<'a> = Vec<(Option<TraceCommand>, Option<WarpTraces<'a>>)>;
// type CommandTraces = Vec<(Option<TraceCommand>, Option<WarpTraces>)>;

fn parse_trace(
    trace_dir: &Path,
    commands_or_trace: &Path,
    mem_only: bool,
) -> eyre::Result<CommandTraces> {
    // ) -> eyre::Result<(CommandTraces, Allocations)> {
    let box_result = parse_box_traces(trace_dir, commands_or_trace);
    let accelsim_result = parse_accelsim_traces(trace_dir, commands_or_trace, mem_only);
    let (traces, allocations) = match (box_result, accelsim_result) {
        (Ok(box_result), Err(_)) => Ok(box_result),
        (Err(_), Ok(accelsim_trace)) => Ok((accelsim_trace, None)),
        (Err(box_err), Err(accelsim_err)) => {
            let err = eyre::eyre!(
                "trace {} is neither a valid accelsim or box trace",
                commands_or_trace.display()
            )
            .with_section(|| box_err.header("box error:"))
            .with_section(|| accelsim_err.header("accelsim error:"));
            Err(err)
        }
        (Ok(_), Ok(_)) => {
            unreachable!(
                "trace {} is both a valid accelsim and box trace",
                commands_or_trace.display()
            )
        }
    }?;

    let allocations = Rc::new(allocations.unwrap_or_default());
    println!(
        "allocations {:#?}",
        allocations
            .iter()
            .map(|(_, alloc)| alloc)
            .filter(|alloc| alloc.num_bytes() > 32)
            .collect::<Vec<_>>()
    );

    let traces = traces
        .into_iter()
        .map(|(cmd, trace)| {
            let warp_traces = trace
                .map(trace_model::MemAccessTrace::to_warp_traces)
                .map(|trace| {
                    WarpTraces(
                        trace
                            .into_iter()
                            .map(|(k, v)| {
                                // let allocation = allocations.get(&addr);
                                // for addr in inst.inner.addrs.iter_mut() {
                                //     if let Some(allocation) = allocations.get(&addr) {
                                //         *addr = addr.saturating_sub(allocation.start_addr);
                                //     }
                                // }
                                let v = v
                                    .into_iter()
                                    .map(|inst| InstructionComparator {
                                        inner: inst,
                                        allocations: Rc::clone(&allocations),
                                    })
                                    .map(TraceInstruction::Valid)
                                    .collect();
                                (k, v)
                            })
                            .collect(),
                    )
                });

            (cmd.map(TraceCommand), warp_traces)
        })
        .collect();

    Ok(traces)
    // Ok((traces, allocations))
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
// pub struct InstructionComparator<'a> {
pub struct InstructionComparator {
    // allocation: Option<Allocation>,
    allocations: Rc<Allocations>,
    inner: trace_model::MemAccessTraceEntry,
}

impl InstructionComparator {
    // impl<'a> InstructionComparator<'a> {
    pub fn into_inner(self) -> trace_model::MemAccessTraceEntry {
        self.inner
    }

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
        (
            &self.inner.block_id,
            self.inner.warp_id_in_block,
            self.inner.instr_idx,
            self.inner.instr_offset,
            &self.inner.instr_opcode,
            &self.inner.active_mask,
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
        write!(
            f,
            "     [ block {} warp{:>3} ] inst_idx={:<4}  offset={:<4} {}\t active={}",
            self.inner.block_id,
            self.inner.warp_id_in_block,
            self.inner.instr_idx,
            self.inner.instr_offset,
            style(format!("{:>15}", &self.inner.instr_opcode)).cyan(),
            self.inner.active_mask.to_bit_string_colored(None),
        )
    }
}

fn format_addr(addr: u64, allocations: &Allocations) -> String {
    match allocations.get(&addr) {
        Some(allocation) => {
            let relative_addr = addr.saturating_sub(allocation.start_addr);
            format!("{}+{}", allocation.id, relative_addr)
        }
        None => format!("{}", addr),
    }
}

impl std::fmt::Debug for InstructionComparator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let valid_addresses = self.inner.valid_addresses().collect::<Vec<_>>();

        enum AddressFormat {
            Original(Vec<u64>),
            BaseStride {
                base: u64,
                stride: u64,
                count: usize,
            },
            SingleAddress {
                addr: u64,
                count: usize,
            },
        }

        let address_format = if valid_addresses.len() > 1 {
            let strides = valid_addresses
                .windows(2)
                .map(|w| w[1] - w[0])
                .collect::<Vec<_>>();
            if strides.iter().all_equal() {
                if strides[0] != 0 {
                    AddressFormat::BaseStride {
                        base: valid_addresses[0],
                        stride: strides[0],
                        count: valid_addresses.len(),
                    }
                } else {
                    AddressFormat::SingleAddress {
                        addr: valid_addresses[0],
                        count: valid_addresses.len(),
                    }
                }
            } else {
                AddressFormat::Original(valid_addresses)
            }
        } else {
            AddressFormat::Original(valid_addresses)
        };

        struct Addresses<'a> {
            format: AddressFormat,
            allocations: &'a Allocations,
        }

        impl<'a> std::fmt::Display for Addresses<'a> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match &self.format {
                    AddressFormat::Original(addresses) => {
                        write!(
                            f,
                            "{:>2}x [{}]",
                            addresses.len(),
                            addresses
                                .into_iter()
                                .map(|addr| format_addr(*addr, &self.allocations))
                                .join(", ")
                        )
                    }
                    AddressFormat::SingleAddress { addr, count } => {
                        write!(
                            f,
                            "{:>2}x {:>7}",
                            count,
                            format_addr(*addr, &self.allocations)
                        )
                    }
                    AddressFormat::BaseStride {
                        base,
                        stride,
                        count,
                    } => {
                        let start = format_addr(*base, &self.allocations);
                        let end = format_addr(base + stride * *count as u64, &self.allocations);
                        write!(
                            f,
                            "{:>2}x {:>7} - {:<7} stride={:<3}",
                            count, start, end, stride
                        )
                    }
                }
            }
        }

        let addresses = Addresses {
            format: address_format,
            allocations: &*self.allocations,
        };

        write!(
            f,
            "     [ block {} warp{:>3} ] inst_idx={:<4}  offset={:<4} {} [{:<10?}] {:>25} => {:<4}\t active={} adresses = {}",
            self.inner.block_id,
            self.inner.warp_id_in_block,
            self.inner.instr_idx,
            self.inner.instr_offset,
            style(format!("{:>15}", &self.inner.instr_opcode)).cyan(),
            self.inner.instr_mem_space,
            self.inner.source_registers().iter().map(|r| format!("R{}", r)).collect::<Vec<_>>().join(","),
            self.inner.dest_registers().iter().map(|r| format!("R{}", r)).collect::<Vec<_>>().join(","),
            self.inner.active_mask.to_bit_string_colored(None),
            addresses,
        )
    }
}

#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceInstruction {
    // pub enum TraceInstruction<'a> {
    Valid(InstructionComparator),
    Filtered(InstructionComparator),
}

impl AsRef<InstructionComparator> for TraceInstruction {
    // impl<'a> AsRef<InstructionComparator<'a>> for TraceInstruction<'a> {
    fn as_ref(&self) -> &InstructionComparator {
        match self {
            Self::Valid(inst) => &inst,
            Self::Filtered(inst) => &inst,
        }
    }
}

impl TraceInstruction {
    // impl<'a> TraceInstruction<'a> {
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
// struct WarpTraces(pub indexmap::IndexMap<(trace_model::Dim, u32), Vec<TraceInstruction>>);

// impl From<trace_model::WarpTraces> for WarpTraces {
//     fn from(trace: trace_model::WarpTraces) -> Self {
//         Self(
//             trace
//                 .into_iter()
//                 .map(|(k, v)| {
//                     let v = v
//                         .into_iter()
//                         .map(InstructionComparator)
//                         .map(TraceInstruction::Valid)
//                         .collect();
//                     (k, v)
//                 })
//                 .collect(),
//         )
//     }
// }

fn trace_info(
    commands_or_trace: &Path,
    mem_only: bool,
    block_id: Option<&trace_model::Dim>,
    warp_id: Option<usize>,
    verbose: bool,
    summary: bool,
    inst_limit: Option<usize>,
) -> eyre::Result<()> {
    let trace_dir = commands_or_trace.parent().unwrap();
    // let (command_traces, allocations) = parse_trace(trace_dir, commands_or_trace, mem_only)?;
    let command_traces = parse_trace(trace_dir, commands_or_trace, mem_only)?;

    #[derive(Debug, Default)]
    struct InstructionSummary {
        pub count: usize,
        pub read_addresses: Vec<u64>,
        pub write_addresses: Vec<u64>,
    }

    let mut per_instruction_summary: IndexMap<String, InstructionSummary> = IndexMap::new();

    let mut total_instructions = 0;
    for (i, (cmd, warp_traces)) in command_traces.iter().enumerate() {
        println!("command {i}: {cmd:?}");
        let Some(warp_traces) = warp_traces else {
            continue;
        };

        let mut warps: Vec<_> = warp_traces
            .0
            .keys()
            .sorted_by_key(|(warp_id, _)| warp_id.clone())
            .collect();

        if let Some(block_id) = block_id {
            warps.retain(|(trace_block_id, _)| trace_block_id == block_id)
        }
        if let Some(warp_id) = warp_id {
            warps.retain(|(_, trace_warp_id_in_block)| *trace_warp_id_in_block as usize == warp_id)
        }

        // let base_addr = warp_traces
        //     .0
        //     .values()
        //     .flat_map(|warp_trace| warp_trace)
        //     .map(|inst| inst.as_ref())
        //     .map(|inst| &inst.0)
        //     .filter(|inst| inst.is_memory_instruction())
        //     .flat_map(|inst| inst.valid_addresses())
        //     .min()
        //     .unwrap_or(0);

        for warp_id in warps.iter() {
            let (block_id, warp_id_in_block) = warp_id;
            let warp_trace: Vec<_> = get_warp_instructions(warp_traces, warp_id, mem_only); // , &allocations);
            let (valid, filtered): (Vec<_>, Vec<_>) = partition_instructions(warp_trace);

            if !summary {
                println!(
                    "\t=> block {: <10} warp={: <3}\t {} instructions ({} filtered)",
                    block_id.to_string(),
                    warp_id_in_block,
                    valid.len(),
                    filtered.len(),
                );
            }

            // let colors = [
            //     console::Style::new().red(),
            //     console::Style::new().green(),
            //     console::Style::new().yellow(),
            //     console::Style::new().blue(),
            //     console::Style::new().magenta(),
            //     console::Style::new().cyan(),
            // ];
            // let mut current_color = 0;
            // let mut last_opcode: Option<String> = None;
            for (_trace_idx, inst) in valid.iter().enumerate() {
                // if Some(&inst.0.instr_opcode) != last_opcode.as_ref() {
                //     last_opcode = Some(inst.0.instr_opcode.clone());
                //     current_color = (current_color + 1) % colors.len();
                // }
                // inst.0.instr_opcode = colors[current_color]
                //     .apply_to(inst.0.instr_opcode)
                //     .to_string();
                if verbose {
                    println!("{:#?}", inst.inner);
                } else if !summary {
                    println!("{:?}", inst);
                }
                let summary = per_instruction_summary
                    .entry(inst.inner.instr_opcode.to_string())
                    .or_default();
                summary.count += 1;
                if inst.inner.instr_is_mem && inst.inner.instr_is_load {
                    summary.read_addresses.extend(inst.inner.valid_addresses());
                }
                if inst.inner.instr_is_mem && inst.inner.instr_is_store {
                    summary.write_addresses.extend(inst.inner.valid_addresses());
                }

                total_instructions += 1;
                if let Some(inst_limit) = inst_limit {
                    if total_instructions >= inst_limit {
                        return Ok(());
                    }
                }
            }
        }
    }

    let per_instruction_summary = per_instruction_summary
        .into_iter()
        .sorted_by_key(|(_, summary)| summary.count)
        .collect::<Vec<_>>();

    println!("\n ===== SUMMARY =====\n");
    for (instruction, summary) in per_instruction_summary {
        let header = format!(
            "{:>8}x {:<15}",
            style(summary.count).cyan(),
            style(instruction).yellow()
        );
        let indent: String = (0..utils::visible_characters(&header))
            .into_iter()
            .map(|_| ' ')
            .collect();
        println!("{}", header);
        println!(
            "{}{:>5} addresses read     => unique: {:?}",
            indent,
            summary.read_addresses.len(),
            summary.read_addresses.iter().collect::<IndexSet<_>>()
        );
        println!(
            "{}{:>5} addresses written  => unique: {:?}",
            indent,
            summary.write_addresses.len(),
            summary.write_addresses.iter().collect::<IndexSet<_>>()
        );
    }
    Ok(())
}

pub type WarpId = (trace_model::Dim, u32);

fn get_warp_instructions(
    // fn get_warp_instructions<'a>(
    // warp_traces: &WarpTraces<'a>,
    warp_traces: &WarpTraces,
    warp_id: &WarpId,
    mem_only: bool,
    // allocations: &'a Allocations,
    // ) -> Vec<TraceInstruction<'a>> {
) -> Vec<TraceInstruction> {
    warp_traces
        .0
        .get(warp_id)
        .unwrap_or(&vec![])
        .iter()
        .cloned()
        .map(|inst| {
            if (mem_only && !inst.as_ref().inner.is_memory_instruction())
                || inst.as_ref().inner.active_mask.not_any()
            {
                let inst = inst.into_inner();
                TraceInstruction::Filtered(inst)
            // } else if inst.as_ref().0.active_mask.not_any() {
            //     TraceInstruction::Filtered(inst.into_inner())
            } else {
                inst
            }
        })
        .collect()
}

fn compare_trace(
    left: Option<&WarpTraces>,
    left_path: &Path,
    // left_allocations: &Allocations,
    right: Option<&WarpTraces>,
    right_path: &Path,
    // right_allocations: &Allocations,
    mem_only: bool,
    block_id: Option<&trace_model::Dim>,
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

    let mut different_warps = Vec::new();
    let mut checked_warps = Vec::new();

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
            let mut all_warps: Vec<_> = left_warps
                .iter()
                .chain(right_warps.iter())
                .copied()
                .collect::<IndexSet<_>>()
                .into_iter()
                .sorted()
                .collect();

            if let Some(block_id) = block_id {
                all_warps.retain(|(trace_block_id, _)| trace_block_id == block_id)
            }

            for warp_id in all_warps.iter() {
                let (block_id, warp_id_in_block) = warp_id;
                let left_warp_trace: Vec<_> = get_warp_instructions(left, warp_id, mem_only); // , left_allocations);
                let right_warp_trace: Vec<_> = get_warp_instructions(right, warp_id, mem_only); // , right_allocations);

                let (mut left_valid, left_filtered): (Vec<_>, Vec<_>) =
                    partition_instructions(left_warp_trace);
                let (mut right_valid, right_filtered): (Vec<_>, Vec<_>) =
                    partition_instructions(right_warp_trace);

                for trace in [&mut left_valid, &mut right_valid] {
                    for inst in trace.iter_mut() {
                        inst.inner.instr_offset = 0;
                        inst.inner.instr_idx = 0;

                        inst.inner.src_regs.fill(0);
                        inst.inner.num_src_regs = 0;

                        inst.inner.dest_regs.fill(0);
                        inst.inner.num_dest_regs = 0;

                        inst.inner.line_num = 0;
                        inst.inner.warp_id_in_sm = 0;
                        inst.inner.instr_data_width = 0;
                        inst.inner.sm_id = 0;
                        inst.inner.cuda_ctx = 0;

                        let min = inst.inner.addrs.iter().min().copied().unwrap_or(0);
                        for addr in inst.inner.addrs.iter_mut() {
                            *addr = addr.checked_sub(min).unwrap_or(0);
                        }
                    }
                }

                let left_valid_full = left_valid
                    .iter()
                    .cloned()
                    .map(InstructionComparator::into_inner)
                    .collect::<Vec<_>>();
                let right_valid_full = right_valid
                    .iter()
                    .cloned()
                    .map(InstructionComparator::into_inner)
                    .collect::<Vec<_>>();

                if left_valid_full != right_valid_full {
                    println!(
                        "\t=> block {} warp {:<3} filtered: {} (left) {} (right)",
                        block_id,
                        warp_id_in_block,
                        left_filtered.len(),
                        right_filtered.len(),
                    );
                    diff!(left_label, left_valid, right_label, right_valid);
                    diff!(left_label, left_valid_full, right_label, right_valid_full);
                    different_warps.push((block_id, warp_id_in_block));
                }

                checked_warps.push((block_id, warp_id_in_block));
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
    let style = if different_warps.is_empty() {
        console::Style::new().green()
    } else {
        console::Style::new().red()
    };
    println!(
        "different warps: {}",
        style.apply_to(format!(
            "{}/{} {:?}",
            different_warps.len(),
            checked_warps.len(),
            different_warps
        ))
    );
}

fn compare_traces(
    left_commands_or_trace: &Path,
    right_commands_or_trace: &Path,
    print_traces: bool,
    mem_only: bool,
    verbose: bool,
    summary: bool,
    block_id: Option<&trace_model::Dim>,
) -> eyre::Result<()> {
    // let (left_command_traces, left_allocations) = {
    let left_command_traces = {
        let trace_dir = left_commands_or_trace.parent().unwrap();
        let command_traces = parse_trace(trace_dir, left_commands_or_trace, mem_only)?;
        command_traces.into_iter().collect::<IndexMap<_, _>>()
        // let (command_traces, allocations) =
        //     parse_trace(trace_dir, left_commands_or_trace, mem_only)?;
        // (
        //     command_traces.into_iter().collect::<IndexMap<_, _>>(),
        //     allocations,
        // )
    };
    // let (right_command_traces, right_allocations) = {
    let right_command_traces = {
        let trace_dir = right_commands_or_trace.parent().unwrap();
        let command_traces = parse_trace(trace_dir, right_commands_or_trace, mem_only)?;
        command_traces.into_iter().collect::<IndexMap<_, _>>()
        // let (command_traces, allocations) =
        //     parse_trace(trace_dir, right_commands_or_trace, mem_only)?;
        // (
        //     command_traces.into_iter().collect::<IndexMap<_, _>>(),
        //     allocations,
        // )
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
            .iter()
            .map(|(warp_id, warp_traces)| (*warp_id, *warp_traces))
            .chain(std::iter::repeat((&None, &None)));
        let right_traces = right_kernel_launches
            .iter()
            .map(|(warp_id, warp_traces)| (*warp_id, *warp_traces))
            .chain(std::iter::repeat((&None, &None)));

        for ((i, (_left_command, left_trace)), (_right_command, right_trace)) in (0
            ..num_kernel_traces)
            .into_iter()
            .zip(left_traces)
            .zip(right_traces)
        {
            dbg!(i, left_trace.is_some(), right_trace.is_some());
            if summary {
                match (left_trace, right_trace) {
                    (Some(left_trace), Some(right_trace)) => {
                        let block_ids: IndexSet<_> = left_trace
                            .0
                            .keys()
                            .chain(right_trace.0.keys())
                            .map(|(block_id, _)| block_id)
                            .collect();
                        dbg!(block_ids.len());
                        dbg!(block_ids.iter().min());
                        dbg!(block_ids.iter().max());
                    }
                    _ => {}
                }
            } else {
                compare_trace(
                    left_trace.as_ref(),
                    left_commands_or_trace,
                    // &left_allocations,
                    right_trace.as_ref(),
                    right_commands_or_trace,
                    // &right_allocations,
                    mem_only,
                    block_id,
                );
            }
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

pub fn partition_instructions<'a, C, FC>(
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
        Command::Info {
            ref block_id,
            warp_id,
            instruction_limit,
            summary,
        } => {
            for trace in &options.traces {
                trace_info(
                    trace,
                    options.memory_only,
                    block_id.as_ref(),
                    warp_id,
                    options.verbose,
                    summary,
                    instruction_limit,
                )?;
            }
        }
        Command::Compare {
            print,
            summary,
            ref block_id,
        } => {
            if options.traces.len() != 2 {
                eyre::bail!(
                    "can only compare exactly 2 trace files, got {}",
                    options.traces.len()
                );
            }
            compare_traces(
                &options.traces[0],
                &options.traces[1],
                print,
                options.memory_only,
                options.verbose,
                summary,
                block_id.as_ref(),
            )?;
        }
    }

    Ok(())
}
