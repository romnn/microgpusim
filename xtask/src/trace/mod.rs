pub mod compare;
pub mod info;
pub mod metrics;
pub mod parse;

use clap::Parser;
use color_eyre::eyre;
use console::style;
use gpucachesim::allocation::Allocations;
use itertools::Itertools;
use std::path::PathBuf;
use std::rc::Rc;
use trace_model::ToBitString;

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
    Metrics {
        #[clap(
            long = "iterations",
            default_value = "10",
            help = "number of samples for deserialization rounds"
        )]
        iterations: usize,
        #[clap(long = "stat-file", help = "stat file to write statistics into")]
        stat_file: PathBuf,
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

fn format_addr(addr: u64, allocations: &Allocations) -> String {
    match allocations.get(&addr) {
        Some(allocation) => {
            let relative_addr = addr.saturating_sub(allocation.start_addr);
            format!("{}+{}", allocation.id, relative_addr)
        }
        None => format!("{}", addr),
    }
}

pub type CommandTraces = Vec<(Option<TraceCommand>, Option<WarpTraces>)>;

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
pub struct InstructionComparator {
    allocations: Rc<Allocations>,
    inner: trace_model::MemAccessTraceEntry,
}

impl InstructionComparator {
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

pub type WarpId = (trace_model::Dim, u32);

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct WarpTraces(pub indexmap::IndexMap<(trace_model::Dim, u32), Vec<TraceInstruction>>);

fn get_warp_instructions(
    warp_traces: &WarpTraces,
    warp_id: &WarpId,
    mem_only: bool,
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
            } else {
                inst
            }
        })
        .collect()
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
                info::trace_info(
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
            compare::compare_traces(
                &options.traces[0],
                &options.traces[1],
                print,
                options.memory_only,
                options.verbose,
                summary,
                block_id.as_ref(),
            )?;
        }
        Command::Metrics {
            iterations,
            ref stat_file,
        } => {
            metrics::trace_metrics(&options.traces, stat_file, iterations)?;
        }
    }

    Ok(())
}
