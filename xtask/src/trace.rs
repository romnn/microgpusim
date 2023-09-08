use accelsim::tracegen;
use clap::Parser;
use color_eyre::eyre;
use std::path::{Path, PathBuf};

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
    let command_traces = tracegen::reader::read_traces_for_commands(trace_dir, commands, mem_only)?;
    Ok(command_traces)
}

fn parse_box_traces(
    trace_dir: &Path,
    commands: &Path,
) -> eyre::Result<tracegen::reader::CommandTraces> {
    use itertools::Itertools;

    let commands: Vec<trace_model::Command> = {
        let reader = utils::fs::open_readable(commands)?;
        let commands = serde_json::from_reader(reader)?;
        commands
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

fn parse_trace(
    trace_dir: &Path,
    commands_path: &Path,
    mem_only: bool,
) -> eyre::Result<tracegen::reader::CommandTraces> {
    let commands_traces = parse_box_traces(trace_dir, commands_path)
        .or_else(|_err| parse_accelsim_traces(trace_dir, commands_path, mem_only))?;

    Ok(commands_traces)
}

fn trace_info(commands: &Path) -> eyre::Result<()> {
    let trace_dir = commands.parent().unwrap();
    let mem_only = false;
    let command_traces = parse_trace(trace_dir, commands, mem_only)?;
    // dbg!(command_traces);

    for (i, (cmd, traces)) in command_traces.into_iter().enumerate() {
        println!("command {i}: {:?}", cmd);
        let Some(traces) = traces else {
            continue;
        };
        let warp_traces = traces.to_warp_traces();
        dbg!(&warp_traces[&(trace_model::Dim::ZERO, 0)]
            .iter()
            .map(|entry| (&entry.instr_opcode, &entry.active_mask))
            .collect::<Vec<_>>());
    }
    Ok(())
}

fn compare_traces(
    left_commands_path: &Path,
    right_commands_path: &Path,
    print_traces: bool,
) -> eyre::Result<()> {
    use bitvec::field::BitField;
    use gpucachesim::mem_fetch::ToBitString;
    use itertools::Itertools;
    use std::collections::{HashMap, HashSet};

    let mem_only = false;
    let left_command_traces: HashMap<_, _> = {
        let trace_dir = left_commands_path.parent().unwrap();
        let command_traces = parse_trace(trace_dir, left_commands_path, mem_only)?;
        command_traces.into_iter().collect()
    };
    let right_command_traces: HashMap<_, _> = {
        let trace_dir = left_commands_path.parent().unwrap();
        let command_traces = parse_trace(trace_dir, left_commands_path, mem_only)?;
        command_traces.into_iter().collect()
    };

    let left_commands: HashSet<_> = left_command_traces.keys().collect();
    let right_commands: HashSet<_> = right_command_traces.keys().collect();
    let all_commands: Vec<_> = left_commands.union(&right_commands).sorted().collect();

    let empty = trace_model::MemAccessTrace::default();
    for (_cmd_idx, cmd) in all_commands.into_iter().enumerate() {
        let left_trace = left_command_traces[cmd].as_ref().unwrap_or(&empty);
        let right_trace = right_command_traces[cmd].as_ref().unwrap_or(&empty);
        println!("===> command {:?}", cmd);

        if left_trace != right_trace {
            utils::diff::diff!(left: &left_trace, right: &right_trace);
        } else if print_traces {
            // print matching trace
            let warp_traces = left_trace.clone().to_warp_traces();
            for ((block_id, warp_id), trace) in warp_traces.iter() {
                println!(
                    "#### block={:<10} warp={:<2}",
                    block_id.to_string(),
                    warp_id
                );
                for (_trace_idx, entry) in trace.iter().enumerate() {
                    let mut active_mask = gpucachesim::warp::ActiveMask::ZERO;
                    active_mask.store(entry.active_mask);
                    println!(
                        "     [ block {} warp{:>3} ]\t inst_idx={:<4}  offset={:<4}\t {:<20}\t\t active={}",
                        entry.block_id.to_string(),
                        entry.warp_id_in_block,
                        entry.instr_idx,
                        entry.instr_offset,
                        entry.instr_opcode,
                        active_mask.to_bit_string(),
                    );
                }
            }
        }
        println!("");
    }

    Ok(())
}

pub fn run(options: Options) -> eyre::Result<()> {
    match options.command {
        Command::Info => {
            for trace in options.traces.iter() {
                trace_info(&trace)?;
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
