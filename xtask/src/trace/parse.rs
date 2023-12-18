use color_eyre::{eyre, Section, SectionExt};
use gpucachesim::allocation::Allocations;
use itertools::Itertools;
use once_cell::sync::Lazy;
use regex::Regex;
use std::path::Path;
use std::rc::Rc;

static ACCELSIM_KERNEL_TRACE_FILE_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"kernel-(\d+).*").unwrap());

fn parse_accelsim_traces(
    trace_dir: &Path,
    commands_or_trace: &Path,
    mem_only: bool,
) -> eyre::Result<accelsim::tracegen::reader::CommandTraces> {
    use accelsim::tracegen::reader::Command as AccelsimCommand;
    let mut command_traces = if commands_or_trace
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        == Some("traceg")
    {
        let extract_kernel_launch_id = || {
            // extract kernel launch id from file name similar to "kernel-1.traceg"
            let file_name = commands_or_trace
                .file_stem()
                .map(std::ffi::OsStr::to_str)
                .flatten()?;
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
        accelsim::tracegen::reader::read_command_traces(trace_dir, commands_or_trace, mem_only)
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
            ..
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
) -> eyre::Result<(
    accelsim::tracegen::reader::CommandTraces,
    Option<Allocations>,
)> {
    if commands_or_trace
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        == Some("json")
    {
        let commands = parse_box_commands(commands_or_trace)?;
        let allocations = get_box_allocations(&commands).ok();
        let command_traces: accelsim::tracegen::reader::CommandTraces = commands
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

pub fn parse_trace(
    trace_dir: &Path,
    commands_or_trace: &Path,
    mem_only: bool,
) -> eyre::Result<super::CommandTraces> {
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
                    super::WarpTraces(
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
                                    .map(|inst| super::InstructionComparator {
                                        inner: inst,
                                        allocations: Rc::clone(&allocations),
                                    })
                                    .map(super::TraceInstruction::Valid)
                                    .collect();
                                (k, v)
                            })
                            .collect(),
                    )
                });

            (cmd.map(super::TraceCommand), warp_traces)
        })
        .collect();

    Ok(traces)
    // Ok((traces, allocations))
}
