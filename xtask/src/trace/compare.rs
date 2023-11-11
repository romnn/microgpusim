use super::parse::parse_trace;
use color_eyre::eyre;
use console::style;
use indexmap::{IndexMap, IndexSet};
use itertools::Itertools;
use std::path::Path;
use utils::diff;

fn compare_trace(
    left: Option<&super::WarpTraces>,
    left_path: &Path,
    // left_allocations: &Allocations,
    right: Option<&super::WarpTraces>,
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
                let left_warp_trace: Vec<_> = super::get_warp_instructions(left, warp_id, mem_only); // , left_allocations);
                let right_warp_trace: Vec<_> =
                    super::get_warp_instructions(right, warp_id, mem_only); // , right_allocations);

                let (mut left_valid, left_filtered): (Vec<_>, Vec<_>) =
                    super::partition_instructions(left_warp_trace);
                let (mut right_valid, right_filtered): (Vec<_>, Vec<_>) =
                    super::partition_instructions(right_warp_trace);

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
                    .map(super::InstructionComparator::into_inner)
                    .collect::<Vec<_>>();
                let right_valid_full = right_valid
                    .iter()
                    .cloned()
                    .map(super::InstructionComparator::into_inner)
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

pub fn compare_traces(
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
            cmd.as_ref()
                .is_some_and(super::TraceCommand::is_kernel_launch)
                || traces.is_some()
        })
        .collect();
    let right_kernel_launches: IndexMap<_, _> = right_command_traces
        .iter()
        .filter(|(cmd, traces)| {
            cmd.as_ref()
                .is_some_and(super::TraceCommand::is_kernel_launch)
                || traces.is_some()
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
