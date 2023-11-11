use super::parse::parse_trace;
use color_eyre::eyre;
use console::style;
use indexmap::{IndexMap, IndexSet};
use itertools::Itertools;
use std::path::Path;

pub fn trace_info(
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
            let warp_trace: Vec<_> = super::get_warp_instructions(warp_traces, warp_id, mem_only); // , &allocations);
            let (valid, filtered): (Vec<_>, Vec<_>) = super::partition_instructions(warp_trace);

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
