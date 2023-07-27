#![allow(warnings)]
use bitvec::{access, array::BitArray, field::BitField, BitArr};
use color_eyre::eyre;
use color_eyre::owo_colors::OwoColorize;
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use trace_model::{Command, MemAccessTraceEntry};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum AddressFormat {
    ListAll = 0,
    BaseStride = 1,
    BaseDelta = 2,
}

fn get_data_width_from_opcode(opcode: &str) -> u32 {
    let opcode_tokens: Vec<_> = opcode
        .split(".")
        .map(|t| t.trim())
        .filter(|t| !t.is_empty())
        .collect();
    // dbg!(&opcode_tokens);
    for token in opcode_tokens {
        assert!(!token.is_empty());

        if token.chars().all(char::is_numeric) {
            return token.parse::<u32>().unwrap() / 8;
        } else if token.chars().nth(0).unwrap() == 'U'
            && token[1..token.len()].chars().all(char::is_numeric)
        {
            // handle the U* case
            return token[1..token.len()].parse::<u32>().unwrap() / 8;
        }
    }
    // default is 4 bytes
    4
}

fn write_kernel_info(
    kernel: &trace_model::KernelLaunch,
    mut out: impl std::io::Write,
) -> eyre::Result<()> {
    // dbg!(&kernel);
    //
    // -kernel name = _Z8mult_gpuIfEvPKT_S2_PS0_mmm
    writeln!(out, "-kernel name = {}", kernel.name)?;
    // -kernel id = 1
    writeln!(out, "-kernel id = {}", kernel.id + 1)?;
    // -grid dim = (4,2,1)
    writeln!(
        out,
        "-grid dim = ({},{},{})",
        kernel.grid.x, kernel.grid.y, kernel.grid.z
    )?;
    // -block dim = (32,32,1)
    writeln!(
        out,
        "-block dim = ({},{},{})",
        kernel.block.x, kernel.block.y, kernel.block.z
    )?;
    // -shmem = 0
    writeln!(out, "-shmem = {}", kernel.shared_mem_bytes)?;
    // -nregs = 31
    writeln!(out, "-nregs = {}", kernel.num_registers)?;
    // -binary version = 61
    writeln!(out, "-binary version = {}", kernel.binary_version)?;
    // -cuda stream id = 0
    writeln!(out, "-cuda stream id = {}", kernel.stream_id)?;
    // -shmem base_addr = 0x00007f0e8e000000
    writeln!(out, "-shmem base_addr = {:#x}", kernel.shared_mem_base_addr)?;
    // -local mem base_addr = 0x00007f0e8c000000
    writeln!(
        out,
        "-local mem base_addr = {:#x}",
        kernel.local_mem_base_addr
    )?;
    // -nvbit version = 1.5.5
    writeln!(out, "-nvbit version = {}", kernel.nvbit_version)?;
    // -accelsim tracer version = 4
    writeln!(out, "-accelsim tracer version = 4")?;
    // -enable lineinfo = 0
    writeln!(out, "-enable lineinfo = 0")?;
    writeln!(out, "")?;
    writeln!(out, "#traces format = [line_num] PC mask dest_num [reg_dests] opcode src_num [reg_srcs] mem_width [adrrescompress?] [mem_addresses]");

    Ok(())
}

fn write_trace_instructions(
    trace: &[MemAccessTraceEntry],
    mut out: impl std::io::Write,
) -> eyre::Result<()> {
    let mut grouped: HashMap<nvbit_model::Dim, HashMap<u32, Vec<&MemAccessTraceEntry>>> =
        HashMap::new();

    for inst in trace {
        let block = grouped.entry(inst.block_id).or_default();
        let warp = block.entry(inst.warp_id_in_block).or_default();
        warp.push(inst);
    }

    let mut blocks: Vec<_> = grouped.into_iter().collect();
    blocks.sort_by_key(|(block_id, _warps)| *block_id);

    for (block, warps) in blocks {
        assert!(!warps.is_empty());
        writeln!(out, "\n#BEGIN_TB\n")?;
        writeln!(out, "thread block = {},{},{}\n", block.x, block.y, block.z)?;

        let mut warps: Vec<_> = warps.into_iter().collect();
        warps.sort_by_key(|(warp_id, _instructions)| *warp_id);

        for (warp_id, instructions) in warps {
            assert!(!instructions.is_empty());
            writeln!(out, "")?;
            writeln!(out, "warp = {}", warp_id)?;
            writeln!(out, "insts = {}", instructions.len())?;

            for inst in instructions {
                // 0008 ffffffff 1 R1 MOV 0 0
                // 0670 ffffffff 0 STG.E 2 R2 R21 4 1 0x7f0e5f718000 4
                // PC mask dest_num [reg_dests] opcode src_num [reg_srcs] mem_width
                // [adrrescompress?] [mem_addresses]
                let mut line = vec![
                    format!("{:04x}", inst.instr_offset),
                    format!("{:08x}", inst.active_mask),
                ];
                line.push(inst.num_dest_regs.to_string());
                for r in 0..(inst.num_dest_regs as usize) {
                    line.push(format!("R{}", inst.dest_regs[r]));
                }
                line.push(inst.instr_opcode.clone());
                line.push(inst.num_src_regs.to_string());
                for r in 0..(inst.num_src_regs as usize) {
                    line.push(format!("R{}", inst.src_regs[r]));
                }
                if inst.instr_is_mem {
                    // mem width
                    line.push(get_data_width_from_opcode(&inst.instr_opcode).to_string());

                    // list all the addresses
                    line.push((AddressFormat::ListAll as usize).to_string());
                    let mut active_mask: BitArr!(for 32, in u32) = BitArray::ZERO;
                    active_mask.store(inst.active_mask);
                    for w in 0..32 {
                        if active_mask[w] {
                            line.push(format!("{:#016x}", inst.addrs[w]));
                        }
                    }
                } else {
                    // mem width is zero for non memory instructions
                    line.push("0".to_string());
                }
                writeln!(out, "{}", line.join(" "))?;
            }
        }

        writeln!(out, "\n#END_TB\n")?;
    }
    Ok(())
}

pub fn generate_commands(
    commands_path: impl AsRef<Path>,
    mut out: impl std::io::Write,
) -> eyre::Result<()> {
    let commands_file = std::fs::OpenOptions::new()
        .read(true)
        .open(commands_path.as_ref())?;
    let reader = std::io::BufReader::new(commands_file);
    let commands: Vec<Command> = serde_json::from_reader(reader)?;
    // dbg!(&commands);

    for cmd in commands {
        match cmd {
            Command::MemAlloc { .. } => {}
            Command::MemcpyHtoD {
                dest_device_addr,
                num_bytes,
                ..
            } => {
                writeln!(out, "MemcpyHtoD,{:#016x},{}", dest_device_addr, num_bytes)?;
            }
            Command::KernelLaunch(kernel) => {
                writeln!(out, "kernel-{}.box.traceg", kernel.id + 1)?;
            }
        }
    }
    writeln!(out, "")?;

    Ok(())
}

pub fn generate_trace(
    trace_dir: impl AsRef<Path>,
    kernel: &trace_model::KernelLaunch,
    mut out: impl std::io::Write,
) -> eyre::Result<()> {
    write_kernel_info(&kernel, &mut out)?;

    let trace_file_path = trace_dir.as_ref().join(&kernel.trace_file);
    let trace_file = std::fs::OpenOptions::new()
        .read(true)
        .open(trace_file_path)?;
    let reader = std::io::BufReader::new(trace_file);
    let mut trace: Vec<MemAccessTraceEntry> = rmp_serde::from_read(reader)?;
    // dbg!(&trace[0]);

    write_trace_instructions(&trace, out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use color_eyre::eyre;
    use std::collections::HashMap;
    use std::io::Write;
    use std::path::PathBuf;
    use trace_model::{Command, MemAccessTraceEntry};

    #[test]
    fn test_generate_commands() -> eyre::Result<()> {
        let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
        let trace_dir = manifest_dir.join("../results/vectorAdd/vectorAdd-100-32/trace");
        let commands_path = trace_dir.join("commands.json");

        let mut commands_writer = std::io::Cursor::new(Vec::new());
        super::generate_commands(&commands_path, &mut commands_writer)?;
        let commands = String::from_utf8_lossy(&commands_writer.into_inner()).to_string();
        println!("{}", &commands);
        Ok(())
    }

    #[test]
    fn test_generate_trace() -> eyre::Result<()> {
        let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
        let trace_dir = manifest_dir.join("../results/vectorAdd/vectorAdd-100-32/trace");
        let commands_path = trace_dir.join("commands.json");

        let commands_file = std::fs::OpenOptions::new()
            .read(true)
            .open(&commands_path)?;
        let reader = std::io::BufReader::new(commands_file);
        let commands: Vec<Command> = serde_json::from_reader(reader)?;
        dbg!(&commands);

        let kernel_launch = commands
            .iter()
            .find(|cmd| matches!(cmd, Command::KernelLaunch(_)));
        let Some(Command::KernelLaunch(kernel)) = kernel_launch else {
            panic!("no kernel launch command found");
        };

        let mut trace_writer = std::io::Cursor::new(Vec::new());
        super::generate_trace(&trace_dir, &kernel, &mut trace_writer)?;
        let trace = String::from_utf8_lossy(&trace_writer.into_inner()).to_string();
        println!("{}", &trace);
        Ok(())
    }
}
