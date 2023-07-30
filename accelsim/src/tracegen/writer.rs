use super::AddressFormat;
use color_eyre::eyre::{self, WrapErr};
use color_eyre::owo_colors::OwoColorize;
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use trace_model::{Command, MemAccessTraceEntry};

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
    use nvbit_model::Dim;
    let mut grouped: HashMap<Dim, HashMap<u32, Vec<&MemAccessTraceEntry>>> = HashMap::new();

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
                    let mem_width = super::get_data_width_from_opcode(&inst.instr_opcode)?;
                    line.push(mem_width.to_string());

                    // list all the addresses
                    line.push((AddressFormat::ListAll as usize).to_string());
                    let active_mask = super::parse_active_mask(inst.active_mask);
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
    let reader = utils::fs::open_readable(commands_path.as_ref())?;
    let commands: Vec<Command> = serde_json::from_reader(reader)?;
    // dbg!(&commands);

    for cmd in commands {
        match cmd {
            Command::MemAlloc(_) => {}
            Command::MemcpyHtoD(trace_model::MemcpyHtoD {
                dest_device_addr,
                num_bytes,
                ..
            }) => {
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
    let reader = utils::fs::open_readable(&trace_file_path)?;
    let mut trace: Vec<MemAccessTraceEntry> = rmp_serde::from_read(reader)
        .wrap_err_with(|| format!("failed to read trace {}", trace_file_path.display()))?;
    // dbg!(&trace[0]);

    write_trace_instructions(&trace, out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use color_eyre::eyre;
    use similar_asserts as diff;
    use std::collections::HashMap;
    use std::io::Write;
    use std::path::PathBuf;
    use trace_model::{Command, MemAccessTraceEntry};

    #[test]
    fn test_generate_commands() -> eyre::Result<()> {
        let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
        let trace_dir = manifest_dir.join("../results/vectorAdd/vectorAdd-100-32/trace");
        let commands_path = trace_dir.join("commands.json");
        let _commands = indoc::indoc! { r#"[
            {
                "MemAlloc": {
                    "allocation_name": null,
                    "device_ptr": 140030084382720,
                    "num_bytes": 400
                }
            },
            {
                "MemAlloc": {
                    "allocation_name": null,
                    "device_ptr": 140030084383232,
                    "num_bytes": 400
                }
            },
            {
                "MemAlloc": {
                    "allocation_name": null,
                    "device_ptr": 140030084383744,
                    "num_bytes": 400
                }
            },
            {
                "MemcpyHtoD": {
                    "allocation_name": null,
                    "dest_device_addr": 140030084382720,
                    "num_bytes": 400
                }
            },
            {
                "MemcpyHtoD": {
                    "allocation_name": null,
                    "dest_device_addr": 140030084383232,
                    "num_bytes": 400
                }
            },
            {
                "MemcpyHtoD": {
                    "allocation_name": null,
                    "dest_device_addr": 140030084383744,
                    "num_bytes": 400
                }
            },
            {
                "MemAlloc": {
                    "allocation_name": null,
                    "device_ptr": 140030084384256,
                    "num_bytes": 32
                }
            },
            {
                "MemcpyHtoD": {
                    "allocation_name": null,
                    "dest_device_addr": 140030084384256,
                    "num_bytes": 32
                }
            },
            {
                "MemAlloc": {
                    "allocation_name": null,
                    "device_ptr": 140030084384768,
                    "num_bytes": 32
                }
            },
            {
                "MemcpyHtoD": {
                    "allocation_name": null,
                    "dest_device_addr": 140030084384768,
                    "num_bytes": 32
                }
            },
            {
                "MemAlloc": {
                    "allocation_name": null,
                    "device_ptr": 140030084385280,
                    "num_bytes": 32
                }
            },
            {
                "MemcpyHtoD": {
                    "allocation_name": null,
                    "dest_device_addr": 140030084385280,
                    "num_bytes": 32
                }
            },
            {
                "MemAlloc": {
                    "allocation_name": null,
                    "device_ptr": 140030084385792,
                    "num_bytes": 32
                }
            },
            {
                "MemcpyHtoD": {
                    "allocation_name": null,
                    "dest_device_addr": 140030084385792,
                    "num_bytes": 32
                }
            },
            {
                "MemAlloc": {
                    "allocation_name": null,
                    "device_ptr": 140030084386304,
                    "num_bytes": 32
                }
            },
            {
                "MemcpyHtoD": {
                    "allocation_name": null,
                    "dest_device_addr": 140030084386304,
                    "num_bytes": 32
                }
            },
            {
                "KernelLaunch": {
                    "name": "void vecAdd<float>(float*, float*, float*, int)",
                    "trace_file": "kernel-0.msgpack",
                    "id": 0,
                    "grid": {
                        "x": 1,
                        "y": 1,
                        "z": 1
                    },
                    "block": {
                        "x": 1024,
                        "y": 1,
                        "z": 1
                    },
                    "shared_mem_bytes": 0,
                    "num_registers": 8,
                    "binary_version": 61,
                    "stream_id": 0,
                    "shared_mem_base_addr": 140030781685760,
                    "local_mem_base_addr": 140030748131328,
                    "nvbit_version": "1.5.5"
                }
            }
        ]"#};
        let mut commands_writer = std::io::Cursor::new(Vec::new());
        super::generate_commands(&commands_path, &mut commands_writer)?;
        let commands = String::from_utf8_lossy(&commands_writer.into_inner()).to_string();
        println!("{}", &commands);
        // diff::assert_eq!(
        //     have: commands.trim(),
        //     want: indoc::indoc! {r"
        //         MemcpyHtoD,0x007f5b4b700000,400
        //         MemcpyHtoD,0x007f5b4b700200,400
        //         MemcpyHtoD,0x007f5b4b700400,400
        //         MemcpyHtoD,0x007f5b4b700600,32
        //         MemcpyHtoD,0x007f5b4b700800,32
        //         MemcpyHtoD,0x007f5b4b700a00,32
        //         MemcpyHtoD,0x007f5b4b700c00,32
        //         MemcpyHtoD,0x007f5b4b700e00,32
        //         kernel-1.box.traceg"}
        // );
        Ok(())
    }

    #[test]
    fn test_generate_trace() -> eyre::Result<()> {
        let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
        let trace_dir = manifest_dir.join("../results/vectorAdd/vectorAdd-100-32/trace");
        let commands_path = trace_dir.join("commands.json");

        let reader = utils::fs::open_readable(&commands_path)?;
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
        // diff::assert_eq!(have: &trace, want: r"");
        Ok(())
    }
}