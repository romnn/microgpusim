use super::{AddressFormat, WARP_SIZE};
use color_eyre::eyre::{self, WrapErr};
use itertools::Itertools;
use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashSet;
use std::path::Path;
use trace_model::Dim;

static LOAD_OPCODES: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        // TODO: for now, we ignore constant loads, consider it as ALU_OP
        // "LDC"
        "LD", "LDG", "LDL", "LDS", "LDSM", "LDGSTS",
    ]
    .into_iter()
    .collect()
});
static STORE_OPCODES: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    ["ST", "STG", "STL", "STS", "ATOM", "ATOMS", "ATOMG", "RED"]
        .into_iter()
        .collect()
});

#[derive(Debug, Clone)]
pub struct KernelLaunchMetadata {
    pub trace_version: usize,
    pub line_info: bool,
}

#[derive(Debug, Clone)]
pub enum Command {
    MemcpyHtoD(trace_model::MemcpyHtoD),
    KernelLaunch((trace_model::KernelLaunch, KernelLaunchMetadata)),
}

static DIM_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"\(?(\d+),(\d+),(\d+)\)?").unwrap());

pub fn parse_dim(value: &str) -> Option<Dim> {
    let dim = DIM_REGEX.captures(value)?;
    let x = dim.get(1)?.as_str().trim().parse().ok()?;
    let y = dim.get(2)?.as_str().trim().parse().ok()?;
    let z = dim.get(3)?.as_str().trim().parse().ok()?;
    Some(Dim { x, y, z })
}

#[inline]
fn missing(k: &str) -> eyre::Report {
    eyre::eyre!("missing {k}")
}

#[inline]
pub fn parse_decimal<T>(value: Option<&str>, name: &str) -> eyre::Result<T>
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::error::Error + Send + Sync + 'static,
{
    let value: &str = value.ok_or_else(|| missing(name))?.trim();
    let value = value
        .parse::<T>()
        .wrap_err_with(|| format!("bad {}: {:?}", name, value))?;
    Ok(value)
}

#[inline]
pub fn parse_hex<T>(value: Option<&str>, name: &str) -> eyre::Result<T>
where
    T: num_traits::Num,
    <T as num_traits::Num>::FromStrRadixErr: std::error::Error + Send + Sync + 'static,
{
    let value: &str = value.ok_or_else(|| missing(name))?.trim();
    let numeric_value = value.trim_start_matches("0x");
    let numeric_value = T::from_str_radix(numeric_value, 16)
        .wrap_err_with(|| format!("bad {}: {:?}", name, value))?;
    Ok(numeric_value)
}

pub fn parse_kernel_launch(
    traces_dir: impl AsRef<Path>,
    line: String,
) -> eyre::Result<(trace_model::KernelLaunch, KernelLaunchMetadata)> {
    use std::io::BufRead;

    // kernel-1.traceg
    let kernel_trace_file_name = line;

    let mut metadata = KernelLaunchMetadata {
        trace_version: 0,
        line_info: false,
    };
    let mut kernel_launch = trace_model::KernelLaunch {
        name: "".to_string(),
        trace_file: kernel_trace_file_name.clone(),
        id: 0,
        grid: Dim { x: 0, y: 0, z: 0 },
        block: Dim { x: 0, y: 0, z: 0 },
        shared_mem_bytes: 0,
        num_registers: 0,
        binary_version: 0,
        stream_id: 0,
        shared_mem_base_addr: 0,
        local_mem_base_addr: 0,
        nvbit_version: "".to_string(),
    };

    let kernel_trace_path = traces_dir.as_ref().join(&kernel_trace_file_name);
    let kernel_trace_file = std::fs::OpenOptions::new()
        .read(true)
        .open(&kernel_trace_path)?;
    let reader = std::io::BufReader::new(kernel_trace_file);

    for line in reader.lines() {
        let line = line?.trim().to_string();
        if line.is_empty() {
            continue;
        } else if line.starts_with("#") {
            // the trace format, ignore this and assume fixed format for now
            // the begin of the instruction stream
            break;
        } else if line.starts_with("-") {
            let (key, value) = line
                .split_once('=')
                .ok_or(eyre::eyre!("bad key value pair {:?}", line))?;
            let key = key.trim().trim_start_matches("-").trim();
            let key: Vec<_> = key.split(" ").map(str::trim).collect();
            let value = value.trim();
            match key.as_slice() {
                ["kernel", "name"] => {
                    kernel_launch.name = value.to_string();
                }
                ["kernel", "id"] => {
                    kernel_launch.id = value.trim().parse()?;
                }
                ["grid", "dim"] => {
                    kernel_launch.grid =
                        parse_dim(value).ok_or(eyre::eyre!("invalid dim: {:?}", value))?;
                }
                ["block", "dim"] => {
                    kernel_launch.block =
                        parse_dim(value).ok_or(eyre::eyre!("invalid dim: {:?}", value))?;
                }
                ["shmem"] => {
                    kernel_launch.shared_mem_bytes = value.trim().parse()?;
                }
                ["nregs"] => {
                    kernel_launch.num_registers = value.trim().parse()?;
                }
                ["cuda", "stream", "id"] => {
                    kernel_launch.stream_id = value.trim().parse()?;
                }
                ["binary", "version"] => {
                    kernel_launch.binary_version = value.trim().parse()?;
                }
                ["enable", "lineinfo"] => {
                    metadata.line_info = value.trim().parse::<u8>()? != 0;
                }
                ["nvbit", "version"] => {
                    kernel_launch.nvbit_version = value.to_string();
                }
                ["accelsim", "tracer", "version"] => {
                    metadata.trace_version = value.trim().parse()?;
                }
                ["shmem", "base_addr"] => {
                    kernel_launch.shared_mem_base_addr =
                        parse_hex(Some(value), "shared mem base addr")?;
                }
                ["local", "mem", "base_addr"] => {
                    kernel_launch.local_mem_base_addr =
                        parse_hex(Some(value), "local mem base addr")?;
                }
                key => eyre::bail!("unknown key: {:?}", key),
            }
        }
    }
    Ok((kernel_launch, metadata))
}

pub fn parse_memcopy_host_to_device(line: String) -> eyre::Result<Command> {
    // MemcpyHtoD,0x00007f7845700000,400
    let (_, addr, num_bytes) = line
        .split(",")
        .map(str::trim)
        .collect_tuple()
        .ok_or(eyre::eyre!("invalid memcopy command {:?}", line))?;
    let dest_device_addr = parse_hex(Some(addr), "dest device address")?;
    let num_bytes = parse_decimal(Some(num_bytes), "num bytes")?;
    Ok(Command::MemcpyHtoD(trace_model::MemcpyHtoD {
        allocation_name: None,
        dest_device_addr,
        num_bytes,
    }))
}

pub fn read_commands(
    trace_dir: impl AsRef<Path>,
    mut reader: impl std::io::BufRead,
) -> eyre::Result<Vec<Command>> {
    let mut commands = Vec::new();
    for line in reader.lines() {
        let line = line?.trim().to_string();
        if line.is_empty() {
            continue;
        } else if line.starts_with("MemcpyHtoD") {
            commands.push(parse_memcopy_host_to_device(line)?);
        } else if line.starts_with("kernel") {
            let (kernel_launch, metadata) = parse_kernel_launch(trace_dir.as_ref(), line)?;
            commands.push(Command::KernelLaunch((kernel_launch, metadata)));
        }
    }
    Ok(commands)
}

#[inline]
fn base_stride_decompress(
    addrs: &mut [u64],
    base_address: u64,
    stride: i32,
    active_mask: &super::ActiveMask,
) {
    let mut first_bit1_found = false;
    let mut last_bit1_found = false;
    let mut current_address = base_address;
    for w in 0..WARP_SIZE {
        if active_mask[w] && !first_bit1_found {
            first_bit1_found = true;
            addrs[w] = base_address;
        } else if first_bit1_found && !last_bit1_found {
            if active_mask[w] {
                current_address += stride.unsigned_abs() as u64;
                addrs[w] = current_address;
            } else {
                last_bit1_found = true;
            }
        }
    }
}

#[inline]
fn base_delta_decompress(
    addrs: &mut [u64],
    base_address: u64,
    deltas: Vec<i64>,
    active_mask: &super::ActiveMask,
) {
    let mut first_bit1_found = false;
    let mut last_address = 0;
    let mut delta_index = 0;
    for w in 0..WARP_SIZE {
        if active_mask[w] && !first_bit1_found {
            addrs[w] = base_address;
            first_bit1_found = true;
            last_address = base_address;
        } else if active_mask[w] && first_bit1_found {
            assert!(delta_index < deltas.len());
            addrs[w] = (last_address as i64 + deltas[delta_index]) as u64;
            delta_index += 1;
            last_address = addrs[w];
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct TraceInstruction {
    pub line_num: Option<u32>,
    pub pc: u32,
    pub active_mask: u32,
    pub opcode: String,
    pub mem_width: u32,
    pub mem_addresses: [u64; WARP_SIZE],
    pub dest_regs: Vec<u32>,
    pub src_regs: Vec<u32>,
}

#[inline]
pub fn parse_trace_instruction(
    line: &[&str],
    trace_version: usize,
    line_info: bool,
) -> eyre::Result<TraceInstruction> {
    assert!(
        trace_version >= 0,
        "trace version {} not supported",
        trace_version
    );
    let mut values: std::collections::VecDeque<&str> = line.iter().copied().collect();

    let line_num: Option<u32> = if line_info {
        Some(parse_decimal(values.pop_front(), "line num")?)
    } else {
        None
    };

    let pc: u32 = parse_hex(values.pop_front(), "pc")?;
    let raw_active_mask: u32 = parse_hex(values.pop_front(), "active mask")?;
    let active_mask = super::parse_active_mask(raw_active_mask);

    let num_dest_regs: usize = parse_decimal(values.pop_front(), "num dest regs")?;
    assert!(
        num_dest_regs < 20,
        "astronomical number of source registers"
    );
    let mut dest_regs: Vec<u32> = (0..num_dest_regs)
        .map(|r| {
            let dest_reg = values
                .pop_front()
                .map(str::trim)
                .map(|r| r.trim_start_matches("R"));

            parse_decimal(dest_reg, &format!("dest reg {r}"))
        })
        .try_collect()?;

    let opcode = values.pop_front().ok_or(missing("opcode"))?.to_string();

    let num_src_regs: usize = parse_decimal(values.pop_front(), "num src regs")?;
    assert!(num_src_regs < 20, "astronomical number of source registers");
    let mut src_regs: Vec<u32> = (0..num_src_regs)
        .map(|r| {
            let src_reg = values
                .pop_front()
                .map(str::trim)
                .map(|r| r.trim_start_matches("R"));
            parse_decimal(src_reg, &format!("src reg {r}"))
        })
        .try_collect()?;

    let mem_width: u32 = parse_decimal(values.pop_front(), "mem width")?;

    let mut mem_addresses: [u64; WARP_SIZE] = [0; WARP_SIZE];

    // parse addresses
    if mem_width > 0 {
        let width = super::get_data_width_from_opcode(&opcode)?;

        let address_format: usize = parse_decimal(values.pop_front(), "mem address format")?;
        let address_format = AddressFormat::from_repr(address_format)
            .ok_or_else(|| eyre::eyre!("unknown mem address format: {:?}", address_format))?;

        let num_addresses = values.len();
        let num_active_threads = active_mask.count_ones();

        if let AddressFormat::ListAll | AddressFormat::BaseDelta = address_format {
            if num_addresses != num_active_threads {
                eyre::bail!(
                    "have {} addresses for {} active threads in warp",
                    num_addresses,
                    num_active_threads
                );
            }
        }

        match address_format {
            AddressFormat::ListAll => {
                // read addresses one by one from the file
                for w in 0..WARP_SIZE {
                    if active_mask[w] {
                        mem_addresses[w] =
                            parse_hex(values.pop_front(), &format!("address #{}", w))?;
                    }
                }
            }
            AddressFormat::BaseStride => {
                // read addresses as base address and stride
                let base_address: u64 = parse_hex(values.pop_front(), "base address")?;
                let stride: i32 = parse_decimal(values.pop_front(), "stride")?;
                base_stride_decompress(&mut mem_addresses, base_address, stride, &active_mask);
            }
            AddressFormat::BaseDelta => {
                // read addresses as base address and deltas
                let base_address: u64 = parse_hex(values.pop_front(), "base address")?;
                let deltas: Vec<i64> = values
                    .into_iter()
                    .map(|delta| parse_decimal(Some(delta), "delta"))
                    .try_collect()?;

                // 1 base address + 31 deltas
                assert!(deltas.len() < WARP_SIZE);
                base_delta_decompress(&mut mem_addresses, base_address, deltas, &active_mask);
            }
        }
    }

    Ok(TraceInstruction {
        line_num,
        pc,
        active_mask: raw_active_mask,
        opcode,
        mem_width,
        mem_addresses,
        dest_regs,
        src_regs,
    })
}

#[inline]
pub fn parse_instruction(
    trace_instruction: TraceInstruction,
    block_id: Dim,
    warp_id: u32,
    kernel: &trace_model::KernelLaunch,
) -> eyre::Result<Option<trace_model::MemAccessTraceEntry>> {
    let opcode_tokens: Vec<_> = trace_instruction.opcode.split(".").collect();
    assert!(!opcode_tokens.is_empty());
    let opcode1 = opcode_tokens[0].to_uppercase();

    let instr_is_mem = trace_instruction.mem_width > 0;
    let instr_is_extended = opcode_tokens.contains(&"E");

    let instr_is_load = instr_is_mem && LOAD_OPCODES.contains(opcode1.as_str());
    let instr_is_store = instr_is_mem && STORE_OPCODES.contains(opcode1.as_str());

    let instr_mem_space = match opcode1.as_str() {
        "EXIT" => nvbit_model::MemorySpace::None,
        "LDC" => nvbit_model::MemorySpace::Constant, // cannot store constants
        "LDG" | "STG" => nvbit_model::MemorySpace::Global,
        "LDL" | "STL" => nvbit_model::MemorySpace::Local,
        "LDS" | "STS" => nvbit_model::MemorySpace::Shared,
        opcode @ "LDSM" => panic!("do not know how to handle opcode {}", opcode),
        opcode if instr_is_mem => panic!("unknown opcode {}", opcode),
        _ => return Ok(None), // skip non memory instruction
    };

    let mut dest_regs = [0; 1];
    let num_dest_regs = trace_instruction.dest_regs.len() as u32;
    for (i, reg) in trace_instruction.dest_regs.into_iter().enumerate() {
        dest_regs[i] = reg;
    }
    let mut src_regs = [0; 5];
    let num_src_regs = trace_instruction.src_regs.len() as u32;
    for (i, reg) in trace_instruction.src_regs.into_iter().enumerate() {
        src_regs[i] = reg;
    }

    Ok(Some(trace_model::MemAccessTraceEntry {
        cuda_ctx: 0, // cannot infer that (not required)
        sm_id: 0,    // cannot infer that (not required)
        kernel_id: kernel.id,
        block_id,
        warp_id_in_sm: warp_id, // accelsim does not record warp_id_in_sm (not required)
        warp_id_in_block: warp_id,
        warp_size: WARP_SIZE as u32,
        line_num: trace_instruction.line_num.unwrap_or(0),
        instr_data_width: 0, // cannot infer that (not required)
        instr_opcode: trace_instruction.opcode,
        instr_offset: trace_instruction.pc,
        instr_idx: 0, // we cannot recover that (not required)
        // cannot infer predicate (not required)
        instr_predicate: nvbit_model::Predicate {
            num: 0,
            is_neg: false,
            is_uniform: false,
        },
        instr_mem_space,
        instr_is_mem,
        instr_is_load,
        instr_is_store,
        instr_is_extended,
        dest_regs,
        num_dest_regs,
        src_regs,
        num_src_regs,
        active_mask: trace_instruction.active_mask,
        addrs: trace_instruction.mem_addresses,
    }))
}

pub fn read_trace_instructions(
    reader: impl std::io::BufRead,
    trace_version: usize,
    line_info: bool,
    kernel: &trace_model::KernelLaunch,
) -> eyre::Result<Vec<trace_model::MemAccessTraceEntry>> {
    let mut instructions = Vec::new();
    let mut lines = reader.lines();
    for line in &mut lines {
        if line?.trim().starts_with("#") {
            // begin of instruction stream
            break;
        }
    }

    let mut block = Dim::ZERO;
    let mut start_of_tb_stream_found = false;

    let mut warp_id = 0;
    let mut insts_num = 0;
    let mut inst_count = 0;

    for line in lines {
        let line = line?.trim().to_string();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(" ").map(str::trim).collect();
        match parts.as_slice() {
            ["#BEGIN_TB"] => {
                assert!(!start_of_tb_stream_found);
                start_of_tb_stream_found = true;
            }
            ["#END_TB"] => {
                assert!(start_of_tb_stream_found);
                start_of_tb_stream_found = false;
            }
            ["thread", "block", ..] => {
                assert!(start_of_tb_stream_found);
                let err = || eyre::eyre!("bad key value pair {:?}", line);
                let (_, value) = line.split_once('=').ok_or_else(err)?;

                block = parse_dim(value)
                    .ok_or(eyre::eyre!("invalid dim: {:?}", value))
                    .wrap_err_with(err)?;
            }
            ["warp", ..] => {
                assert!(start_of_tb_stream_found);
                let err = || eyre::eyre!("bad key value pair {:?}", line);
                let (_, value) = line.split_once('=').ok_or_else(err)?;

                warp_id = value.trim().parse().wrap_err_with(err)?;
            }
            ["insts", ..] => {
                assert!(start_of_tb_stream_found);
                let err = || eyre::eyre!("bad key value pair {:?}", line);
                let (_, value) = line.split_once('=').ok_or_else(err)?;

                insts_num = value.trim().parse().wrap_err_with(err)?;
            }
            instruction => {
                let trace_instruction =
                    parse_trace_instruction(instruction, trace_version, line_info)
                        .wrap_err_with(|| format!("bad instruction: {:?}", instruction))?;
                if let Some(parsed_instruction) =
                    parse_instruction(trace_instruction.clone(), block.clone(), warp_id, kernel)
                        .wrap_err_with(|| format!("bad instruction: {:?}", trace_instruction))?
                {
                    instructions.push(parsed_instruction);
                }
            }
        }
    }
    // sort instructions like the accelsim tracer
    instructions.sort_by_key(|inst| {
        let block_sort_key = trace_model::dim::accelsim_block_id(&inst.block_id, &kernel.grid);
        (block_sort_key, inst.warp_id_in_block)
    });
    Ok(instructions)
}

// pub fn accelsim_tracer_block_sort_key(block_id: &Dim, grid: &Dim) -> u64 {
//     let block_x = block_id.x as u64;
//     let block_y = block_id.y as u64;
//     let block_z = block_id.z as u64;
//
//     let grid_x = grid.x as u64;
//     let grid_y = grid.y as u64;
//     let grid_z = grid.z as u64;
//
//     // tb_id = tb_id_z * grid_dim_y * grid_dim_x + tb_id_y * grid_dim_x + tb_id_x;
//     block_z * grid_y * grid_x + block_y * grid_x + block_x
// }
//
// pub fn accelsim_tracer_sort_key(inst: &trace_model::MemAccessTraceEntry, grid: &Dim) -> (u64, u32) {
//     let block_sort_key = accelsim_tracer_block_sort_key(&inst.block_id, grid);
//     (block_sort_key, inst.warp_id_in_block)
// }

#[cfg(test)]
mod tests {
    use color_eyre::eyre;
    use similar_asserts as diff;
    use std::path::{Path, PathBuf};
    use trace_model::Dim;

    fn open_file(path: &Path) -> std::io::Result<std::io::BufReader<std::fs::File>> {
        let file = std::fs::OpenOptions::new().read(true).open(path)?;
        let reader = std::io::BufReader::new(file);
        Ok(reader)
    }

    #[test]
    fn test_read_store_local_mem_instruction_multiple_addresses() -> eyre::Result<()> {
        let line: Vec<_> = "02d8 ffffffff 1 R10 LDG.E 1 R10 4 1 0x7f0e5f717000 4"
            .split(" ")
            .map(str::trim)
            .collect();
        let have = super::parse_trace_instruction(&line, 4, false)?;
        let mut mem_addresses = [0x7f0e5f717000; super::WARP_SIZE];
        for (i, base) in mem_addresses.iter_mut().enumerate() {
            *base += (i as u64) * 4;
        }

        let want = super::TraceInstruction {
            line_num: None,
            pc: 0x02d8,
            active_mask: 0xFFFFFFFF,
            opcode: "LDG.E".to_string(),
            mem_width: 4,
            mem_addresses,
            dest_regs: vec![10],
            src_regs: vec![10],
        };
        diff::assert_eq!(have: have, want: want);
        Ok(())
    }

    #[test]
    fn test_read_store_local_mem_instruction_single_address() -> eyre::Result<()> {
        let line: Vec<_> = "00a8 ffffffff 0 STL.64 2 R1 R10 8 1 0xfffcd0 0"
            .split(" ")
            .map(str::trim)
            .collect();
        let have = super::parse_trace_instruction(&line, 4, false)?;
        let want = super::TraceInstruction {
            line_num: None,
            pc: 0xa8,
            active_mask: 0xFFFFFFFF,
            opcode: "STL.64".to_string(),
            mem_width: 8,
            mem_addresses: [0xfffcd0; super::WARP_SIZE],
            dest_regs: vec![],
            src_regs: vec![1, 10],
        };
        diff::assert_eq!(have: have, want: want);
        Ok(())
    }

    #[test]
    fn test_read_commands() -> eyre::Result<()> {
        // let commands = indoc::indoc! {r#"
        //     MemcpyHtoD,0x00007f7845700000,400
        //     MemcpyHtoD,0x00007f7845700200,400
        //     MemcpyHtoD,0x00007f7845700400,400
        //     kernel-1.traceg"#
        // };

        let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
        let trace_dir = manifest_dir.join("../results/vectorAdd/vectorAdd-100-32/accelsim-trace");
        let kernelslist = trace_dir.join("kernelslist.g");

        let reader = open_file(&kernelslist)?;
        let commands = super::read_commands(trace_dir, reader)?;
        println!("{:#?}", &commands);
        Ok(())
    }

    #[test]
    fn test_read_trace_instructions() -> eyre::Result<()> {
        let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
        let trace_dir = manifest_dir.join("../results/vectorAdd/vectorAdd-100-32/accelsim-trace");
        let kernelslist = trace_dir.join("kernelslist.g");

        let reader = open_file(&kernelslist)?;
        let commands = super::read_commands(&trace_dir, reader)?;

        let (kernel, metadata) = commands
            .iter()
            .filter_map(|cmd| match cmd {
                super::Command::KernelLaunch(kernel) => Some(kernel),
                _ => None,
            })
            .next()
            .unwrap();

        let kernel_trace_path = trace_dir.join(&kernel.trace_file);
        let reader = open_file(&kernel_trace_path)?;
        let trace = super::read_trace_instructions(
            reader,
            metadata.trace_version,
            metadata.line_info,
            &kernel,
        )?;
        dbg!(&trace[..5]);
        dbg!(trace.len());
        Ok(())
    }
}
