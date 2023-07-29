use super::{AddressFormat, WARP_SIZE};
use color_eyre::eyre::{self, WrapErr};
use nvbit_model::Dim;
use once_cell::sync::Lazy;
use regex::Regex;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct KernelLaunchMetadata {
    trace_version: usize,
    line_info: bool,
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

fn missing(k: &str) -> eyre::Report {
    eyre::eyre!("missing {k}")
}

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

pub fn parse_hex<T>(value: Option<&str>, name: &str) -> eyre::Result<T>
where
    T: num_traits::Num,
    <T as num_traits::Num>::FromStrRadixErr: std::error::Error + Send + Sync + 'static,
{
    let value: &str = value.ok_or_else(|| missing(name))?.trim();
    let value =
        T::from_str_radix(value, 16).wrap_err_with(|| format!("bad {}: {:?}", name, value))?;
    Ok(value)
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
                        u64::from_str_radix(value.trim_start_matches("0x"), 16)?;
                }
                ["local", "mem", "base_addr"] => {
                    kernel_launch.local_mem_base_addr =
                        u64::from_str_radix(value.trim_start_matches("0x"), 16)?;
                }
                key => eyre::bail!("unknown key: {:?}", key),
            }
        }
    }
    Ok((kernel_launch, metadata))
}

pub fn parse_memcopy_host_to_device(line: String) -> eyre::Result<Command> {
    use itertools::Itertools;
    // MemcpyHtoD,0x00007f7845700000,400
    let (_, addr, num_bytes) = line
        .split(",")
        .map(str::trim)
        .collect_tuple()
        .ok_or(eyre::eyre!("invalid memcopy command {:?}", line))?;
    let dest_device_addr = u64::from_str_radix(addr.trim_start_matches("0x"), 16)?;
    let num_bytes = num_bytes.trim().parse()?;
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

pub fn parse_instruction(
    line: &[&str],
    trace_version: usize,
    line_info: bool,
    block_id: Dim,
    warp_id: u32,
    kernel: &trace_model::KernelLaunch,
) -> eyre::Result<trace_model::MemAccessTraceEntry> {
    assert!(
        trace_version >= 0,
        "trace version {} not supported",
        trace_version
    );
    let mut values: std::collections::VecDeque<&str> = line.iter().copied().collect();

    let line_num: u32 = if line_info {
        // let line_num = values.pop_front().ok_or(missing("line num"))?.trim();
        // line_num
        //     .parse()
        //     .wrap_err_with(|| format!("bad line num: {:?}", line_num))?

        parse_decimal(values.pop_front(), "line num")?
    } else {
        0
    };

    // let pc = u32::from_str_radix(values.pop_front().ok_or(missing("pc"))?, 16)?;
    let pc: u32 = parse_hex(values.pop_front(), "pc")?;
    let raw_active_mask: u32 = parse_hex(values.pop_front(), "active mask")?;
    // let raw_active_mask = u32::from_str_radix(values.pop_front().ok_or(missing("mask"))?, 16)?;
    let active_mask = super::parse_active_mask(raw_active_mask);

    // let num_dest_regs = values.pop_front().ok_or(missing("dest num"))?.trim();
    // let num_dest_regs: usize = num_dest_regs
    //     .parse()
    //     .wrap_err_with(|| format!("bad dest register count: {:?}", num_dest_regs))?;
    let num_dest_regs: usize = parse_decimal(values.pop_front(), "num dest regs")?;
    assert!(num_dest_regs <= 1, "too many dest registers");
    let mut dest_regs: [u32; 1] = [0; 1];
    for r in 0..num_dest_regs {
        let dest_reg = values
            .pop_front()
            .map(str::trim)
            .map(|r| r.trim_start_matches("R"));

        // .ok_or_else(|| missing(&format!("dest reg {r}")))?
        // .trim();
        dest_regs[r] = parse_decimal(dest_reg, &format!("dest reg {r}"))?;
        // dest_regs[r] = dest_reg
        //     .trim_start_matches("R")
        //     .parse()
        //     .wrap_err_with(|| format!("bad dest register: {:?}", dest_reg))?;
    }

    let opcode = values.pop_front().ok_or(missing("opcode"))?;

    let num_src_regs: usize = parse_decimal(values.pop_front(), "num src regs")?;
    // let num_src_regs = values.pop_front().ok_or(missing("src num"))?.trim();
    // let num_src_regs: usize = num_src_regs
    //     .parse()
    //     .wrap_err_with(|| format!("bad src register count: {:?}", num_src_regs))?;
    let mut src_regs: [u32; 5] = [0; 5];
    for r in 0..num_src_regs {
        let src_reg = values
            .pop_front()
            .map(str::trim)
            .map(|r| r.trim_start_matches("R"));
        // .ok_or_else(|| missing(&format!("src reg {r}")))?
        // .trim();
        src_regs[r] = parse_decimal(src_reg, &format!("src reg {r}"))?;
        // src_regs[r] = src_reg
        //     .trim_start_matches("R")
        //     .parse()
        //     .wrap_err_with(|| format!("bad src register: {:?}", src_reg))?;
    }

    // let mem_width = values.pop_front();
    // let mem_width: u32 = mem_width
    //     .ok_or(missing("mem width"))?
    //     .trim()
    //     .parse()
    //     .wrap_err_with(|| format!("bad mem width: {:?}", mem_width))?;
    let mem_width: u32 = parse_decimal(values.pop_front(), "mem width")?;

    let mut addrs: [u64; 32] = [0; 32];

    // parse addresses
    if mem_width > 0 {
        let width = super::get_data_width_from_opcode(opcode)?;

        let address_format: usize = parse_decimal(values.pop_front(), "mem address format")?;
        let address_format = AddressFormat::from_repr(address_format)
            .ok_or_else(|| eyre::eyre!("unknown mem address format: {:?}", address_format))?;

        // let address_mode = values
        //     .pop_front()
        //     .ok_or(missing("mem address format"))?
        //     .trim();
        //
        // let address_mode = address_mode
        //     .parse()
        //     .ok()
        //     .and_then(AddressFormat::from_repr)
        //     .ok_or_else(|| eyre::eyre!("bad mem address format: {:?}", address_mode))?;

        match address_format {
            AddressFormat::ListAll => {
                // read addresses one by one from the file
                for w in 0..WARP_SIZE {
                    if active_mask[w] {
                        addrs[w] = parse_hex(values.pop_front(), &format!("address #{}", w))?;
                    }
                }
            }
            AddressFormat::BaseStride => {
                // read addresses as base address and stride
                let base_address: u64 = parse_hex(values.pop_front(), "base address")?;
                let stride: i32 = parse_decimal(values.pop_front(), "stride")?;
                // memadd_info->base_stride_decompress(base_address, stride, mask_bits);
            }
            AddressFormat::BaseDelta => {
                // read addresses as base address and deltas
                let base_address: u64 = parse_hex(values.pop_front(), "base address")?;
                // let stride: i32 = parse_decimal(values.pop_front(), "stride")?;
                let mut deltas = Vec::new();
                for w in 0..WARP_SIZE {
                    if active_mask[w] {
                        let delta: u64 =
                            parse_decimal(values.pop_front(), &format!("delta {}", w))?;
                        deltas.push(delta);
                    }
                }
                //   memadd_info->base_delta_decompress(base_address, deltas, mask_bits);
            }
        }
    }

    let instr_predicate = nvbit_model::Predicate {
        num: 0,
        is_neg: false,
        is_uniform: false,
    };
    let instr_mem_space = nvbit_model::MemorySpace::Global;
    Ok(trace_model::MemAccessTraceEntry {
        cuda_ctx: 0,
        kernel_id: kernel.id,
        block_id,
        thread_id: Dim { x: 0, y: 0, z: 0 },
        unique_thread_id: warp_id,
        global_warp_id: warp_id,
        warp_id_in_sm: warp_id,
        warp_id_in_block: warp_id,
        warp_size: WARP_SIZE as u32,
        line_num,
        instr_data_width: 0, // todo
        instr_opcode: opcode.to_string(),
        instr_offset: pc,
        instr_idx: 0, // we cannot recover that
        instr_predicate,
        instr_mem_space,
        instr_is_mem: mem_width > 0,
        instr_is_load: false,     // todo
        instr_is_store: false,    // todo
        instr_is_extended: false, // todo
        dest_regs,
        num_dest_regs: num_dest_regs as u32,
        src_regs,
        num_src_regs: num_src_regs as u32,
        active_mask: raw_active_mask,
        addrs,
    })
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

    let mut block = Dim { x: 0, y: 0, z: 0 };
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
                let parsed_instruction = parse_instruction(
                    instruction,
                    trace_version,
                    line_info,
                    block.clone(),
                    warp_id,
                    kernel,
                )
                .wrap_err_with(|| format!("bad instruction: {:?}", instruction))?;
                println!("{:?}", instruction);
                println!("{:#?}", parsed_instruction);
                break;
                instructions.push(parsed_instruction);
            }
        }
    }
    Ok(instructions)
}

#[cfg(test)]
mod tests {
    use color_eyre::eyre;
    use std::path::{Path, PathBuf};

    fn open_file(path: &Path) -> std::io::Result<std::io::BufReader<std::fs::File>> {
        let file = std::fs::OpenOptions::new().read(true).open(path)?;
        let reader = std::io::BufReader::new(file);
        Ok(reader)
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
        println!("{:#?}", &trace);
        assert!(false);
        Ok(())
    }
}
