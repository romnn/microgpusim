// #![allow(warnings)]

pub mod reader;
pub mod writer;

pub const WARP_SIZE: u32 = 32;

#[derive(strum::FromRepr, Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum AddressFormat {
    ListAll = 0,
    BaseStride = 1,
    BaseDelta = 2,
}

type ActiveMask = bitvec::BitArr!(for 32, in u32);

fn parse_active_mask(raw_mask: u32) -> ActiveMask {
    use bitvec::field::BitField;
    let mut active_mask = bitvec::array::BitArray::ZERO;
    active_mask.store(raw_mask);
    active_mask
}

fn is_number(s: &str) -> bool {
    !s.is_empty() && s.chars().all(char::is_numeric)
}

fn get_data_width_from_opcode(opcode: &str) -> Result<u32, std::num::ParseIntError> {
    let opcode_tokens: Vec<_> = opcode
        .split('.')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .collect();

    for token in opcode_tokens {
        assert!(!token.is_empty());

        if is_number(token) {
            return Ok(token.parse::<u32>()? / 8);
        } else if let Some('U') = token.chars().next() {
            if is_number(&token[1..token.len()]) {
                // handle the U* case
                return Ok(token[1..token.len()].parse::<u32>()? / 8);
            }
        }
    }
    // default is 4 bytes
    Ok(4)
}

use color_eyre::eyre;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub struct Conversion<'a> {
    pub native_commands_path: &'a Path,
    pub box_traces_dir: &'a Path,
    pub accelsim_traces_dir: &'a Path,
}

pub fn convert_accelsim_to_box_traces(options: &Conversion<'_>) -> eyre::Result<PathBuf> {
    use itertools::Itertools;
    use reader::Command as AccelsimCommand;
    use serde::Serialize;

    let Conversion {
        native_commands_path,
        box_traces_dir,
        accelsim_traces_dir,
    } = options;
    assert!(native_commands_path.is_file());
    let generated_box_commands_path = box_traces_dir.join("accelsim.commands.json");
    println!(
        "generating commands {}",
        generated_box_commands_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
    );

    let reader = utils::fs::open_readable(native_commands_path)?;
    let accelsim_commands = reader::read_commands(accelsim_traces_dir, reader)?;

    let commands: Vec<_> = accelsim_commands
        .into_iter()
        .map(|cmd| match cmd {
            AccelsimCommand::MemcpyHtoD(memcopy) => {
                Ok::<_, eyre::Report>(trace_model::Command::MemcpyHtoD(memcopy))
            }
            AccelsimCommand::KernelLaunch((mut kernel, metadata)) => {
                // transform kernel instruction trace
                let kernel_trace_path = accelsim_traces_dir.join(&kernel.trace_file);
                let reader = utils::fs::open_readable(kernel_trace_path)?;
                let mem_only = false;
                let parsed_trace = reader::read_trace_instructions(
                    reader,
                    metadata.trace_version,
                    metadata.line_info,
                    mem_only,
                    &kernel,
                )?;

                let generated_kernel_trace_name = format!("accelsim-kernel-{}.msgpack", kernel.id);
                let generated_kernel_trace_path = box_traces_dir.join(&generated_kernel_trace_name);

                let mut writer = utils::fs::open_writable(&generated_kernel_trace_path)?;
                rmp_serde::encode::write(&mut writer, &parsed_trace)?;

                // also save as json for inspection
                let mut writer =
                    utils::fs::open_writable(generated_kernel_trace_path.with_extension("json"))?;
                serde_json::to_writer_pretty(&mut writer, &parsed_trace)?;

                // update the kernel trace path
                kernel.trace_file = generated_kernel_trace_name;

                Ok::<_, eyre::Report>(trace_model::Command::KernelLaunch(kernel))
            }
        })
        .try_collect()?;

    let mut json_serializer = serde_json::Serializer::with_formatter(
        utils::fs::open_writable(&generated_box_commands_path)?,
        serde_json::ser::PrettyFormatter::with_indent(b"    "),
    );
    commands.serialize(&mut json_serializer)?;
    Ok(generated_box_commands_path)
}

pub fn convert_box_to_accelsim_traces(options: &Conversion<'_>) -> eyre::Result<PathBuf> {
    use trace_model::Command;
    let Conversion {
        native_commands_path,
        box_traces_dir,
        accelsim_traces_dir,
    } = options;
    assert!(native_commands_path.is_file());
    let generated_kernelslist_path = accelsim_traces_dir.join("box-kernelslist.g");
    println!(
        "generating commands {}",
        generated_kernelslist_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
    );
    let mut commands_writer = utils::fs::open_writable(&generated_kernelslist_path)?;
    writer::generate_commands(native_commands_path, &mut commands_writer)?;
    drop(commands_writer);

    let reader = utils::fs::open_readable(native_commands_path)?;
    let commands: Vec<Command> = serde_json::from_reader(reader)?;

    for cmd in commands {
        if let Command::KernelLaunch(kernel) = cmd {
            // generate trace for kernel
            let generated_kernel_trace_path =
                accelsim_traces_dir.join(format!("kernel-{}.box.traceg", kernel.id + 1));
            println!(
                "generating trace {} for kernel {}",
                generated_kernel_trace_path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy(),
                kernel.id
            );
            let mut trace_writer = utils::fs::open_writable(generated_kernel_trace_path)?;
            writer::generate_trace(box_traces_dir, &kernel, &mut trace_writer)?;
        }
    }
    Ok(generated_kernelslist_path)
}
