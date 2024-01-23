#![allow(
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::cast_possible_truncation
)]

pub mod alloc;
pub mod cfg;
pub mod kernel;
pub mod model;
pub mod tracegen;

pub use exec_derive::instrument_control_flow;
pub use kernel::{Kernel, ThreadBlock, ThreadIndex};
pub use model::MemorySpace;

use color_eyre::eyre;
use std::path::Path;

/// Convert multi-dimensional index into flat linear index.
///
/// Users may override this to provide complex index transformations.
pub trait ToLinear {
    fn to_linear(&self) -> usize;
}

/// Simple linear index.
impl ToLinear for usize {
    fn to_linear(&self) -> usize {
        *self
    }
}

pub fn write_traces(
    mut commands: Vec<trace_model::Command>,
    kernel_traces: Vec<(
        trace_model::command::KernelLaunch,
        trace_model::MemAccessTrace,
    )>,
    traces_dir: &Path,
) -> eyre::Result<()> {
    use serde::Serialize;
    utils::fs::create_dirs(&traces_dir).map_err(eyre::Report::from)?;
    for command in commands.iter_mut() {
        if let trace_model::command::Command::KernelLaunch(kernel_launch) = command {
            let kernel_trace_name = format!("kernel-{}.msgpack", kernel_launch.id);
            let kernel_trace_path = traces_dir.join(&kernel_trace_name);

            let mut writer =
                utils::fs::open_writable(&kernel_trace_path).map_err(eyre::Report::from)?;
            let (kernel_launch_config, kernel_trace) = &kernel_traces[kernel_launch.id as usize];
            assert_eq!(kernel_launch_config, kernel_launch);
            rmp_serde::encode::write(&mut writer, &kernel_trace).map_err(eyre::Report::from)?;
            // update the kernel trace path
            kernel_launch.trace_file = kernel_trace_name;

            log::info!("written {} {}", &command, kernel_trace_path.display());
        }
    }

    let commands_path = traces_dir.join("commands.json");
    let mut json_serializer = serde_json::Serializer::with_formatter(
        utils::fs::open_writable(&commands_path).map_err(eyre::Report::from)?,
        serde_json::ser::PrettyFormatter::with_indent(b"    "),
    );
    commands
        .serialize(&mut json_serializer)
        .map_err(eyre::Report::from)?;
    log::info!("written {}", commands_path.display());
    Ok(())
}

#[cfg(test)]
pub mod tests {
    static INIT: std::sync::Once = std::sync::Once::new();

    pub fn init_test() {
        INIT.call_once(|| {
            env_logger::builder().is_test(true).init();
            color_eyre::install().unwrap();
        });
    }
}
