#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::missing_safety_doc
)]
// #![allow(warnings)]

pub mod addrdec;
pub mod bitset;
pub mod cache;
pub mod cluster;
pub mod collector_unit;
pub mod core;
pub mod interconnect;
pub mod mem_fetch;
pub mod memory_paritition_unit;
pub mod memory_sub_partition;
pub mod operand_collector;
pub mod port;
pub mod register_set;
pub mod scheduler_unit;
pub mod vec;
pub mod warp;
pub mod warp_inst;

pub use playground_sys::{bindings, bridge::types, main, stats};

use self::cluster::Cluster;
use self::core::Core;
use memory_paritition_unit::MemoryPartitionUnit;
use memory_sub_partition::MemorySubPartition;
use std::marker::PhantomData;

#[must_use]
pub fn is_debug() -> bool {
    playground_sys::is_debug()
    // #[cfg(feature = "debug_build")]
    // let is_debug = true;
    // #[cfg(not(feature = "debug_build"))]
    // let is_debug = false;
    // is_debug
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    FFI(#[from] std::ffi::NulError),

    #[error("accelsim playground exited with code {0}")]
    ExitCode(i32),
}

#[derive(Clone)]
#[derive(Default)]
pub struct Config {
    pub print_stats: bool,
    pub accelsim_compat_mode: bool,
    pub stats_file: Option<std::ffi::CString>,
}



impl Config {
    fn to_accelsim_config(&self) -> playground_sys::main::accelsim_config {
        use std::ffi::c_char;
        let stats_file: *const c_char = match self.stats_file {
            Some(ref stats_file) => stats_file.as_c_str().as_ptr(),
            None => std::ptr::null(),
        };
        playground_sys::main::accelsim_config {
            print_stats: self.print_stats,
            accelsim_compat_mode: self.accelsim_compat_mode,
            stats_file,
        }
    }
}

#[derive()]
pub struct Accelsim<'a> {
    inner: cxx::UniquePtr<playground_sys::main::accelsim_bridge>,
    stats: playground_sys::stats::Stats,
    #[allow(dead_code)]
    config: Config,
    phantom: PhantomData<&'a playground_sys::main::accelsim_bridge>,
}

impl<'a> Accelsim<'a> {
    pub fn new(config: Config, args: &[&str]) -> Result<Self, Error> {
        let exe = std::env::current_exe()?;
        let mut ffi_argv: Vec<&str> = vec![exe.as_os_str().to_str().unwrap()];
        ffi_argv.extend(args);

        let accelsim_bridge = playground_sys::main::new_accelsim_bridge(
            config.to_accelsim_config(),
            ffi_argv.as_slice(),
        );

        let num_cores = accelsim_bridge.get_cores().len();
        let num_sub_partitions = accelsim_bridge.get_sub_partitions().len();
        Ok(Self {
            inner: accelsim_bridge,
            stats: crate::stats::Stats::new(num_cores, num_sub_partitions),
            // we must keep config alive for the entire duration of this accelsim instance
            config,
            phantom: PhantomData,
        })
    }

    pub fn sub_partitions(&'a self) -> impl Iterator<Item = MemorySubPartition<'a>> + '_ {
        self.inner
            .get_sub_partitions()
            .iter()
            .map(MemorySubPartition)
    }

    pub fn partition_units(&'a self) -> impl Iterator<Item = MemoryPartitionUnit<'a>> + '_ {
        self.inner
            .get_partition_units()
            .iter()
            .map(MemoryPartitionUnit)
    }

    pub fn cores(&'a self) -> impl Iterator<Item = Core<'a>> + '_ {
        self.inner.get_cores().iter().map(Core)
    }

    pub fn clusters(&'a self) -> impl Iterator<Item = Cluster<'a>> + '_ {
        self.inner.get_clusters().iter().map(Cluster)
    }

    pub fn run_to_completion(&mut self) {
        self.inner.pin_mut().run_to_completion();
    }

    #[must_use]
    pub fn stats(&mut self) -> &crate::stats::Stats {
        self.inner.transfer_stats(&mut self.stats);
        &self.stats
    }

    #[must_use]
    pub fn commands_left(&self) -> bool {
        self.inner.commands_left()
    }

    #[must_use]
    pub fn kernels_left(&self) -> bool {
        self.inner.kernels_left()
    }

    pub fn process_commands(&mut self) {
        self.inner.pin_mut().process_commands();
    }

    pub fn launch_kernels(&mut self) {
        self.inner.pin_mut().launch_kernels();
    }

    #[must_use]
    pub fn active(&self) -> bool {
        self.inner.active()
    }

    pub fn cycle(&mut self) {
        self.inner.pin_mut().cycle();
    }

    #[must_use]
    pub fn get_cycle(&self) -> u64 {
        self.inner.get_cycle()
    }

    #[must_use]
    pub fn finished_kernel_uid(&mut self) -> Option<u32> {
        match self.inner.pin_mut().get_finished_kernel_uid() {
            0 => None,
            valid_uid => Some(valid_uid),
        }
    }

    pub fn cleanup_finished_kernel(&mut self, uid: u32) {
        self.inner.pin_mut().cleanup_finished_kernel(uid);
    }

    #[must_use]
    pub fn limit_reached(&self) -> bool {
        self.inner.limit_reached()
    }

    #[must_use]
    pub fn last_cluster_issue(&self) -> u32 {
        self.inner.get_last_cluster_issue()
    }

    pub fn custom_run_to_completion(&mut self) {
        use std::io::Write;
        while self.commands_left() || self.kernels_left() {
            self.process_commands();
            self.launch_kernels();

            let mut finished_kernel_uid: Option<u32> = None;
            loop {
                if !self.active() {
                    break;
                }
                self.cycle();

                finished_kernel_uid = self.finished_kernel_uid();
                if finished_kernel_uid.is_some() {
                    break;
                }
            }

            if let Some(uid) = finished_kernel_uid {
                self.cleanup_finished_kernel(uid);
            }

            if self.limit_reached() {
                println!(
                    "GPGPU-Sim: ** break due to reaching the maximum cycles (or instructions) **"
                );
                break;
            }
        }
        let _ = std::io::stdout().flush();
    }
}

pub fn run(config: Config, args: &[&str]) -> Result<crate::stats::Stats, Error> {
    let mut accelsim = Accelsim::new(config, args)?;
    accelsim.run_to_completion();
    Ok(accelsim.stats().clone())
}
