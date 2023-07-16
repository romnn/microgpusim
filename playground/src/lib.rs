#![allow(warnings)]

pub mod addrdec;
pub mod interconnect;

pub use playground_sys::{bindings, stats, trace_shd_warp};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("accelsim playground exited with code {0}")]
    ExitCode(i32),
}

#[derive()]
pub struct Config(playground_sys::main::accelsim_config);

impl Default for Config {
    fn default() -> Self {
        Self(playground_sys::main::accelsim_config { test: 0 })
    }
}

#[derive()]
#[repr(transparent)]
// pub struct MemFetch(*mut default::mem_fetch);
pub struct MemFetch<'a>(&'a playground_sys::main::mem_fetch_bridge);

#[derive()]
#[repr(transparent)]
pub struct MemorySubPartition<'a>(&'a playground_sys::main::memory_sub_partition_bridge);
// pub struct MemorySubPartition(*mut default::memory_sub_partition_bridge);

impl<'a> MemorySubPartition<'a> {
    // #[must_use]
    // // pub fn interconn_to_l2_queue(&'a self) -> impl Iterator<Item = MemFetch<'a>> + 'a {
    // pub fn interconn_to_l2_queue(&'a self) -> Vec<MemFetch<'a>> {
    //     let queue = self.0.get_icnt_L2_queue();
    //     queue.into_iter().map(MemFetch).collect()
    //     // unsafe {
    //     // .as_ref()
    //     // .unwrap()
    //     // .iter()
    //     // .map(|bridge| MemFetch(bridge.get_mem_fetch()))
    //     // .map(MemFetch)
    //     // .collect()
    //     // }
    // }
}

#[derive()]
pub struct Accelsim {
    inner: cxx::UniquePtr<playground_sys::main::accelsim_bridge>,
    stats: crate::stats::Stats,
}

impl Accelsim {
    pub fn new(config: &Config, args: &[&str]) -> Result<Self, Error> {
        let exe = std::env::current_exe()?;
        let mut ffi_argv: Vec<&str> = vec![exe.as_os_str().to_str().unwrap()];
        ffi_argv.extend(args);

        let mut accelsim_bridge =
            playground_sys::main::new_accelsim_bridge(config.0, ffi_argv.as_slice());

        Ok(Self {
            inner: accelsim_bridge,
            stats: crate::stats::Stats::default(),
        })
    }

    // todo
    // pub fn sub_partitions(&mut self) -> &Vec<MemorySubPartitionShim> {
    // pub fn sub_partitions(&mut self) -> impl Iterator<Item = &memory_sub_partition_bridge> {
    pub fn sub_partitions<'a>(&'a mut self) -> impl Iterator<Item = MemorySubPartition<'a>> + '_ {
        self.inner
            .get_sub_partitions_vec()
            .iter()
            .map(MemorySubPartition)
        // .map(|bridge| MemorySubPartition(*bridge.get()))
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

pub fn run(config: &Config, args: &[&str]) -> Result<crate::stats::Stats, Error> {
    let mut accelsim = Accelsim::new(config, args)?;
    accelsim.run_to_completion();
    Ok(accelsim.stats().clone())
}
