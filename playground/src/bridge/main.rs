use std::io::Write;

use crate::bindings;

super::extern_type!(bindings::accelsim_config, "accelsim_config");

#[cxx::bridge]
mod default {
    unsafe extern "C++" {
        include!("playground/src/ref/bridge/main.hpp");

        type accelsim_config = crate::bindings::accelsim_config;

        type accelsim_bridge;

        #[must_use]
        fn new_accelsim_bridge(
            config: accelsim_config,
            argv: &[&str],
        ) -> UniquePtr<accelsim_bridge>;
        fn run_to_completion(self: Pin<&mut accelsim_bridge>);
        fn process_commands(self: Pin<&mut accelsim_bridge>);
        fn launch_kernels(self: Pin<&mut accelsim_bridge>);
        fn cycle(self: Pin<&mut accelsim_bridge>);
        fn cleanup_finished_kernel(self: Pin<&mut accelsim_bridge>, kernel_uid: u32);
        fn get_finished_kernel_uid(self: Pin<&mut accelsim_bridge>) -> u32;

        fn active(self: &accelsim_bridge) -> bool;
        fn limit_reached(self: &accelsim_bridge) -> bool;
        fn commands_left(self: &accelsim_bridge) -> bool;
        fn active_kernels(self: &accelsim_bridge) -> bool;
        fn kernels_left(self: &accelsim_bridge) -> bool;

        // iterate over sub partitions
        fn sub_partitions(self: &accelsim_bridge) -> bool;
        // iterate over queues

        // NOTE: stat transfer functions defined in stats.cc bridge
    }
}

pub(super) use default::{accelsim_bridge, accelsim_config};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("accelsim playground exited with code {0}")]
    ExitCode(i32),
}

#[derive()]
pub struct Config(default::accelsim_config);

impl Default for Config {
    fn default() -> Self {
        Self(default::accelsim_config { test: 0 })
    }
}

#[derive()]
pub struct Accelsim {
    inner: cxx::UniquePtr<default::accelsim_bridge>,
    stats: super::Stats,
}

impl Accelsim {
    pub fn new(config: &Config, args: &[&str]) -> Result<Self, Error> {
        let exe = std::env::current_exe()?;
        let mut ffi_argv: Vec<&str> = vec![exe.as_os_str().to_str().unwrap()];
        ffi_argv.extend(args);

        let mut accelsim_bridge = default::new_accelsim_bridge(config.0, ffi_argv.as_slice());

        Ok(Self {
            inner: accelsim_bridge,
            stats: super::Stats::default(),
        })
    }

    pub fn run_to_completion(&mut self) {
        self.inner.pin_mut().run_to_completion();
    }

    #[must_use]
    pub fn stats(&mut self) -> &super::Stats {
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
                std::io::stdout().flush();
                break;
            }
        }
    }
}

pub fn run(config: &Config, args: &[&str]) -> Result<super::Stats, Error> {
    let mut accelsim = Accelsim::new(config, args)?;
    accelsim.run_to_completion();
    Ok(accelsim.stats().clone())
}
