#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::missing_safety_doc
)]
#![allow(warnings)]

pub mod addrdec;
pub mod interconnect;

pub use playground_sys::{bindings, main, mem_fetch, stats, trace_shd_warp};

use std::marker::PhantomData;
use std::ops::Deref;

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

#[derive(Clone)]
pub struct MemFetch<'a> {
    inner: cxx::SharedPtr<playground_sys::main::mem_fetch_bridge>,
    phantom: PhantomData<&'a playground_sys::main::mem_fetch_bridge>,
}

// impl<'a> AsRef<playground_sys::mem_fetch::mem_fetch> for MemFetch<'a> {
//     fn as_ref(&self) -> &'a playground_sys::mem_fetch::mem_fetch {
//         unsafe { &*self.ptr as &_ }
//     }
// }

impl<'a> std::ops::Deref for MemFetch<'a> {
    type Target = playground_sys::mem_fetch::mem_fetch;

    fn deref(&self) -> &'a Self::Target {
        unsafe { self.inner.inner().as_ref().unwrap() }
    }
}

fn get_mem_fetches<'a>(
    queue: &cxx::UniquePtr<cxx::CxxVector<playground_sys::main::mem_fetch_ptr>>,
) -> Vec<MemFetch<'a>> {
    use playground_sys::main::new_mem_fetch_bridge;
    queue
        .into_iter()
        .map(|ptr| MemFetch {
            inner: unsafe { new_mem_fetch_bridge(ptr.get()) },
            phantom: PhantomData,
        })
        .collect()
}

#[derive()]
pub struct MemoryPartitionUnit<'a>(&'a playground_sys::main::memory_partition_unit_bridge);

impl<'a> MemoryPartitionUnit<'a> {
    #[must_use]
    pub fn dram_latency_queue(&self) -> Vec<MemFetch<'a>> {
        get_mem_fetches(&self.0.get_dram_latency_queue())
    }
}

#[derive()]
pub struct MemorySubPartition<'a>(&'a playground_sys::main::memory_sub_partition_bridge);

impl<'a> MemorySubPartition<'a> {
    #[must_use]
    pub fn interconn_to_l2_queue(&self) -> Vec<MemFetch<'a>> {
        get_mem_fetches(&self.0.get_icnt_L2_queue())
    }
    #[must_use]
    pub fn l2_to_interconn_queue(&self) -> Vec<MemFetch<'a>> {
        get_mem_fetches(&self.0.get_L2_icnt_queue())
    }
    #[must_use]
    pub fn dram_to_l2_queue(&self) -> Vec<MemFetch<'a>> {
        get_mem_fetches(&self.0.get_dram_L2_queue())
    }
    #[must_use]
    pub fn l2_to_dram_queue(&self) -> Vec<MemFetch<'a>> {
        get_mem_fetches(&self.0.get_L2_dram_queue())
    }
}

#[derive(Clone)]
pub struct WarpInstr<'a> {
    inner: cxx::SharedPtr<playground_sys::main::warp_inst_bridge>,
    phantom: PhantomData<&'a playground_sys::main::warp_inst_bridge>,
}

impl<'a> WarpInstr<'a> {
    pub fn opcode_str(&self) -> &str {
        let opcode = self.deref().opcode_str();
        let opcode = unsafe { std::ffi::CStr::from_ptr(opcode) };
        opcode.to_str().unwrap()
    }
}

impl<'a> std::ops::Deref for WarpInstr<'a> {
    type Target = playground_sys::main::warp_inst_t;

    fn deref(&self) -> &'a Self::Target {
        unsafe { self.inner.inner().as_ref().unwrap() }
    }
}

#[derive()]
pub struct RegisterSet<'a> {
    pub inner: cxx::SharedPtr<playground_sys::main::register_set_bridge>,
    phantom: PhantomData<&'a playground_sys::main::register_set_bridge>,
}

impl<'a> RegisterSet<'a> {
    pub fn name(&self) -> String {
        let name = unsafe { std::ffi::CStr::from_ptr(self.get_name()) };
        name.to_string_lossy().to_string()
    }

    pub fn registers(&self) -> Vec<WarpInstr<'a>> {
        use playground_sys::main::new_warp_inst_bridge;
        self.inner
            .get_registers()
            .iter()
            .map(|ptr| WarpInstr {
                inner: unsafe { new_warp_inst_bridge(ptr.get()) },
                phantom: PhantomData,
            })
            .collect()
    }
}

impl<'a> std::ops::Deref for RegisterSet<'a> {
    type Target = playground_sys::main::register_set;

    fn deref(&self) -> &'a Self::Target {
        // unsafe { &*self.inner.inner() as &_ }
        unsafe { self.inner.inner().as_ref().unwrap() }
    }
}

#[derive()]
pub struct Port<'a> {
    // ptr: cxx::UniquePtr<playground_sys::main::input_port_bridge>,
    pub inner: cxx::SharedPtr<playground_sys::main::input_port_bridge>,
    phantom: PhantomData<&'a playground_sys::main::input_port_t>,
}

fn get_register_sets<'a>(
    regs: cxx::UniquePtr<::cxx::CxxVector<playground_sys::main::register_set_ptr>>,
) -> Vec<RegisterSet<'a>> {
    use playground_sys::main::new_register_set_bridge;
    regs.iter()
        .map(|ptr| RegisterSet {
            inner: unsafe { new_register_set_bridge(ptr.get()) },
            phantom: PhantomData,
        })
        .collect()
}

impl<'a> Port<'a> {
    pub fn cu_sets(&'a self) -> impl Iterator<Item = &u32> {
        self.inner.get_cu_sets().iter()
    }

    pub fn in_ports(&'a self) -> Vec<RegisterSet<'a>> {
        get_register_sets(self.inner.get_in_ports())
    }

    pub fn out_ports(&'a self) -> Vec<RegisterSet<'a>> {
        get_register_sets(self.inner.get_out_ports())
    }
}

#[derive()]
pub struct OperandCollector<'a> {
    inner: cxx::SharedPtr<playground_sys::main::operand_collector_bridge>,
    phantom: PhantomData<&'a playground_sys::main::opndcoll_rfu_t>,
}

impl<'a> std::ops::Deref for OperandCollector<'a> {
    type Target = playground_sys::main::opndcoll_rfu_t;

    fn deref(&self) -> &'a Self::Target {
        unsafe { self.inner.inner().as_ref().unwrap() }
    }
}

impl<'a> OperandCollector<'a> {
    pub fn ports(&'a self) -> Vec<Port<'a>> {
        use playground_sys::main::new_input_port_bridge;
        self.inner
            .get_input_ports()
            .iter()
            .map(|port| Port {
                inner: unsafe { new_input_port_bridge(port) },
                phantom: PhantomData,
            })
            .collect()
    }
}

#[derive()]
pub struct Core<'a>(&'a playground_sys::main::core_bridge);

impl<'a> Core<'a> {
    #[must_use]
    pub fn register_sets(&self) -> Vec<RegisterSet<'a>> {
        use playground_sys::main::new_register_set_bridge;
        self.0
            .get_register_sets()
            .iter()
            .map(|ptr| RegisterSet {
                inner: unsafe { new_register_set_bridge(ptr.get()) },
                phantom: PhantomData,
            })
            .collect()
    }

    #[must_use]
    pub fn operand_collector(&self) -> OperandCollector<'a> {
        OperandCollector {
            inner: self.0.get_operand_collector(),
            phantom: PhantomData,
        }
    }
}

#[derive()]
pub struct Accelsim<'a> {
    inner: cxx::UniquePtr<playground_sys::main::accelsim_bridge>,
    stats: crate::stats::Stats,
    phantom: PhantomData<&'a playground_sys::main::accelsim_bridge>,
}

impl<'a> Accelsim<'a> {
    pub fn new(config: &Config, args: &[&str]) -> Result<Self, Error> {
        let exe = std::env::current_exe()?;
        let mut ffi_argv: Vec<&str> = vec![exe.as_os_str().to_str().unwrap()];
        ffi_argv.extend(args);

        let mut accelsim_bridge =
            playground_sys::main::new_accelsim_bridge(config.0, ffi_argv.as_slice());

        Ok(Self {
            inner: accelsim_bridge,
            stats: crate::stats::Stats::default(),
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
