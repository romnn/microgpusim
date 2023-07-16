#![allow(warnings)]

pub mod addrdec;
pub mod interconnect;

pub use playground_sys::{bindings, main, mem_fetch, stats, trace_shd_warp};

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
#[repr(transparent)]
pub struct MemFetch<'a> {
    // ptr: *const playground_sys::main::mem_fetch_bridge,
    ptr: *const playground_sys::mem_fetch::mem_fetch,
    phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a> AsRef<playground_sys::mem_fetch::mem_fetch> for MemFetch<'a> {
    fn as_ref(&self) -> &'a playground_sys::mem_fetch::mem_fetch {
        unsafe { &*self.ptr as &_ }
    }
}

// impl<'a> std::fmt::Display for MemFetch<'a> {
//     fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
//         if self.is_reply() {
//             write!(f, "Reply")?
//         } else {
//             write!(f, "Req")?
//         }
//         let addr = self.get_addr();
//         let relative_addr = self.get_relative_addr();
//         let access_type = self.get_access_type();
//         if addr == relative_addr {
//             write!(f, "({:?}@{})", access_type, addr)
//         } else {
//             let alloc_id = self.get_alloc_id();
//             let alloc_id = self.get_alloc_id();
//             write!(f, "({:?}@{}+{})", access_type, alloc_id, relative_addr)
//         }
//     }
// }

// impl<'a> MemFetch<'a> {
// pub fn as_ref(&self) -> &playground_sys::main::mem_fetch_bridge {
//     unsafe { &*self.ptr as &_ }
// }

// pub fn mem_fetch(&self) -> &playground_sys::mem_fetch::mem_fetch {
//     unsafe { &*self.ptr as &_ }
// }
// }

// impl std::ops::Deref
impl<'a> std::ops::Deref for MemFetch<'a> {
    type Target = playground_sys::mem_fetch::mem_fetch;

    fn deref(&self) -> &'a Self::Target {
        unsafe { &*self.ptr as &_ }
        // unsafe { &*self.as_ref().get_mem_fetch() as &_ }
        // &unsafe { *self.ptr.as_ref().unwrap().get_mem_fetch().cast_const() }
        // &unsafe { *self.ptr.as_ref().unwrap().get_mem_fetch().cast_const() }
    }
}

// pub struct MemFetch(*mut playground_sys::mem_fetch::mem_fetch);
// pub struct MemFetch<'a>(&'a playground_sys::main::mem_fetch_bridge);

#[derive()]
#[repr(transparent)]
pub struct MemorySubPartition<'a>(&'a playground_sys::main::memory_sub_partition_bridge);

impl<'a> MemorySubPartition<'a> {
    #[must_use]
    pub fn interconn_to_l2_queue(&self) -> Vec<MemFetch<'a>> {
        let queue = self.0.get_icnt_L2_queue();
        queue
            .into_iter()
            .map(|fetch| MemFetch {
                ptr: fetch.get_mem_fetch(),
                phantom: std::marker::PhantomData,
            })
            .collect()
    }
}

#[derive()]
pub struct Accelsim<'a> {
    // pub struct Accelsim {
    inner: cxx::UniquePtr<playground_sys::main::accelsim_bridge>,
    stats: crate::stats::Stats,
    phantom: std::marker::PhantomData<&'a playground_sys::main::accelsim_bridge>,
    // pub sub_partitions: Vec<MemorySubPartition<'a>>,
}

impl<'a> Accelsim<'a> {
    // impl Accelsim {
    pub fn new(config: &Config, args: &[&str]) -> Result<Self, Error> {
        let exe = std::env::current_exe()?;
        let mut ffi_argv: Vec<&str> = vec![exe.as_os_str().to_str().unwrap()];
        ffi_argv.extend(args);

        let mut accelsim_bridge =
            playground_sys::main::new_accelsim_bridge(config.0, ffi_argv.as_slice());

        // let sub_partitions = accelsim_bridge
        //     .get_sub_partitions_vec()
        //     .into_iter()
        //     .map(MemorySubPartition)
        //     .collect();

        Ok(Self {
            inner: accelsim_bridge,
            stats: crate::stats::Stats::default(),
            phantom: std::marker::PhantomData,
            // sub_partitions,
        })
    }

    // todo
    // pub fn sub_partitions(&mut self) -> &Vec<MemorySubPartitionShim> {
    // pub fn sub_partitions(&mut self) -> impl Iterator<Item = &memory_sub_partition_bridge> {
    // pub fn sub_partitions(&'a self) -> impl Iterator<Item = MemorySubPartition<'a>> + '_ {
    //     self.inner
    //         .get_sub_partitions_vec()
    //         .iter()
    //         .map(MemorySubPartition)
    //     // .map(|bridge| MemorySubPartition(*bridge.get()))
    // }

    pub fn sub_partitions(&'a self) -> impl Iterator<Item = MemorySubPartition<'a>> + '_ {
        self.inner
            .get_sub_partitions_vec()
            .iter()
            .map(MemorySubPartition)
    }

    // works
    // pub fn sub_partitions(
    //     &'a self,
    // ) -> impl Iterator<Item = &'a playground_sys::main::memory_sub_partition_bridge> + '_ {
    //     self.inner.get_sub_partitions_vec().iter()
    // }

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
