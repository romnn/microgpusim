use crate::bindings;
use std::collections::HashMap;

super::extern_type!(bindings::accelsim_config, "accelsim_config");

pub type RequestStatus = bindings::cache_request_status;
pub type AccessType = bindings::mem_access_type;
pub type ReservationFailure = bindings::cache_reservation_fail_reason;

#[derive(Debug, Default)]
pub struct CacheStats {
    pub accesses: HashMap<(AccessType, AccessStat), u64>,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum AccessStat {
    ReservationFailure(ReservationFailure),
    Status(RequestStatus),
}

#[derive(Debug, Default)]
pub struct Stats {
    // memory accesses
    pub num_mem_write: usize,
    pub num_mem_read: usize,
    pub num_mem_const: usize,
    pub num_mem_texture: usize,
    pub num_mem_read_global: usize,
    pub num_mem_write_global: usize,
    pub num_mem_read_local: usize,
    pub num_mem_write_local: usize,
    // instructions
    pub num_load_instructions: usize,
    pub num_store_instructions: usize,
    pub num_shared_mem_instructions: usize,
    pub num_sstarr_instructions: usize,
    pub num_texture_instructions: usize,
    pub num_const_instructions: usize,
    pub num_param_instructions: usize,
    // other stuff
    pub num_mem_l2_writeback: usize,
    pub num_mem_l1_write_allocate: usize,
    pub num_mem_l2_write_allocate: usize,

    // per cache stats
    pub l1i_stats: HashMap<usize, CacheStats>,
    pub l1c_stats: HashMap<usize, CacheStats>,
    pub l1t_stats: HashMap<usize, CacheStats>,
    pub l1d_stats: HashMap<usize, CacheStats>,
    pub l2d_stats: HashMap<usize, CacheStats>,
    // todo: dram stats
}

impl Stats {
    pub fn add_accesses(
        &mut self,
        cache_kind: CacheKind,
        cache_index: usize,
        kind: u32,
        status: u32,
        failed: bool,
        num_accesses: u64,
    ) {
        let cache = match cache_kind {
            CacheKind::L1T => &mut self.l1t_stats,
            CacheKind::L1I => &mut self.l1i_stats,
            CacheKind::L1C => &mut self.l1c_stats,
            CacheKind::L1D => &mut self.l1d_stats,
            CacheKind::L2D => &mut self.l2d_stats,
            _ => {
                return;
            }
        };

        let kind: AccessType = unsafe { std::mem::transmute(kind) };
        let status = if failed {
            AccessStat::ReservationFailure(unsafe { std::mem::transmute(status) })
        } else {
            AccessStat::Status(unsafe { std::mem::transmute(status) })
        };
        let cache = cache.entry(cache_index).or_default();
        *cache.accesses.entry((kind, status)).or_insert(0) += num_accesses;
    }

    // memory accesses
    pub fn set_num_mem_write(&mut self, v: usize) {
        self.num_mem_write = v;
    }
    pub fn set_num_mem_read(&mut self, v: usize) {
        self.num_mem_read = v;
    }
    pub fn set_num_mem_const(&mut self, v: usize) {
        self.num_mem_const = v;
    }
    pub fn set_num_mem_texture(&mut self, v: usize) {
        self.num_mem_texture = v;
    }
    pub fn set_num_mem_read_global(&mut self, v: usize) {
        self.num_mem_read_global = v;
    }
    pub fn set_num_mem_write_global(&mut self, v: usize) {
        self.num_mem_write_global = v;
    }
    pub fn set_num_mem_read_local(&mut self, v: usize) {
        self.num_mem_read_local = v;
    }
    pub fn set_num_mem_write_local(&mut self, v: usize) {
        self.num_mem_write_local = v;
    }

    // instructions
    pub fn set_num_load_instructions(&mut self, v: usize) {
        self.num_load_instructions = v;
    }
    pub fn set_num_store_instructions(&mut self, v: usize) {
        self.num_store_instructions = v;
    }
    pub fn set_num_shared_mem_instructions(&mut self, v: usize) {
        self.num_shared_mem_instructions = v;
    }
    pub fn set_num_sstarr_instructions(&mut self, v: usize) {
        self.num_sstarr_instructions = v;
    }
    pub fn set_num_texture_instructions(&mut self, v: usize) {
        self.num_texture_instructions = v;
    }
    pub fn set_num_const_instructions(&mut self, v: usize) {
        self.num_const_instructions = v;
    }
    pub fn set_num_param_instructions(&mut self, v: usize) {
        self.num_param_instructions = v;
    }

    // other stuff
    pub fn set_num_mem_l2_writeback(&mut self, v: usize) {
        self.num_mem_l2_writeback = v;
    }
}

#[cxx::bridge]
mod default {
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
    pub enum CacheKind {
        // l1 caches
        L1T,
        L1I,
        L1C,
        L1D,
        // l2 cache
        L2D,
    }

    extern "Rust" {

        type Stats;

        fn add_accesses(
            self: &mut Stats,
            cache_kind: CacheKind,
            cache_index: usize,
            kind: u32,
            status: u32,
            failed: bool,
            num_accesses: u64,
        );

        // memory accesses
        fn set_num_mem_write(self: &mut Stats, v: usize);
        fn set_num_mem_read(self: &mut Stats, v: usize);
        fn set_num_mem_const(self: &mut Stats, v: usize);
        fn set_num_mem_texture(self: &mut Stats, v: usize);
        fn set_num_mem_read_global(self: &mut Stats, v: usize);
        fn set_num_mem_write_global(self: &mut Stats, v: usize);
        fn set_num_mem_read_local(self: &mut Stats, v: usize);
        fn set_num_mem_write_local(self: &mut Stats, v: usize);

        // instructions
        fn set_num_load_instructions(self: &mut Stats, v: usize);
        fn set_num_store_instructions(self: &mut Stats, v: usize);
        fn set_num_shared_mem_instructions(self: &mut Stats, v: usize);
        fn set_num_sstarr_instructions(self: &mut Stats, v: usize);
        fn set_num_texture_instructions(self: &mut Stats, v: usize);
        fn set_num_const_instructions(self: &mut Stats, v: usize);
        fn set_num_param_instructions(self: &mut Stats, v: usize);

        // fn set_num_mem_l2_writeback(self: &mut Stats, v: usize);
        // fn set_num_mem_l1_write_allocate(self: &mut Stats, v: usize);
        // fn set_num_mem_l2_write_allocate(self: &mut Stats, v: usize);
    }

    unsafe extern "C++" {
        include!("playground/src/ref/bridge/main.hpp");

        type accelsim_config = crate::bindings::accelsim_config;

        #[must_use]
        fn accelsim(config: accelsim_config, argv: &[&str], stats: &mut Stats) -> i32;
    }
}

pub use default::CacheKind;

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

pub fn run(config: &Config, args: &[&str]) -> Result<Stats, Error> {
    let exe = std::env::current_exe()?;
    let mut ffi_argv: Vec<&str> = vec![exe.as_os_str().to_str().unwrap()];
    ffi_argv.extend(args);

    let mut stats = Stats::default();
    let ret_code = default::accelsim(config.0, ffi_argv.as_slice(), &mut stats);
    if ret_code == 0 {
        Ok(stats)
    } else {
        Err(Error::ExitCode(ret_code))
    }
}
