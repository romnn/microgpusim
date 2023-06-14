use crate::bindings;
use cxx::{type_id, ExternType};

unsafe impl ExternType for bindings::accelsim_config {
    type Id = type_id!("accelsim_config");
    type Kind = cxx::kind::Trivial;
}

#[derive(Debug, Default)]
pub struct AccelsimStats {}

impl AccelsimStats {
    pub fn set_l2_cache_stat(&mut self, test: u32) {}
}

#[cxx::bridge]
mod default {
    extern "Rust" {
        type AccelsimStats;
        fn set_l2_cache_stat(self: &mut AccelsimStats, test: u32);
    }

    unsafe extern "C++" {
        include!("playground/src/ref/bridge/main.hpp");

        type accelsim_config = crate::bindings::accelsim_config;
        // type accelsim_stats; //= crate::bindings::accelsim_stats;

        // fn accelsim(config: accelsim_config, argv: &[&str]) -> i32;
        // fn accelsim(config: accelsim_config, argv: &[&str]) -> UniquePtr<accelsim_stats>;
        fn accelsim(config: accelsim_config, argv: &[&str], stats: &mut AccelsimStats) -> i32;
    }
}

pub use default::*;
