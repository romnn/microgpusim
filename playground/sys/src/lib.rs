#![allow(warnings)]

pub mod bridge;
pub mod stats;

#[allow(
    warnings,
    clippy::all,
    clippy::pedantic,
    clippy::restriction,
    clippy::nursery
)]
pub mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub use bridge::{
    addrdec, cache_config, interconnect, main, mem_fetch, readonly_cache, scheduler_unit,
    trace_shd_warp,
};
