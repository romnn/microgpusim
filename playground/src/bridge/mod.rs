#![allow(
    clippy::missing_errors_doc,
    clippy::missing_safety_doc,
    clippy::missing_panics_doc
)]

pub mod addrdec;
pub mod cache_config;
pub mod interconnect;
pub mod main;
pub mod mem_fetch;
pub mod readonly_cache;
pub mod scheduler_unit;
pub mod stats;
pub mod trace_shd_warp;

macro_rules! extern_type {
    ($type:ty, $id:literal) => {
        unsafe impl cxx::ExternType for $type {
            type Id = cxx::type_id!($id);
            type Kind = cxx::kind::Trivial;
        }
    };
}

pub(crate) use extern_type;

pub use self::stats::StatsBridge as Stats;
