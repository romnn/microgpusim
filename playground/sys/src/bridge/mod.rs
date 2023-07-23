#![allow(
    clippy::missing_errors_doc,
    clippy::missing_safety_doc,
    clippy::missing_panics_doc
)]

pub mod core;
pub mod input_port;
pub mod main;
pub mod mem_fetch;
pub mod memory_partition_unit;
pub mod operand_collector;
pub mod register_set;
pub mod scheduler_unit;
pub mod stats;
pub mod types;
pub mod warp_inst;

macro_rules! extern_type {
    ($type:ty, $id:literal) => {
        unsafe impl cxx::ExternType for $type {
            type Id = cxx::type_id!($id);
            type Kind = cxx::kind::Trivial;
        }
    };
}

pub(crate) use extern_type;
