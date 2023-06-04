use crate::bindings;
use cxx::{type_id, ExternType};

unsafe impl ExternType for bindings::accelsim_config {
    type Id = type_id!("accelsim_config");
    type Kind = cxx::kind::Trivial;
}

#[cxx::bridge]
mod default {
    unsafe extern "C++" {
        include!("playground/src/ref/bridge/main.hpp");

        type accelsim_config = crate::bindings::accelsim_config;

        fn accelsim(config: accelsim_config, argv: &[&str]) -> i32;
    }
}

pub use default::*;
