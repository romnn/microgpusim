use crate::bindings;
use cxx::{type_id, ExternType};

unsafe impl ExternType for bindings::accelsim_config {
    type Id = type_id!("accelsim_config");
    type Kind = cxx::kind::Trivial;
}


#[cxx::bridge]
mod default {
    unsafe extern "C++" {
        include!("playground/src/bindings.hpp");

        type accelsim_config = crate::bindings::accelsim_config;

        // unsafe fn accelsim(argc: i32, argv: *const *mut c_char) -> i32;
        fn accelsim(config: accelsim_config) -> i32;
    }
}

pub use default::*;
