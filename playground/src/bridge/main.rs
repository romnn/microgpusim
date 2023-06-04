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

        // unsafe fn accelsim(argc: i32, argv: *const *mut c_char) -> i32;
        // fn accelsim(config: accelsim_config) -> i32;

        // `&str` is not nul terminated on the C++ side, using `String` gives us `.c_str()`
        fn accelsim(config: accelsim_config, argv: &[&str]) -> i32;
        // fn accelsim(config: accelsim_config, argv: &[&str]) -> i32;
        // fn accelsim(config: accelsim_config, argv: &CxxVector<CxxString>) -> i32;
        // fn accelsim(config: accelsim_config, argv: &CxxString<CxxString>) -> i32;
    }
}

pub use default::*;
