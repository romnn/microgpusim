// use crate::bindings;
// use cxx::{type_id, ExternType};
//
// unsafe impl ExternType for bindings::cache_config_params {
//     type Id = type_id!("cache_config_params");
//     type Kind = cxx::kind::Trivial;
// }

#[cxx::bridge]
mod default {
    unsafe extern "C++" {
        include!("playground/src/bindings.hpp");

        type scheduler_unit;
        fn new_scheduler_unit() -> UniquePtr<scheduler_unit>;

        // fn powli(x: i64, y: i64) -> i64;
        // fn LOGB2_32(v: u32) -> u32;
        // fn next_powerOf2(n: u32) -> u32;
        // fn addrdec_packbits(mask: u64, val: u64, high: u8, low: u8) -> u64;
        // unsafe fn addrdec_getmasklimit(mask: u64, high: *mut u8, low: *mut u8);
    }
}

pub use default::*;
