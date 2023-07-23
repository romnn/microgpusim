#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("playground-sys/src/ref/register_set.hpp");

        type register_set;
        fn get_name(self: &register_set) -> *const c_char;
    }
}

pub use ffi::*;
