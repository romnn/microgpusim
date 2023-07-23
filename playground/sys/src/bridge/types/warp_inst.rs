#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("playground-sys/src/ref/warp_instr.hpp");

        type warp_inst_t;
        #[must_use]
        fn empty(self: &warp_inst_t) -> bool;
        fn opcode_str(self: &warp_inst_t) -> *const c_char;
        fn get_pc(self: &warp_inst_t) -> u32;
        fn warp_id(self: &warp_inst_t) -> u32;
    }
}

pub use ffi::*;
