use crate::bindings;

crate::bridge::extern_type!(bindings::addrdec_t, "addrdec_t");
crate::bridge::extern_type!(
    bindings::linear_to_raw_address_translation_params,
    "linear_to_raw_address_translation_params"
);

#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("playground-sys/src/bindings.hpp");

        type addrdec_t = crate::bindings::addrdec_t;

        type linear_to_raw_address_translation;
        type linear_to_raw_address_translation_params =
            crate::bindings::linear_to_raw_address_translation_params;

        #[must_use]
        fn new_address_translation(
            params: linear_to_raw_address_translation_params,
        ) -> UniquePtr<linear_to_raw_address_translation>;
        fn configure(self: Pin<&mut linear_to_raw_address_translation>);
        fn init(
            self: Pin<&mut linear_to_raw_address_translation>,
            n_channel: u32,
            n_sub_partition_in_channel: u32,
        );

        unsafe fn addrdec_tlx(
            self: &linear_to_raw_address_translation,
            addr: u64,
            tlx: *mut addrdec_t,
        );
        #[must_use]
        fn partition_address(self: &linear_to_raw_address_translation, addr: u64) -> u64;

        #[must_use]
        fn powli(x: i64, y: i64) -> i64;
        #[must_use]
        fn LOGB2_32(v: u32) -> u32;
        #[must_use]
        fn next_powerOf2(n: u32) -> u32;
        #[must_use]
        fn addrdec_packbits(mask: u64, val: u64, high: u8, low: u8) -> u64;
        unsafe fn addrdec_getmasklimit(mask: u64, high: *mut u8, low: *mut u8);
    }
}

pub use ffi::*;
