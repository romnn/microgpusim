#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("playground-sys/src/ref/bridge/bitset.hpp");

        type bitset;

        #[must_use]
        fn new_bitset() -> UniquePtr<bitset>;

        fn reset(self: Pin<&mut bitset>);
        fn set(self: Pin<&mut bitset>, pos: usize, set: bool);
        fn shift_right(self: Pin<&mut bitset>, n: usize);
        fn shift_left(self: Pin<&mut bitset>, n: usize);
        #[must_use]
        fn size(self: &bitset) -> usize;
        #[must_use]
        fn to_string(self: &bitset) -> UniquePtr<CxxString>;
    }
}

pub use ffi::*;
