use crate::bindings;

super::extern_type!(bindings::addrdec_t, "addrdec_t");
super::extern_type!(
    bindings::linear_to_raw_address_translation_params,
    "linear_to_raw_address_translation_params"
);

#[cxx::bridge]
mod default {
    unsafe extern "C++" {
        include!("playground/src/bindings.hpp");

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
        fn print(self: &linear_to_raw_address_translation);

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

pub use default::{addrdec_t as AddrDec, next_powerOf2, powli, LOGB2_32};

#[must_use]
pub fn mask_limit(mask: u64) -> (u8, u8) {
    let mut low = 0;
    let mut high = 64;
    unsafe {
        default::addrdec_getmasklimit(
            mask,
            std::ptr::addr_of_mut!(high),
            std::ptr::addr_of_mut!(low),
        );
    }
    (low, high)
}

#[must_use]
pub fn packbits(mask: u64, val: u64, low: u8, high: u8) -> u64 {
    assert!(low <= 64);
    assert!(high <= 64);
    default::addrdec_packbits(mask, val, high, low)
}

pub struct AddressTranslation(cxx::UniquePtr<default::linear_to_raw_address_translation>);

impl std::fmt::Debug for AddressTranslation {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.0.print();
        Ok(())
    }
}

impl AddressTranslation {
    #[must_use]
    pub fn new(num_channels: u32, num_sub_partitions_per_channel: u32) -> Self {
        let params = default::linear_to_raw_address_translation_params {
            run_test: false,
            gpgpu_mem_address_mask: 1, // new address mask
            memory_partition_indexing: bindings::partition_index_function::CONSECUTIVE,
        };
        let mut inner = default::new_address_translation(params);
        // do not initialize cli options to be empty
        // inner.pin_mut().configure();
        inner
            .pin_mut()
            .init(num_channels, num_sub_partitions_per_channel);
        Self(inner)
    }

    #[must_use]
    pub fn partition_address(&self, addr: u64) -> u64 {
        self.0.partition_address(addr)
    }

    #[must_use]
    pub fn tlx(&self, addr: u64) -> AddrDec {
        let mut tlx = AddrDec {
            chip: 0,
            bk: 0,
            row: 0,
            col: 0,
            burst: 0,
            sub_partition: 0,
        };

        unsafe {
            self.0.addrdec_tlx(addr, std::ptr::addr_of_mut!(tlx));
        }
        tlx
    }
}
