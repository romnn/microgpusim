pub use playground_sys::types::addrdec::{addrdec_t as AddrDec, next_powerOf2, powli, LOGB2_32};
use playground_sys::{bindings, types};

#[must_use]
pub fn mask_limit(mask: u64) -> (u8, u8) {
    let mut low = 0;
    let mut high = 64;
    unsafe {
        types::addrdec::addrdec_getmasklimit(
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
    types::addrdec::addrdec_packbits(mask, val, high, low)
}

pub struct AddressTranslation(cxx::UniquePtr<types::addrdec::linear_to_raw_address_translation>);

impl std::fmt::Debug for AddressTranslation {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.0.print();
        Ok(())
    }
}

impl AddressTranslation {
    #[must_use]
    pub fn new(num_channels: u32, num_sub_partitions_per_channel: u32) -> Self {
        let addrdec_option =
            "dramid@8;00000000.00000000.00000000.00000000.0000RRRR.RRRRRRRR.RBBBCCCC.BCCSSSSS";
        let addrdec_option = std::ffi::CString::new(addrdec_option).unwrap();
        let params = types::addrdec::linear_to_raw_address_translation_params {
            addrdec_option: addrdec_option.as_ptr(),
            run_test: false,
            gpgpu_mem_address_mask: 1, // new address mask
            memory_partition_indexing: bindings::partition_index_function::CONSECUTIVE,
        };
        let mut inner = types::addrdec::new_address_translation(params);
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
