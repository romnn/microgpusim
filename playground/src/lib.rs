#![allow(warnings)]

#[allow(
    warnings,
    clippy::all,
    clippy::pedantic,
    clippy::restriction,
    clippy::nursery
)]
pub mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

#[cfg(test)]
mod tests {
    use super::bindings;
    use super::*;
    use pretty_assertions::assert_eq;
    use std::ffi;
    use std::mem::MaybeUninit;

    #[test]
    fn test_addrdec_packbits() {
        let decoded = unsafe { bindings::addrdec_packbits(0, 0, 0, 64) };
        assert_eq!(decoded, 0)
    }

    // #[test]
    // fn test_scanf_cache_config() {
    //     let config = "N:16:128:24,L:R:m:N:L,F:128:4,128:2";
    //     let config = ffi::CString::new(config).unwrap();
    //     let mut cache_config = MaybeUninit::uninit();
    //     unsafe {
    //         bindings::parse_cache_config(config.as_ptr() as *mut _, cache_config.as_mut_ptr());
    //     }
    //     let cache_config = unsafe { cache_config.assume_init() };
    //     assert_eq!(
    //         cache_config,
    //         bindings::CacheConfig {
    //             ct: 'N' as ffi::c_char,
    //             m_nset: 16,
    //             m_line_sz: 128,
    //             m_assoc: 24,
    //             rp: 'L' as ffi::c_char,
    //             wp: 'R' as ffi::c_char,
    //             ap: 'm' as ffi::c_char,
    //             wap: 'N' as ffi::c_char,
    //             sif: 'L' as ffi::c_char,
    //             mshr_type: 'F' as ffi::c_char,
    //             m_mshr_entries: 128,
    //             m_mshr_max_merge: 4,
    //             m_miss_queue_size: 128,
    //             m_result_fifo_entries: 2,
    //             m_data_port_width: 0,
    //         }
    //     );
    // }
}
