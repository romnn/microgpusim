#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("playground-sys/src/ref/bridge/cache.hpp");

        type cache_block_t;
        #[must_use]
        fn get_tag(self: &cache_block_t) -> u64;
        #[must_use]
        fn get_block_addr(self: &cache_block_t) -> u64;
        #[must_use]
        fn get_last_access_time(self: &cache_block_t) -> u64;

        #[must_use]
        fn is_invalid_line(self: &cache_block_t) -> bool;
        #[must_use]
        fn is_valid_line(self: &cache_block_t) -> bool;
        #[must_use]
        fn is_reserved_line(self: &cache_block_t) -> bool;
        #[must_use]
        fn is_modified_line(self: &cache_block_t) -> bool;

        type cache_block_ptr;
        #[must_use]
        fn get(self: &cache_block_ptr) -> *const cache_block_t;

        type baseline_cache;
        type cache_bridge;

        #[must_use]
        unsafe fn new_cache_bridge(ptr: *const baseline_cache) -> SharedPtr<cache_bridge>;
        #[must_use]
        fn inner(self: &cache_bridge) -> *const baseline_cache;
        #[must_use]
        fn get_lines(self: &cache_bridge) -> UniquePtr<CxxVector<cache_block_ptr>>;
    }
}

pub use ffi::*;
