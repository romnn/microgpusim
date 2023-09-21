use crate::bindings;

super::extern_type!(bindings::cache_block_state, "cache_block_state");

#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("playground-sys/src/ref/bridge/cache.hpp");

        type cache_block_state = crate::bindings::cache_block_state;

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

        type cache_block_bridge;
        #[must_use]
        unsafe fn new_cache_block_bridge(
            ptr: *const cache_block_t,
        ) -> SharedPtr<cache_block_bridge>;
        #[must_use]
        fn inner(self: &cache_block_bridge) -> *const cache_block_t;
        #[must_use]
        fn get_sector_status(self: &cache_block_bridge) -> UniquePtr<CxxVector<cache_block_state>>;
        #[must_use]
        fn get_last_sector_access_time(self: &cache_block_bridge) -> UniquePtr<CxxVector<u32>>;

        type baseline_cache;
        type cache_bridge;

        #[must_use]
        unsafe fn new_cache_bridge(ptr: *const baseline_cache) -> SharedPtr<cache_bridge>;
        #[must_use]
        fn inner(self: &cache_bridge) -> *const baseline_cache;
        #[must_use]
        fn get_lines(self: &cache_bridge) -> UniquePtr<CxxVector<cache_block_ptr>>;
    }

    // explicit instantiation for cache_block_state to implement VecElement
    impl CxxVector<cache_block_state> {}
}

pub use ffi::*;
