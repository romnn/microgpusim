#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("playground-sys/src/ref/bridge/cache.hpp");

        type cache_block_t;
        fn get_tag(self: &cache_block_t) -> u64;
        fn get_block_addr(self: &cache_block_t) -> u64;
        fn is_invalid_line(self: &cache_block_t) -> bool;
        fn is_valid_line(self: &cache_block_t) -> bool;
        fn is_reserved_line(self: &cache_block_t) -> bool;
        fn is_modified_line(self: &cache_block_t) -> bool;

        type cache_block_ptr;
        fn get(self: &cache_block_ptr) -> *const cache_block_t;

        type baseline_cache;
        type cache_bridge;

        unsafe fn new_cache_bridge(ptr: *const baseline_cache) -> SharedPtr<cache_bridge>;
        fn inner(self: &cache_bridge) -> *const baseline_cache;
        fn get_lines(self: &cache_bridge) -> UniquePtr<CxxVector<cache_block_ptr>>;
    }

    // explicit instantiation for input_port_t to implement VecElement
    // impl CxxVector<input_port_t> {}
}

pub use ffi::*;
