use crate::bindings;
use cxx::{type_id, ExternType};

unsafe impl ExternType for bindings::mem_fetch_status {
    type Id = type_id!("mem_fetch_status");
    type Kind = cxx::kind::Trivial;
}

#[cxx::bridge]
mod default {
    unsafe extern "C++" {
        include!("playground/src/bridge.hpp");

        type read_only_cache;

        // todo: build bridges for those
        type cache_config;
        type mem_fetch;
        type mem_fetch_interface;

        type mem_fetch_status = crate::bindings::mem_fetch_status;

        fn new_read_only_cache(
            name: &CxxString,
            config: UniquePtr<cache_config>,
            core_id: i32,
            type_id: i32,
            memport: UniquePtr<mem_fetch_interface>,
            status: mem_fetch_status,
        ) -> UniquePtr<read_only_cache>;

        fn cycle(self: Pin<&mut read_only_cache>);
        unsafe fn fill(self: Pin<&mut read_only_cache>, fetch: *mut mem_fetch, time: u32);
        unsafe fn waiting_for_fill(self: Pin<&mut read_only_cache>, fetch: *mut mem_fetch) -> bool;
        fn access_ready(self: &read_only_cache) -> bool;
        unsafe fn next_access(self: Pin<&mut read_only_cache>) -> *mut mem_fetch;
        fn flush(self: Pin<&mut read_only_cache>);
        fn invalidate(self: Pin<&mut read_only_cache>);
        fn data_port_free(self: &read_only_cache) -> bool;
        fn fill_port_free(self: &read_only_cache) -> bool;
        fn miss_queue_full(self: &read_only_cache, num_miss: u32) -> bool;
    }
}

pub use default::*;
