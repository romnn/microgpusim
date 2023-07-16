use crate::bindings;

super::extern_type!(bindings::mf_type, "mf_type");
super::extern_type!(bindings::mem_access_type, "mem_access_type");
super::extern_type!(bindings::mem_fetch_status, "mem_fetch_status");

#[cxx::bridge]
mod default {
    unsafe extern "C++" {
        include!("playground-sys/src/bridge.hpp");

        type memory_config;
        type mf_type = crate::bindings::mf_type;
        type mem_access_type = crate::bindings::mem_access_type;
        type mem_fetch_status = crate::bindings::mem_fetch_status;

        type mem_access_t;
        // fn new_mem_access_t() -> UniquePtr<mem_access_t>;

        type mem_fetch;

        fn is_reply(self: &mem_fetch) -> bool;
        fn get_data_size(self: &mem_fetch) -> u32;
        fn get_ctrl_size(self: &mem_fetch) -> u32;
        fn size(self: &mem_fetch) -> u32;
        fn is_write(self: &mem_fetch) -> bool;
        fn get_addr(self: &mem_fetch) -> u64;
        fn get_relative_addr(self: &mem_fetch) -> u64;
        fn get_alloc_start_addr(self: &mem_fetch) -> u64;
        fn get_alloc_id(self: &mem_fetch) -> u32;
        fn get_access_size(self: &mem_fetch) -> u32;
        fn get_partition_addr(self: &mem_fetch) -> u64;
        fn get_sub_partition_id(self: &mem_fetch) -> u32;
        fn get_is_write(self: &mem_fetch) -> bool;
        fn get_request_uid(self: &mem_fetch) -> u32;
        fn get_sid(self: &mem_fetch) -> u32;
        fn get_tpc(self: &mem_fetch) -> u32;
        fn get_wid(self: &mem_fetch) -> u32;
        fn istexture(self: &mem_fetch) -> bool;
        fn isconst(self: &mem_fetch) -> bool;
        fn isatomic(self: &mem_fetch) -> bool;
        fn get_type(self: &mem_fetch) -> mf_type;
        fn get_access_type(self: &mem_fetch) -> mem_access_type;
        fn get_pc(self: &mem_fetch) -> u64;
        fn get_status(self: &mem_fetch) -> mem_fetch_status;
        //
        // unsafe fn new_mem_fetch(
        //     ctrl_size: u32,
        //     warp_id: u32,
        //     core_id: u32,
        //     cluster_id: u32,
        //     config: *const memory_config,
        //     cycle: u64,
        // ) -> UniquePtr<mem_fetch>;

    }
}

pub(super) use default::{mem_access_t, mem_access_type, mem_fetch, mem_fetch_status, mf_type};

// #[repr(transparent)]
// pub struct MemFetch(cxx::UniquePtr<default::mem_fetch>);
//
// impl MemFetch {
//     pub fn as_ptr(&self) -> *const default::mem_fetch {
//         self.0.as_ptr()
//     }
//
//     pub fn as_mut_ptr(&mut self) -> *mut default::mem_fetch {
//         self.0.as_mut_ptr()
//     }
// }
