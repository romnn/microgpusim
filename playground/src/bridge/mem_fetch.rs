#[cxx::bridge]
mod default {
    unsafe extern "C++" {
        include!("playground/src/bridge.hpp");

        // this is not implemented yet, see hotfix in box_interconnect.cc instead

        type memory_config;

        type mem_access_t;
        // fn new_mem_access_t() -> UniquePtr<mem_access_t>;

        type mem_fetch;
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

#[repr(transparent)]
pub struct MemFetch(cxx::UniquePtr<default::mem_fetch>);

impl MemFetch {
    pub fn as_ptr(&self) -> *const default::mem_fetch {
        self.0.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut default::mem_fetch {
        self.0.as_mut_ptr()
    }
}
