#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("playground-sys/src/bridge.hpp");

        type c_void;

        type IntersimConfig;
        #[must_use]
        fn new_intersim_config() -> UniquePtr<IntersimConfig>;
        #[must_use]
        fn ParseFile(self: Pin<&mut IntersimConfig>, filename: &CxxString) -> i32;
        #[must_use]
        fn GetInt(self: &IntersimConfig, field: &CxxString) -> i32;
        #[must_use]
        fn GetStr<'a, 'b>(self: &'a IntersimConfig, field: &'b CxxString) -> &'a CxxString;
        #[must_use]
        fn GetFloat(self: &IntersimConfig, field: &CxxString) -> f64;

        type BoxInterconnect;
        #[must_use]
        unsafe fn new_box_interconnect(config_file: *const c_char) -> UniquePtr<BoxInterconnect>;

        #[must_use]
        fn HasBuffer(self: &BoxInterconnect, deviceID: u32, size: u32) -> bool;
        fn Advance(self: Pin<&mut BoxInterconnect>);
        #[must_use]
        fn Busy(self: &BoxInterconnect) -> bool;
        #[must_use]
        unsafe fn Pop(self: Pin<&mut BoxInterconnect>, deviceID: u32) -> *mut c_void;
        unsafe fn Push(
            self: Pin<&mut BoxInterconnect>,
            input_deviceID: u32,
            output_deviceID: u32,
            data: *mut c_void,
            size: u32,
        );
        fn Init(self: Pin<&mut BoxInterconnect>);
        fn CreateInterconnect(self: Pin<&mut BoxInterconnect>, n_shader: u32, n_mem: u32);
        fn DisplayMapStdout(self: &BoxInterconnect, dim: u32, count: u32);

        #[must_use]
        fn GetNumNodes(self: &BoxInterconnect) -> u32;
        #[must_use]
        fn GetNumMemories(self: &BoxInterconnect) -> u32;
        #[must_use]
        fn GetNumShaders(self: &BoxInterconnect) -> u32;
        #[must_use]
        fn GetConfig(self: &BoxInterconnect) -> SharedPtr<IntersimConfig>;

        type InterconnectInterface;
        #[must_use]
        unsafe fn new_interconnect_interface(
            config_file: *const c_char,
        ) -> UniquePtr<InterconnectInterface>;

        #[must_use]
        fn HasBuffer(self: &InterconnectInterface, deviceID: u32, size: u32) -> bool;
        fn Advance(self: Pin<&mut InterconnectInterface>);
        #[must_use]
        fn Busy(self: &InterconnectInterface) -> bool;
        #[must_use]
        unsafe fn Pop(self: Pin<&mut InterconnectInterface>, deviceID: u32) -> *mut c_void;
        unsafe fn Push(
            self: Pin<&mut InterconnectInterface>,
            input_deviceID: u32,
            output_deviceID: u32,
            data: *mut c_void,
            size: u32,
        );
        fn Init(self: Pin<&mut InterconnectInterface>);
        fn CreateInterconnect(self: Pin<&mut InterconnectInterface>, n_shader: u32, n_mem: u32);
        fn DisplayMapStdout(self: &InterconnectInterface, dim: u32, count: u32);

        #[must_use]
        fn GetNumNodes(self: &InterconnectInterface) -> u32;
        #[must_use]
        fn GetNumMemories(self: &InterconnectInterface) -> u32;
        #[must_use]
        fn GetNumShaders(self: &InterconnectInterface) -> u32;
        #[must_use]
        fn GetConfig(self: &InterconnectInterface) -> SharedPtr<IntersimConfig>;
    }
}

pub use ffi::*;
