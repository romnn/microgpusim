use color_eyre::eyre;
use std::ffi::CString;
use std::os::raw::c_char;
use std::path::Path;

#[cxx::bridge]
mod default {
    unsafe extern "C++" {
        include!("playground/src/bridge.hpp");

        type c_void;

        type IntersimConfig;
        fn new_intersim_config() -> UniquePtr<IntersimConfig>;
        fn ParseFile(self: Pin<&mut IntersimConfig>, filename: &CxxString);
        fn GetInt(self: &IntersimConfig, field: &CxxString) -> i32;
        fn GetStr<'a, 'b>(self: &'a IntersimConfig, field: &'b CxxString) -> &'a CxxString;
        fn GetFloat(self: &IntersimConfig, field: &CxxString) -> f64;

        type BoxInterconnect;
        unsafe fn new_box_interconnect(config_file: *const c_char) -> UniquePtr<BoxInterconnect>;

        fn HasBuffer(self: &BoxInterconnect, deviceID: u32, size: u32) -> bool;
        fn Advance(self: Pin<&mut BoxInterconnect>);
        fn Busy(self: &BoxInterconnect) -> bool;
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
        fn DisplayMap(self: &BoxInterconnect, dim: u32, count: u32);
        fn GetNumNodes(self: &BoxInterconnect) -> u32;
        fn GetNumMemories(self: &BoxInterconnect) -> u32;
        fn GetNumShaders(self: &BoxInterconnect) -> u32;
        fn GetConfig(self: &BoxInterconnect) -> SharedPtr<IntersimConfig>;

        type InterconnectInterface;
        unsafe fn new_interconnect_interface(
            config_file: *const c_char,
        ) -> UniquePtr<InterconnectInterface>;

        fn HasBuffer(self: &InterconnectInterface, deviceID: u32, size: u32) -> bool;
        fn Advance(self: Pin<&mut InterconnectInterface>);
        fn Busy(self: &InterconnectInterface) -> bool;
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
        fn DisplayMap(self: &InterconnectInterface, dim: u32, count: u32);
        fn GetNumNodes(self: &InterconnectInterface) -> u32;
        fn GetNumMemories(self: &InterconnectInterface) -> u32;
        fn GetNumShaders(self: &InterconnectInterface) -> u32;
        fn GetConfig(self: &InterconnectInterface) -> SharedPtr<IntersimConfig>;
    }
}

pub struct InterconnectInterface<T> {
    inner: cxx::UniquePtr<default::InterconnectInterface>,
    phantom: std::marker::PhantomData<T>,
}

impl<T> InterconnectInterface<T> {
    pub fn num_nodes(&self) -> u32 {
        self.inner.GetNumNodes()
    }

    pub fn num_shaders(&self) -> u32 {
        self.inner.GetNumShaders()
    }

    pub fn num_memories(&self) -> u32 {
        self.inner.GetNumMemories()
    }

    fn init(&mut self) {
        self.inner.pin_mut().Init();
    }

    fn create_interconnect(&mut self, num_clusters: u32, num_mem_sub_partitions: u32) {
        self.inner
            .pin_mut()
            .CreateInterconnect(num_clusters, num_mem_sub_partitions);
    }

    pub fn new(config_file: &Path, num_clusters: u32, num_mem_sub_partitions: u32) -> Self {
        let config_file = config_file.canonicalize().unwrap();
        let config_file = CString::new(&*config_file.to_string_lossy()).unwrap();
        let inner = unsafe { default::new_interconnect_interface(config_file.as_ptr()) };
        let mut interconn = Self {
            inner,
            phantom: std::marker::PhantomData,
        };
        interconn.create_interconnect(num_clusters, num_mem_sub_partitions);
        interconn.init();
        interconn
    }
}

pub struct BoxInterconnect(default::BoxInterconnect);

pub trait Interconnect<T> {
    fn advance(&mut self);
    fn must_pop(&mut self, node: u32) -> eyre::Result<(u16, Box<T>)>;
    fn push(&mut self, src_node: u32, dest_node: u32, value: Box<T>);
    fn pop(&mut self, node: u32) -> Option<Box<T>>;
}

impl<T> Interconnect<T> for InterconnectInterface<T> {
    fn advance(&mut self) {
        self.inner.pin_mut().Advance()
    }

    fn must_pop(&mut self, node: u32) -> eyre::Result<(u16, Box<T>)> {
        for cycle in 0..u16::MAX {
            if let Some(data) = self.pop(node) {
                return Ok((cycle, data));
            }
            self.advance();
        }
        Err(eyre::eyre!(
            "timeout waiting for message after {} cycles",
            u16::MAX
        ))
    }

    fn pop(&mut self, node: u32) -> Option<Box<T>> {
        let value = unsafe { self.inner.pin_mut().Pop(node) };
        if value.is_null() {
            None
        } else {
            let value: Box<T> = unsafe { Box::from_raw(value as *mut T) };
            Some(value)
        }
    }

    fn push(&mut self, src_node: u32, dest_node: u32, value: Box<T>) {
        let mut value: &mut T = Box::leak(value);
        unsafe {
            self.inner.pin_mut().Push(
                src_node,
                dest_node,
                // (&mut value as *mut Box<T>) as *mut default::c_void,
                (value as *mut T) as *mut default::c_void,
                std::mem::size_of::<T>() as u32, // (&value) as u32,
                                                 // std::mem::size_of_val(&value) as u32,
            )
        };
    }
}

pub use default::IntersimConfig;
