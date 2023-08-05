#![allow(clippy::module_name_repetitions)]

use color_eyre::eyre;
use playground_sys::types;
use std::ffi::CString;
use std::path::Path;

pub struct InterconnectInterface(cxx::UniquePtr<types::interconnect::InterconnectInterface>);
pub struct BoxInterconnect(cxx::UniquePtr<types::interconnect::BoxInterconnect>);

impl InterconnectInterface {
    #[must_use]
    pub fn new(config_file: &Path, num_clusters: u32, num_mem_sub_partitions: u32) -> Self {
        let config_file = config_file.canonicalize().unwrap();
        let config_file = CString::new(&*config_file.to_string_lossy()).unwrap();
        let mut interconn =
            Self(unsafe { types::interconnect::new_interconnect_interface(config_file.as_ptr()) });
        interconn.create_interconnect(num_clusters, num_mem_sub_partitions);
        interconn.init();
        interconn
    }
}

impl BoxInterconnect {
    #[must_use]
    pub fn new(config_file: &Path, num_clusters: u32, num_mem_sub_partitions: u32) -> Self {
        let config_file = config_file.canonicalize().unwrap();
        let config_file = CString::new(&*config_file.to_string_lossy()).unwrap();
        let mut interconn =
            Self(unsafe { types::interconnect::new_box_interconnect(config_file.as_ptr()) });
        interconn.create_interconnect(num_clusters, num_mem_sub_partitions);
        interconn.init();
        interconn
    }
}

pub trait BridgedInterconnect {
    fn init(&mut self);
    fn create_interconnect(&mut self, num_clusters: u32, num_mem_sub_partitions: u32);
    fn num_nodes(&self) -> u32;
    fn num_shaders(&self) -> u32;
    fn num_memories(&self) -> u32;
    fn advance(&mut self);
    fn has_buffer(&mut self, node: u32, size: u32) -> bool;
    unsafe fn push(
        &mut self,
        src_node: u32,
        dest_node: u32,
        value: *mut playground_sys::types::c_void,
        size: u32,
    );
    fn pop(&mut self, node: u32) -> *mut playground_sys::types::c_void;
}

impl BridgedInterconnect for InterconnectInterface {
    fn init(&mut self) {
        self.0.pin_mut().Init();
    }
    fn create_interconnect(&mut self, num_clusters: u32, num_mem_sub_partitions: u32) {
        self.0
            .pin_mut()
            .CreateInterconnect(num_clusters, num_mem_sub_partitions);
    }
    #[must_use]
    fn num_nodes(&self) -> u32 {
        self.0.GetNumNodes()
    }
    #[must_use]
    fn num_shaders(&self) -> u32 {
        self.0.GetNumShaders()
    }
    #[must_use]
    fn num_memories(&self) -> u32 {
        self.0.GetNumMemories()
    }
    fn advance(&mut self) {
        self.0.pin_mut().Advance();
    }
    #[must_use]
    fn has_buffer(&mut self, node: u32, size: u32) -> bool {
        self.0.HasBuffer(node, size)
    }
    unsafe fn push(&mut self, src_node: u32, dest_node: u32, value: *mut types::c_void, size: u32) {
        self.0.pin_mut().Push(src_node, dest_node, value, size);
    }
    #[must_use]
    fn pop(&mut self, node: u32) -> *mut types::c_void {
        unsafe { self.0.pin_mut().Pop(node) }
    }
}

impl BridgedInterconnect for BoxInterconnect {
    fn init(&mut self) {
        self.0.pin_mut().Init();
    }
    fn create_interconnect(&mut self, num_clusters: u32, num_mem_sub_partitions: u32) {
        self.0
            .pin_mut()
            .CreateInterconnect(num_clusters, num_mem_sub_partitions);
    }
    #[must_use]
    fn num_nodes(&self) -> u32 {
        self.0.GetNumNodes()
    }
    #[must_use]
    fn num_shaders(&self) -> u32 {
        self.0.GetNumShaders()
    }
    #[must_use]
    fn num_memories(&self) -> u32 {
        self.0.GetNumMemories()
    }
    fn advance(&mut self) {
        self.0.pin_mut().Advance();
    }
    #[must_use]
    fn has_buffer(&mut self, node: u32, size: u32) -> bool {
        self.0.HasBuffer(node, size)
    }
    unsafe fn push(&mut self, src_node: u32, dest_node: u32, value: *mut types::c_void, size: u32) {
        self.0.pin_mut().Push(src_node, dest_node, value, size);
    }
    #[must_use]
    fn pop(&mut self, node: u32) -> *mut types::c_void {
        unsafe { self.0.pin_mut().Pop(node) }
    }
}

pub struct Interconnect<T, I> {
    inner: I,
    phantom: std::marker::PhantomData<T>,
}

impl<T, I> Interconnect<T, I>
where
    I: BridgedInterconnect,
{
    pub fn new(inner: I) -> Self {
        Self {
            inner,
            phantom: std::marker::PhantomData,
        }
    }

    #[must_use]
    pub fn num_nodes(&self) -> u32 {
        self.inner.num_nodes()
    }

    #[must_use]
    pub fn num_shaders(&self) -> u32 {
        self.inner.num_shaders()
    }

    #[must_use]
    pub fn num_memories(&self) -> u32 {
        self.inner.num_memories()
    }

    pub fn advance(&mut self) {
        self.inner.advance();
    }

    pub fn must_pop(&mut self, node: u32, limit: Option<u16>) -> eyre::Result<(u16, Box<T>)> {
        let limit = limit.unwrap_or(u16::MAX);
        assert!(limit > 0);
        for cycle in 0..limit {
            if let Some(data) = self.pop(node) {
                return Ok((cycle, data));
            }
            self.advance();
        }
        Err(eyre::eyre!(
            "timeout waiting for message after {} cycles",
            limit
        ))
    }

    #[must_use]
    pub fn pop(&mut self, node: u32) -> Option<Box<T>> {
        let value = self.inner.pop(node);
        if value.is_null() {
            None
        } else {
            let value: Box<T> = unsafe { Box::from_raw(value.cast::<T>()) };
            Some(value)
        }
    }

    pub fn push(&mut self, src_node: u32, dest_node: u32, value: Box<T>) {
        let value: &mut T = Box::leak(value);
        assert!(self.inner.has_buffer(src_node, 8));
        unsafe {
            self.inner.push(
                src_node,
                dest_node,
                (value as *mut T).cast::<types::c_void>(),
                u32::try_from(std::mem::size_of::<T>()).unwrap(),
            );
        }
    }
}

pub struct IntersimConfig(cxx::UniquePtr<types::interconnect::IntersimConfig>);

impl std::fmt::Display for IntersimConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("IntersimConfig").finish()
    }
}

impl std::fmt::Debug for IntersimConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("IntersimConfig").finish()
    }
}

impl Default for IntersimConfig {
    #[must_use]
    fn default() -> Self {
        Self::new()
    }
}

impl IntersimConfig {
    #[must_use]
    pub fn new() -> Self {
        Self(types::interconnect::new_intersim_config())
    }

    pub fn from_file(path: &Path) -> eyre::Result<Self> {
        let mut config = Self(types::interconnect::new_intersim_config());
        config.parse_file(path)?;
        Ok(config)
    }

    fn parse_file(&mut self, path: &Path) -> eyre::Result<()> {
        let config_file = path.canonicalize()?;
        cxx::let_cxx_string!(config_file_string = config_file.to_string_lossy().to_string());
        let ret_code = self.0.pin_mut().ParseFile(&config_file_string);
        if ret_code != 0 {
            eyre::bail!("error parsing config file {}", config_file.display());
        }
        Ok(())
    }

    #[must_use]
    pub fn get_int(&self, field: impl AsRef<str>) -> i32 {
        cxx::let_cxx_string!(field = field.as_ref());
        self.0.GetInt(&field)
    }

    #[must_use]
    pub fn get_string(&self, field: impl AsRef<str>) -> String {
        cxx::let_cxx_string!(field = field.as_ref());
        self.0.GetStr(&field).to_string_lossy().to_string()
    }

    #[must_use]
    pub fn get_bool(&self, field: impl AsRef<str>) -> bool {
        self.get_int(field) != 0
    }
}
