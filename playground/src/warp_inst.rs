use playground_sys::warp_inst::warp_inst_bridge;
#[allow(clippy::module_name_repetitions)]
pub use playground_sys::warp_inst::warp_inst_t;
use std::marker::PhantomData;

#[derive(Clone)]
pub struct WarpInstr<'a> {
    inner: cxx::SharedPtr<warp_inst_bridge>,
    phantom: PhantomData<&'a warp_inst_bridge>,
}

impl<'a> WarpInstr<'a> {
    #[inline]
    pub(crate) unsafe fn wrap_ptr(ptr: *const warp_inst_t) -> Self {
        use playground_sys::warp_inst::new_warp_inst_bridge;
        Self {
            inner: new_warp_inst_bridge(ptr),
            phantom: PhantomData,
        }
    }

    #[inline]
    #[must_use]
    pub fn opcode_str(&self) -> &str {
        let inst: &warp_inst_t = self;
        let opcode = unsafe { std::ffi::CStr::from_ptr(inst.opcode_str()) };
        opcode.to_str().unwrap()
    }
}

impl<'a> std::ops::Deref for WarpInstr<'a> {
    type Target = warp_inst_t;

    #[inline]
    fn deref(&self) -> &'a Self::Target {
        unsafe { self.inner.inner().as_ref().unwrap() }
    }
}
