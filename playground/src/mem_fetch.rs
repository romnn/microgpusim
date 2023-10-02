use playground_sys::mem_fetch::{mem_fetch, mem_fetch_bridge, mem_fetch_ptr_shim};
use std::marker::PhantomData;

#[derive(Clone)]
pub struct MemFetch<'a> {
    inner: cxx::SharedPtr<mem_fetch_bridge>,
    phantom: PhantomData<&'a mem_fetch_bridge>,
}

impl<'a> MemFetch<'a> {
    // #[inline]
    pub(crate) unsafe fn wrap_ptr(ptr: *const mem_fetch) -> Self {
        use playground_sys::mem_fetch::new_mem_fetch_bridge;
        Self {
            inner: new_mem_fetch_bridge(ptr),
            phantom: PhantomData,
        }
    }
}

impl<'a> std::ops::Deref for MemFetch<'a> {
    type Target = mem_fetch;

    // #[inline]
    fn deref(&self) -> &'a Self::Target {
        unsafe { self.inner.inner().as_ref().unwrap() }
    }
}

// #[inline]
pub(crate) fn get_mem_fetches<'a>(
    queue: &cxx::UniquePtr<cxx::CxxVector<mem_fetch_ptr_shim>>,
) -> Vec<MemFetch<'a>> {
    queue
        .into_iter()
        .map(|ptr| unsafe { MemFetch::wrap_ptr(ptr.get()) })
        .collect()
}
