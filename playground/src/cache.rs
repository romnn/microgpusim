#[allow(clippy::module_name_repetitions)]
pub use playground_sys::cache::cache_block_t;
use playground_sys::cache::{baseline_cache, cache_bridge};
use std::marker::PhantomData;

#[derive(Clone)]
pub struct L2<'a> {
    inner: cxx::SharedPtr<cache_bridge>,
    phantom: PhantomData<&'a cache_bridge>,
}

impl<'a> std::ops::Deref for L2<'a> {
    type Target = baseline_cache;

    fn deref(&self) -> &'a Self::Target {
        unsafe { self.inner.inner().as_ref().unwrap() }
    }
}

impl<'a> L2<'a> {
    #[must_use]
    pub fn new(inner: cxx::SharedPtr<cache_bridge>) -> Self {
        Self {
            inner,
            phantom: PhantomData,
        }
    }

    #[must_use]
    pub fn lines(&self) -> Vec<&cache_block_t> {
        self.inner
            .get_lines()
            .into_iter()
            .map(|ptr| unsafe { &*ptr.get() as &_ })
            .collect()
    }
}
