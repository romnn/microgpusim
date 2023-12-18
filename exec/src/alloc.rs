use super::kernel::ThreadIndex;
use super::{model, tracegen};
use std::sync::Arc;

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Options {
    pub mem_space: model::MemorySpace,
    pub name: Option<String>,
    pub fill_l2: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            mem_space: model::MemorySpace::Global,
            name: None,
            fill_l2: true,
        }
    }
}

pub trait Allocatable {
    fn length(&self) -> usize;
    fn stride(&self) -> usize;
    fn size(&self) -> usize {
        self.length() * self.stride()
    }
}

impl<'a, T> Allocatable for &'a mut Vec<T> {
    fn length(&self) -> usize {
        self.len()
    }

    fn stride(&self) -> usize {
        std::mem::size_of::<T>()
    }
}

impl<'a, T> Allocatable for &'a Vec<T> {
    fn length(&self) -> usize {
        self.len()
    }

    fn stride(&self) -> usize {
        std::mem::size_of::<T>()
    }
}

impl<T> Allocatable for Vec<T> {
    fn length(&self) -> usize {
        self.len()
    }

    fn stride(&self) -> usize {
        std::mem::size_of::<T>()
    }
}

#[derive(Clone)]
pub struct DevicePtr<T> {
    pub inner: T,
    pub memory: Arc<dyn tracegen::MemoryAccess + Send + Sync>,
    pub mem_space: model::MemorySpace,
    pub bypass_l1: bool,
    pub bypass_l2: bool,
    pub offset: u64,
}

impl<T> std::fmt::Debug for DevicePtr<T>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.inner, f)
    }
}

impl<T> std::fmt::Display for DevicePtr<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.inner, f)
    }
}

impl<T> DevicePtr<T> {
    pub fn load<Idx>(&self, thread_idx: &ThreadIndex, idx: Idx) -> &<T as Index<Idx>>::Output
    where
        T: Index<Idx>,
    {
        let (elem, rel_offset, size) = self.inner.index(idx);
        let addr = self.offset + rel_offset;
        self.memory.load(
            thread_idx,
            addr,
            size,
            self.mem_space,
            self.bypass_l1,
            self.bypass_l2,
        );
        elem
    }

    pub fn load_mut<Idx>(
        &mut self,
        thread_idx: &ThreadIndex,
        idx: Idx,
    ) -> &mut <T as Index<Idx>>::Output
    where
        T: IndexMut<Idx>,
    {
        let (elem, rel_offset, size) = self.inner.index_mut(idx);
        let addr = self.offset + rel_offset;
        self.memory.store(
            thread_idx,
            addr,
            size,
            self.mem_space,
            self.bypass_l1,
            self.bypass_l2,
        );
        elem
    }

    pub fn store<Idx>(
        &mut self,
        thread_idx: &ThreadIndex,
        idx: Idx,
        value: <T as Index<Idx>>::Output,
    ) where
        T: IndexMut<Idx>,
        <T as Index<Idx>>::Output: Sized,
    {
        *self.load_mut(thread_idx, idx) = value;
    }
}

pub trait Index<Idx: ?Sized> {
    type Output: ?Sized;
    fn index(&self, idx: Idx) -> (&Self::Output, u64, u32);
}

pub trait IndexMut<Idx: ?Sized>: Index<Idx> {
    fn index_mut(&mut self, index: Idx) -> (&mut Self::Output, u64, u32);
}

// TODO: consolidate
impl<'a, T, Idx> Index<Idx> for &'a mut Vec<T>
where
    Idx: super::ToLinear,
{
    type Output = T;
    fn index(&self, idx: Idx) -> (&Self::Output, u64, u32) {
        let idx = idx.to_linear();
        let rel_addr = idx as u64 * self.stride() as u64;
        let size = self.stride() as u32;
        (&self[idx], rel_addr, size)
    }
}

impl<'a, T, Idx> Index<Idx> for &'a Vec<T>
where
    Idx: super::ToLinear,
{
    type Output = T;
    fn index(&self, idx: Idx) -> (&Self::Output, u64, u32) {
        let idx = idx.to_linear();
        let rel_addr = idx as u64 * self.stride() as u64;
        let size = self.stride() as u32;
        (&self[idx], rel_addr, size)
    }
}

impl<T, Idx> Index<Idx> for Vec<T>
where
    Idx: super::ToLinear,
{
    type Output = T;
    fn index(&self, idx: Idx) -> (&Self::Output, u64, u32) {
        let idx = idx.to_linear();
        let rel_addr = idx as u64 * self.stride() as u64;
        let size = self.stride() as u32;
        (&self[idx], rel_addr, size)
    }
}

// TODO consolidate
impl<'a, T, Idx> IndexMut<Idx> for &'a mut Vec<T>
where
    Idx: super::ToLinear,
{
    fn index_mut(&mut self, idx: Idx) -> (&mut Self::Output, u64, u32) {
        let idx = idx.to_linear();
        let rel_addr = idx as u64 * self.stride() as u64;
        let size = self.stride() as u32;
        (&mut self[idx], rel_addr, size)
    }
}

impl<T, Idx> IndexMut<Idx> for Vec<T>
where
    Idx: super::ToLinear,
{
    fn index_mut(&mut self, idx: Idx) -> (&mut Self::Output, u64, u32) {
        let idx = idx.to_linear();
        let rel_addr = idx as u64 * self.stride() as u64;
        let size = self.stride() as u32;
        (&mut self[idx], rel_addr, size)
    }
}

impl<T, Idx> std::ops::Index<(&ThreadIndex, Idx)> for DevicePtr<T>
where
    T: Index<Idx>,
{
    type Output = <T as Index<Idx>>::Output;

    fn index(&self, (thread_idx, idx): (&ThreadIndex, Idx)) -> &Self::Output {
        self.load(thread_idx, idx)
        // let (elem, rel_offset, size) = self.inner.index(idx);
        // let addr = self.offset + rel_offset;
        // self.memory.load(thread_idx, addr, size, self.mem_space);
        // elem
    }
}

impl<T, Idx> std::ops::IndexMut<(&ThreadIndex, Idx)> for DevicePtr<T>
where
    T: IndexMut<Idx>,
{
    fn index_mut(&mut self, (thread_idx, idx): (&ThreadIndex, Idx)) -> &mut Self::Output {
        self.load_mut(thread_idx, idx)
        // let (elem, rel_offset, size) = self.inner.index_mut(idx);
        // let addr = self.offset + rel_offset;
        // self.memory.store(thread_idx, addr, size, self.mem_space);
        // elem
    }
}
