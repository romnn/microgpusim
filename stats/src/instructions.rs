use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(
    Debug,
    Copy,
    Clone,
    strum::EnumIter,
    Hash,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
)]
pub enum MemorySpace {
    // undefined_space = 0,
    // reg_space,
    Local,
    // local_space,
    Shared,
    // shared_space,
    // sstarr_space,
    // param_space_unclassified,
    // global to all threads in a kernel (read-only)
    // param_space_kernel,
    // local to a thread (read-writable)
    // param_space_local,
    Constant,
    // const_space,
    Texture,
    // tex_space,
    // surf_space,
    Global,
    // global_space,
    // generic_space,
    // instruction_space,
}

// pub type CsvRow = ((Option<usize>, MemorySpace, bool), u64);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CsvRow {
    pub kernel_name: String,
    pub kernel_name_mangled: String,
    pub kernel_launch_id: usize,
    pub allocation_id: Option<usize>,
    pub memory_space: MemorySpace,
    pub is_write: bool,
    pub num_instructions: u64,
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstructionCounts {
    pub kernel_info: super::KernelInfo,
    pub inner: HashMap<(Option<usize>, MemorySpace, bool), u64>,
}

impl Default for InstructionCounts {
    fn default() -> Self {
        use strum::IntoEnumIterator;
        let mut inner = HashMap::new();
        for memory_space in MemorySpace::iter() {
            inner.insert((None, memory_space, true), 0);
            inner.insert((None, memory_space, false), 0);
        }
        Self {
            inner,
            kernel_info: super::KernelInfo::default(),
        }
    }
}

impl std::ops::AddAssign for InstructionCounts {
    fn add_assign(&mut self, other: Self) {
        for (k, v) in other.inner {
            *self.inner.entry(k).or_insert(0) += v;
        }
    }
}

impl InstructionCounts {
    #[must_use]
    pub fn into_csv_rows(self) -> Vec<CsvRow> {
        self.inner
            .into_iter()
            // .sort_by_key(|(key, _)| *key)
            .map(
                |((allocation_id, memory_space, is_write), num_instructions)| CsvRow {
                    kernel_name: self.kernel_info.name.clone(),
                    kernel_name_mangled: self.kernel_info.mangled_name.clone(),
                    kernel_launch_id: self.kernel_info.launch_id,
                    allocation_id,
                    memory_space,
                    is_write,
                    num_instructions,
                },
            )
            .collect()
    }

    #[must_use]
    pub fn into_inner(self) -> HashMap<(Option<usize>, MemorySpace, bool), u64> {
        self.inner
    }
}

impl std::fmt::Debug for InstructionCounts {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut instructions: Vec<_> = self
            .inner
            .iter()
            .filter(|(_, &count)| count > 0)
            .map(|((alloc_id, space, is_store), count)| {
                (
                    format!("{:?}[{}]", space, if *is_store { "STORE" } else { "LOAD" }),
                    count,
                )
            })
            .collect();
        instructions.sort_by_key(|(key, _)| key.clone());

        let mut out = f.debug_struct("InstructionCounts");
        for (key, count) in instructions {
            out.field(&key, count);
        }
        out.finish_non_exhaustive()
    }
}

impl std::ops::Deref for InstructionCounts {
    type Target = HashMap<(Option<usize>, MemorySpace, bool), u64>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::ops::DerefMut for InstructionCounts {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl InstructionCounts {
    #[must_use]
    pub fn num_instructions(&self, space: MemorySpace, is_write: bool) -> u64 {
        self.inner
            .iter()
            .filter(|((_, s, w), _)| *s == space && *w == is_write)
            .map(|(_, count)| count)
            .sum()
    }

    #[must_use]
    pub fn get_total(&self, space: MemorySpace) -> u64 {
        self.inner
            .iter()
            .filter(|((_, s, _), _)| *s == space)
            .map(|(_, count)| count)
            .sum()
        // let stores = self.0.get(&(space, true)).unwrap_or(&0);
        // let loads = self.0.get(&(space, false)).unwrap_or(&0);
        // stores + loads
    }

    pub fn inc(
        &mut self,
        alloc_id: Option<usize>,
        space: impl Into<MemorySpace>,
        is_store: bool,
        count: u64,
    ) {
        *self
            .inner
            .entry((alloc_id, space.into(), is_store))
            .or_insert(0) += count;
    }
}
