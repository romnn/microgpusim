use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
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

#[derive(Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstructionCounts(pub HashMap<(MemorySpace, bool), usize>);

impl std::fmt::Debug for InstructionCounts {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut instructions: Vec<_> = self
            .0
            .iter()
            .filter(|(_, &count)| count > 0)
            .map(|((space, is_store), count)| {
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
    type Target = HashMap<(MemorySpace, bool), usize>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for InstructionCounts {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl InstructionCounts {
    #[must_use]
    pub fn get_total(&self, space: MemorySpace) -> usize {
        let stores = self.0.get(&(space, true)).unwrap_or(&0);
        let loads = self.0.get(&(space, false)).unwrap_or(&0);
        stores + loads
    }

    pub fn inc(&mut self, space: impl Into<MemorySpace>, is_store: bool, count: usize) {
        *self.0.entry((space.into(), is_store)).or_insert(0) += count;
    }
}
