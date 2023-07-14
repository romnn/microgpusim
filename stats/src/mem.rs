use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, strum::EnumIter, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AccessKind {
    GLOBAL_ACC_R,
    LOCAL_ACC_R,
    CONST_ACC_R,
    TEXTURE_ACC_R,
    GLOBAL_ACC_W,
    LOCAL_ACC_W,
    L1_WRBK_ACC,
    L2_WRBK_ACC,
    INST_ACC_R,
    L1_WR_ALLOC_R,
    L2_WR_ALLOC_R,
}

impl AccessKind {
    #[must_use]
    pub fn is_write(self) -> bool {
        match self {
            AccessKind::GLOBAL_ACC_R
            | AccessKind::LOCAL_ACC_R
            | AccessKind::CONST_ACC_R
            | AccessKind::TEXTURE_ACC_R
            | AccessKind::INST_ACC_R
            | AccessKind::L1_WR_ALLOC_R
            | AccessKind::L2_WR_ALLOC_R => false,
            // | AccessKind::NUM_MEM_ACCESS_TYPE => false,
            AccessKind::GLOBAL_ACC_W
            | AccessKind::LOCAL_ACC_W
            | AccessKind::L1_WRBK_ACC
            | AccessKind::L2_WRBK_ACC => true,
        }
    }
}

#[derive(Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Accesses(pub HashMap<AccessKind, usize>);

impl Accesses {
    #[must_use]
    pub fn num_writes(&self) -> usize {
        self.0
            .iter()
            .filter(|(kind, _)| kind.is_write())
            .map(|(_, count)| count)
            .sum()
    }

    #[must_use]
    pub fn num_reads(&self) -> usize {
        self.0
            .iter()
            .filter(|(kind, _)| !kind.is_write())
            .map(|(_, count)| count)
            .sum()
    }

    pub fn inc(&mut self, kind: impl Into<AccessKind>, count: usize) {
        *self.0.entry(kind.into()).or_insert(0) += count;
    }
}

impl std::fmt::Debug for Accesses {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut accesses: Vec<_> = self
            .0
            .iter()
            .filter(|(_, &count)| count > 0)
            .map(|(kind, count)| (format!("{kind:?}"), count))
            .collect();
        accesses.sort_by_key(|(key, _)| key.clone());

        let mut out = f.debug_struct("Accesses");
        for (key, count) in accesses {
            out.field(&key, count);
        }
        out.finish()
    }
}

impl std::ops::Deref for Accesses {
    type Target = HashMap<AccessKind, usize>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for Accesses {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
