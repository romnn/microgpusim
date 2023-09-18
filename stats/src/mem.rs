use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(
    Debug,
    strum::EnumIter,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
)]
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

/// Memory access statistics.
///
/// Records the number of memory fetches sent from SMs to the interconnect.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Accesses {
    pub kernel_info: super::KernelInfo,
    pub inner: HashMap<(Option<usize>, AccessKind), u64>,
}

impl Default for Accesses {
    fn default() -> Self {
        use strum::IntoEnumIterator;
        let mut inner = HashMap::new();
        for access_kind in AccessKind::iter() {
            inner.insert((None, access_kind), 0);
        }
        Self {
            inner,
            kernel_info: super::KernelInfo::default(),
        }
    }
}

impl std::ops::AddAssign for Accesses {
    fn add_assign(&mut self, other: Self) {
        for (alloc_id, count) in other.inner {
            // for (kind, count) in per_alloc {
            // *self.0.entry(alloc_id).or_default().entry(key).or_insert(0) += count;
            *self.inner.entry(alloc_id).or_insert(0) += count;
            // }
        }
    }
}

impl FromIterator<((Option<usize>, AccessKind), u64)> for Accesses {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = ((Option<usize>, AccessKind), u64)>,
    {
        Self {
            inner: iter.into_iter().collect(),
            kernel_info: super::KernelInfo::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CsvRow {
    pub kernel_name: String,
    pub kernel_name_mangled: String,
    pub kernel_launch_id: usize,
    pub allocation_id: Option<usize>,
    pub access_kind: AccessKind,
    pub num_accesses: u64,
}

impl Accesses {
    #[must_use]
    pub fn into_inner(self) -> HashMap<(Option<usize>, AccessKind), u64> {
        self.inner
    }

    #[must_use] pub fn num_accesses(&self, kind: AccessKind) -> u64 {
        self.iter()
            .filter(|((_, k), _)| *k == kind)
            .map(|(_, count)| count)
            .sum()
    }

    #[must_use]
    pub fn into_csv_rows(self) -> Vec<CsvRow> {
        // let mut flattened: Vec<_> = self.into_inner().into_iter()
        // use itertools::Itertools;
        self.inner
            .into_iter()
            // .sort_by_key(|(key, _)| *key)
            .map(|((allocation_id, access_kind), num_accesses)| CsvRow {
                kernel_name: self.kernel_info.name.clone(),
                kernel_name_mangled: self.kernel_info.mangled_name.clone(),
                kernel_launch_id: self.kernel_info.launch_id,
                allocation_id,
                access_kind,
                num_accesses,
            })
            .collect()
        // flattened.sort_by_key(|(kind, _)| *kind);
        // flattened
    }

    // #[must_use]
    // pub fn flatten(self) -> Vec<((Option<usize>, AccessKind), u64)> {
    //     let mut flattened: Vec<_> = self.into_inner().into_iter().collect();
    //     flattened.sort_by_key(|(kind, _)| *kind);
    //     flattened
    // }

    #[must_use]
    pub fn num_writes(&self) -> u64 {
        self.inner
            .iter()
            .filter(|((_, kind), _)| kind.is_write())
            .map(|(_, count)| count)
            .sum()
    }

    #[must_use]
    pub fn num_reads(&self) -> u64 {
        self.inner
            .iter()
            .filter(|((_, kind), _)| !kind.is_write())
            .map(|(_, count)| count)
            .sum()
    }

    pub fn inc(&mut self, allocation_id: Option<usize>, kind: impl Into<AccessKind>, count: u64) {
        *self.inner.entry((allocation_id, kind.into())).or_insert(0) += count;
    }
}

impl std::fmt::Debug for Accesses {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut accesses: Vec<_> = self
            .inner
            .iter()
            .filter(|(_, &count)| count > 0)
            .map(|(kind, count)| (format!("{kind:?}"), count))
            .collect();
        accesses.sort_by_key(|(key, _)| key.clone());

        let mut out = f.debug_struct("Accesses");
        for (key, count) in accesses {
            out.field(&key, count);
        }
        out.finish_non_exhaustive()
    }
}

impl std::ops::Deref for Accesses {
    type Target = HashMap<(Option<usize>, AccessKind), u64>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::ops::DerefMut for Accesses {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
