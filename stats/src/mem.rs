use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(
    Debug,
    strum::EnumIter,
    strum::EnumCount,
    strum::FromRepr,
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
#[repr(usize)]
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
    // #[inline]
    pub const fn count() -> usize {
        use strum::EnumCount;
        Self::COUNT
    }

    #[must_use]
    // #[inline]
    pub fn iter() -> <Self as strum::IntoEnumIterator>::Iterator {
        <Self as strum::IntoEnumIterator>::iter()
    }

    #[must_use]
    // #[inline]
    pub fn reads() -> impl Iterator<Item = Self> {
        Self::iter().filter(|kind| kind.is_read())
    }

    #[must_use]
    // #[inline]
    pub fn writes() -> impl Iterator<Item = Self> {
        Self::iter().filter(|kind| kind.is_write())
    }

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
            AccessKind::GLOBAL_ACC_W
            | AccessKind::LOCAL_ACC_W
            | AccessKind::L1_WRBK_ACC
            | AccessKind::L2_WRBK_ACC => true,
        }
    }

    #[must_use]
    pub fn is_read(self) -> bool {
        !self.is_write()
    }
}

#[derive(Debug, Default, Clone, Hash, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct PhysicalAddress {
    pub bk: u64,
    pub chip: u64,
    pub row: u64,
    pub col: u64,
    pub burst: u64,
    pub sub_partition: u64,
}

// #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
// pub enum RequestKind {
//     ReadRequest,
//     WriteRequest,
//     ReadReply,
//     WriteAck,
// }

pub type address = u64;

/// A memory access.
#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Access {
    /// Address in linear, virtual memory space.
    pub addr: address,
    /// Relative address within allocation.
    pub relative_addr: Option<address>,
    /// Physical address in global memory (DRAM).
    pub physical_addr: PhysicalAddress,
    /// Memory partition address.
    pub partition_addr: address,
    /// Access kind.
    pub kind: AccessKind,
    /// The number of bytes requested.
    pub requested_bytes: u32,
    // /// Byte mask.
    // pub byte_mask: u32,
    // /// Warp active mask.
    // pub warp_active_mask: u32,
    /// Allocation ID that this access references.
    pub allocation_id: Option<usize>,

    /// Warp id of the warp inside the block that issued this access.
    pub warp_id: usize,

    // TODO: add block id too..
    /// Kernel launch ID of the kernel that issued this access.
    pub kernel_launch_id: Option<usize>,
    /// Core that issued this access.
    pub core_id: Option<usize>,
    /// Cluster that issued this access.
    pub cluster_id: Option<usize>,

    /// Cycle the access was pushed to the interconnect.
    pub inject_cycle: Option<u64>,
    /// Cycle the access was returned to the requester.
    pub return_cycle: Option<u64>,
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
            *self.inner.entry(alloc_id).or_insert(0) += count;
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

    #[must_use]
    pub fn num_accesses(&self, kind: AccessKind) -> u64 {
        self.iter()
            .filter(|((_, k), _)| *k == kind)
            .map(|(_, count)| count)
            .sum()
    }

    #[must_use]
    pub fn into_csv_rows(self, full: bool) -> Vec<CsvRow> {
        let mut rows = Vec::new();
        for ((allocation_id, access_kind), num_accesses) in self.inner {
            let need_row = rows.is_empty();
            if !full && !need_row && num_accesses < 1 {
                continue;
            }
            rows.push(CsvRow {
                kernel_name: self.kernel_info.name.clone(),
                kernel_name_mangled: self.kernel_info.mangled_name.clone(),
                kernel_launch_id: self.kernel_info.launch_id,
                allocation_id,
                access_kind,
                num_accesses,
            });
        }

        // self.inner
        //     .into_iter()
        //     .map(|((allocation_id, access_kind), num_accesses)| CsvRow {
        //         kernel_name: self.kernel_info.name.clone(),
        //         kernel_name_mangled: self.kernel_info.mangled_name.clone(),
        //         kernel_launch_id: self.kernel_info.launch_id,
        //         allocation_id,
        //         access_kind,
        //         num_accesses,
        //     })
        //     .collect()
        rows
    }

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
