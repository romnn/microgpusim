use super::mem::AccessKind;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(
    Debug,
    strum::EnumIter,
    Clone,
    Copy,
    Hash,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
)]
pub enum RequestStatus {
    HIT = 0,
    HIT_RESERVED,
    MISS,
    RESERVATION_FAIL,
    SECTOR_MISS,
    MSHR_HIT,
}

#[derive(
    Debug,
    strum::EnumIter,
    Clone,
    Copy,
    Hash,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
)]
pub enum ReservationFailure {
    /// all line are reserved
    LINE_ALLOC_FAIL = 0,
    /// MISS queue (i.e. interconnect or DRAM) is full
    MISS_QUEUE_FULL,
    MSHR_ENTRY_FAIL,
    MSHR_MERGE_ENTRY_FAIL,
    MSHR_RW_PENDING,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AccessStat {
    ReservationFailure(ReservationFailure),
    Status(RequestStatus),
}

impl From<RequestStatus> for AccessStat {
    fn from(status: RequestStatus) -> Self {
        AccessStat::Status(status)
    }
}

impl From<ReservationFailure> for AccessStat {
    fn from(failure: ReservationFailure) -> Self {
        AccessStat::ReservationFailure(failure)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AccessStatus(pub (AccessKind, AccessStat));

impl std::fmt::Display for AccessStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.0 {
            (access_kind, AccessStat::Status(status)) => {
                write!(f, "{access_kind:?}[{status:?}]")
            }
            (access_kind, AccessStat::ReservationFailure(failure)) => {
                write!(f, "{access_kind:?}[{failure:?}]")
            }
        }
    }
}

impl AccessStatus {
    pub fn kind(&self) -> &AccessKind {
        &((self.0).0)
    }

    pub fn stat(&self) -> &AccessStat {
        &((self.0).1)
    }

    pub fn is_reservation_failure_reason(&self) -> bool {
        matches!(self.stat(), AccessStat::ReservationFailure(_))
    }

    pub fn is_reservation_fail(&self) -> bool {
        matches!(
            self.stat(),
            AccessStat::Status(RequestStatus::RESERVATION_FAIL)
        )
    }

    pub fn is_global(&self) -> bool {
        self.kind().is_global()
    }

    pub fn is_local(&self) -> bool {
        self.kind().is_local()
    }

    pub fn is_write(&self) -> bool {
        self.kind().is_write()
    }

    pub fn is_read(&self) -> bool {
        self.kind().is_read()
    }

    pub fn is_hit(&self) -> bool {
        matches!(self.stat(), AccessStat::Status(RequestStatus::HIT))
    }

    pub fn is_pending_hit(&self) -> bool {
        matches!(self.stat(), AccessStat::Status(RequestStatus::HIT_RESERVED))
    }

    pub fn is_sector_miss(&self) -> bool {
        matches!(self.stat(), AccessStat::Status(RequestStatus::SECTOR_MISS))
    }

    pub fn is_miss(&self) -> bool {
        matches!(
            self.stat(),
            AccessStat::Status(RequestStatus::MISS | RequestStatus::SECTOR_MISS)
        )
    }
}

/// Per kernel cache statistics.
///
/// Stats at index `i` correspond to the kernel with launch id `i`.
#[derive(Clone, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerKernel {
    pub inner: Vec<Cache>,
    pub no_kernel: Cache,
}

impl AsRef<Vec<Cache>> for PerKernel {
    fn as_ref(&self) -> &Vec<Cache> {
        &self.inner
    }
}

impl PerKernel {
    // #[inline]
    pub fn get_mut(&mut self, idx: Option<usize>) -> &mut Cache {
        match idx {
            None => &mut self.no_kernel,
            Some(idx) => {
                self.inner.resize_with(idx + 1, Cache::default);
                &mut self.inner[idx]
            }
        }
    }

    // #[inline]
    #[must_use]
    pub fn reduce(self) -> Cache {
        todo!()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CsvRow {
    pub kernel_name: String,
    pub kernel_name_mangled: String,
    pub kernel_launch_id: usize,
    pub allocation_id: Option<usize>,
    pub cache_id: usize,
    pub access_kind: AccessKind,
    pub is_write: bool,
    pub access_status: AccessStat,
    pub num_accesses: usize,
}

#[derive(Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Cache {
    pub inner: HashMap<(Option<usize>, AccessStatus), usize>,
    pub num_l1_cache_bank_conflicts: u64,
    pub num_shared_mem_bank_accesses: u64,
    pub num_shared_mem_bank_conflicts: u64,

    #[cfg(feature = "detailed-stats")]
    pub accesses: Vec<(crate::mem::Access, Option<usize>, AccessStatus)>,
}

impl Default for Cache {
    fn default() -> Self {
        use strum::IntoEnumIterator;
        let mut inner = HashMap::new();
        for access_kind in AccessKind::iter() {
            for status in RequestStatus::iter() {
                inner.insert(
                    (
                        None,
                        AccessStatus((access_kind, AccessStat::Status(status))),
                    ),
                    0,
                );
            }
            for failure in ReservationFailure::iter() {
                inner.insert(
                    (
                        None,
                        AccessStatus((access_kind, AccessStat::ReservationFailure(failure))),
                    ),
                    0,
                );
            }
        }
        Self {
            inner,
            num_shared_mem_bank_accesses: 0,
            num_shared_mem_bank_conflicts: 0,
            num_l1_cache_bank_conflicts: 0,
            #[cfg(feature = "detailed-stats")]
            accesses: Vec::new(),
        }
    }
}

impl AsRef<HashMap<(Option<usize>, AccessStatus), usize>> for Cache {
    fn as_ref(&self) -> &HashMap<(Option<usize>, AccessStatus), usize> {
        &self.inner
    }
}

// impl std::ops::DerefMut for Cache {
//     fn deref_mut(&mut self) -> &mut Self::Target {
//         &mut self.inner
//     }
// }

// impl std::ops::Deref for Cache {
//     type Target = HashMap<(Option<usize>, Access), usize>;
//     fn deref(&self) -> &Self::Target {
//         &self.inner
//     }
// }
//
// impl std::ops::DerefMut for Cache {
//     fn deref_mut(&mut self) -> &mut Self::Target {
//         &mut self.inner
//     }
// }

impl std::ops::AddAssign for Cache {
    fn add_assign(&mut self, other: Self) {
        for (k, v) in other.inner {
            *self.inner.entry(k).or_insert(0) += v;
        }
    }
}

impl std::fmt::Debug for Cache {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut accesses: Vec<_> = self
            .inner
            .iter()
            .filter(|(_, &count)| count > 0)
            // .map(|((access_kind, access_stat), count)| {
            // .map(|((alloc_id, access), count)| {
            //     // let key = match access_stat {
            //     //     AccessStat::Status(status) => {
            //     //         format!("{access_kind:?}[{status:?}]")
            //     //     }
            //     //     AccessStat::ReservationFailure(failure) => {
            //     //         format!("{access_kind:?}[{failure:?}]")
            //     //     }
            //     // };
            //     // let key = match alloc_id {
            //     //     None => access.to_string(),
            //     //     Some(id) => (,
            //     // };
            //     ((alloc_id, access), count)
            // })
            .collect();
        accesses.sort_by_key(|(&key, _)| key.clone());

        let mut out = f.debug_struct("CacheStats");
        for ((id, access), count) in accesses {
            out.field(
                &match id {
                    Some(id) => format!("{id}@{access}"),
                    None => access.to_string(),
                },
                count,
            );
        }
        out.finish_non_exhaustive()
    }
}

impl Cache {
    #[must_use]
    pub fn new(inner: HashMap<(Option<usize>, AccessStatus), usize>) -> Self {
        Self {
            inner,
            ..Self::default()
        }
    }

    pub fn get(
        &self,
        alloc_id: Option<usize>,
        kind: impl Into<AccessKind>,
        status: impl Into<AccessStat>,
    ) -> Option<usize> {
        let kind = kind.into();
        let status = status.into();
        let access_stat = AccessStatus((kind, status));
        self.inner.get(&(alloc_id, access_stat)).copied()
    }

    #[must_use]
    pub fn union<'a>(
        &'a self,
        other: &'a Self,
    ) -> impl Iterator<Item = (&'a (Option<usize>, AccessStatus), (usize, usize))> {
        let keys: std::collections::HashSet<_> =
            self.as_ref().keys().chain(other.as_ref().keys()).collect();
        keys.into_iter().map(|k| {
            let left = self.as_ref().get(k).copied().unwrap_or(0);
            let right = other.as_ref().get(k).copied().unwrap_or(0);
            (k, (left, right))
        })
    }

    #[must_use]
    pub fn reduce_allocations(self) -> Self {
        let mut reduced = Self {
            inner: HashMap::default(),
            ..self.clone()
        };
        for ((_, access_status), value) in self.inner {
            *reduced.inner.entry((None, access_status)).or_insert(0) += value;
        }
        reduced
    }

    pub fn iter(
        &self,
    ) -> std::collections::hash_map::Iter<'_, (Option<usize>, AccessStatus), usize> {
        self.inner.iter()
    }

    pub fn shave(&mut self) {
        self.inner.retain(|_, v| *v > 0);
    }

    #[must_use]
    pub fn merge_allocations(self) -> Cache {
        let mut inner = HashMap::new();
        for ((_, access), count) in self.inner {
            *inner.entry((None, access)).or_insert(0) += count;
        }
        Cache {
            inner,
            ..Self::default()
        }
    }

    #[must_use]
    pub fn count_accesses_of_kind(&self, kind: AccessKind) -> usize {
        self.inner
            .iter()
            .filter(|((_, access), _)| access.kind() == &kind)
            .filter(|((_, access), _)| {
                matches!(
                    access.stat(),
                    AccessStat::Status(
                        RequestStatus::HIT
                            | RequestStatus::MISS
                            | RequestStatus::SECTOR_MISS
                            | RequestStatus::HIT_RESERVED
                    )
                )
            })
            .map(|(_, count)| count)
            .sum()
    }

    #[must_use]
    pub fn count_accesses(&self, access: &AccessStatus) -> usize {
        self.inner
            .iter()
            .filter(|((_, acc), _)| acc == access)
            .map(|(_, count)| count)
            .sum()
    }

    // #[must_use]
    // pub fn num_accesses(&self) -> usize {
    //     self.inner
    //         .iter()
    //         .filter(|((_, access), _)| {
    //             matches!(
    //                 access.stat(),
    //
    //         })
    //         .filter(|((_, access), _)| {
    //             matches!(
    //                 access.stat(),
    //                 AccessStat::Status(
    //                     RequestStatus::HIT
    //                         | RequestStatus::MISS
    //                         | RequestStatus::SECTOR_MISS
    //                         | RequestStatus::HIT_RESERVED
    //                 )
    //             )
    //         })
    //         .map(|(_, count)| count)
    //         .sum()
    // }

    #[must_use]
    pub fn num_accesses(&self) -> usize {
        self.inner
            .iter()
            .filter(|((_, access), _)| {
                matches!(
                    access.stat(),
                    AccessStat::Status(
                        RequestStatus::HIT
                            | RequestStatus::MISS
                            | RequestStatus::SECTOR_MISS
                            | RequestStatus::HIT_RESERVED
                    )
                )
            })
            .map(|(_, count)| count)
            .sum()
    }

    #[must_use]
    pub fn num_global_accesses(&self) -> usize {
        self.inner
            .iter()
            .filter(|((_, access), _)| {
                access.is_global()
                    && matches!(
                        access.stat(),
                        AccessStat::Status(
                            RequestStatus::HIT
                                | RequestStatus::MISS
                                | RequestStatus::SECTOR_MISS
                                | RequestStatus::HIT_RESERVED
                        )
                    )
            })
            .map(|(_, count)| count)
            .sum()
    }

    #[must_use]
    pub fn pending_hits(
        &self,
    ) -> impl Iterator<Item = ((Option<usize>, AccessStatus), usize)> + '_ {
        self.inner
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .filter(|((_, access), _)| access.is_pending_hit())
    }

    #[must_use]
    pub fn sector_misses(
        &self,
    ) -> impl Iterator<Item = ((Option<usize>, AccessStatus), usize)> + '_ {
        self.inner
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .filter(|((_, access), _)| access.is_sector_miss())
    }

    #[must_use]
    pub fn num_read_misses(&self) -> usize {
        self.inner
            .iter()
            .filter(|((_, access), _)| access.is_read() && access.is_miss())
            .map(|(_, count)| count)
            .sum()
    }

    #[must_use]
    pub fn num_global_read_misses(&self) -> usize {
        self.inner
            .iter()
            .filter(|((_, access), _)| access.is_global() && access.is_read() && access.is_miss())
            .map(|(_, count)| count)
            .sum()
    }

    #[must_use]
    pub fn num_write_misses(&self) -> usize {
        self.inner
            .iter()
            .filter(|((_, access), _)| access.is_write() && access.is_miss())
            .map(|(_, count)| count)
            .sum()
    }

    #[must_use]
    pub fn num_global_write_misses(&self) -> usize {
        self.inner
            .iter()
            .filter(|((_, access), _)| access.is_global() && access.is_write() && access.is_miss())
            .map(|(_, count)| count)
            .sum()
    }

    #[must_use]
    pub fn num_misses(&self) -> usize {
        self.inner
            .iter()
            .filter(|((_, access), _)| access.is_miss())
            .map(|(_, count)| count)
            .sum()
    }

    #[must_use]
    pub fn num_global_misses(&self) -> usize {
        self.inner
            .iter()
            .filter(|((_, access), _)| access.is_global() && access.is_miss())
            .map(|(_, count)| count)
            .sum()
    }

    #[must_use]
    pub fn num_read_hits(&self) -> usize {
        self.inner
            .iter()
            .filter(|((_, access), _)| {
                access.is_read() && (access.is_hit() || access.is_pending_hit())
            })
            .map(|(_, count)| count)
            .sum()
    }

    #[must_use]
    pub fn num_global_read_hits(&self) -> usize {
        self.inner
            .iter()
            .filter(|((_, access), _)| {
                access.is_read()
                    && access.is_global()
                    && (access.is_hit() || access.is_pending_hit())
            })
            .map(|(_, count)| count)
            .sum()
    }

    #[must_use]
    pub fn num_reads(&self) -> usize {
        self.inner
            .iter()
            .filter(|((_, access), _)| {
                access.is_read() && (access.is_miss() || access.is_hit() || access.is_pending_hit())
            })
            .map(|(_, count)| count)
            .sum()
    }

    #[must_use]
    pub fn num_global_reads(&self) -> usize {
        self.inner
            .iter()
            .filter(|((_, access), _)| {
                access.is_read()
                    && access.is_global()
                    && (access.is_miss() || access.is_hit() || access.is_pending_hit())
            })
            .map(|(_, count)| count)
            .sum()
    }

    #[must_use]
    pub fn num_writes(&self) -> usize {
        self.inner
            .iter()
            .filter(|((_, access), _)| {
                access.is_write()
                    && (access.is_miss() || access.is_hit() || access.is_pending_hit())
            })
            .map(|(_, count)| count)
            .sum()
    }

    #[must_use]
    pub fn num_global_writes(&self) -> usize {
        self.inner
            .iter()
            .filter(|((_, access), _)| {
                access.is_write()
                    && access.is_global()
                    && (access.is_miss() || access.is_hit() || access.is_pending_hit())
            })
            .map(|(_, count)| count)
            .sum()
    }

    #[must_use]
    pub fn num_write_hits(&self) -> usize {
        self.inner
            .iter()
            .filter(|((_, access), _)| {
                access.is_write() && (access.is_hit() || access.is_pending_hit())
            })
            .map(|(_, count)| count)
            .sum()
    }

    #[must_use]
    pub fn num_global_write_hits(&self) -> usize {
        self.inner
            .iter()
            .filter(|((_, access), _)| {
                access.is_write()
                    && access.is_global()
                    && (access.is_hit() || access.is_pending_hit())
            })
            .map(|(_, count)| count)
            .sum()
    }

    #[must_use]
    pub fn num_hits(&self) -> usize {
        self.inner
            .iter()
            .filter(|((_, access), _)| access.is_hit() || access.is_pending_hit())
            .map(|(_, count)| count)
            .sum()
    }

    #[must_use]
    pub fn num_global_hits(&self) -> usize {
        self.inner
            .iter()
            .filter(|((_, access), _)| {
                access.is_global() && (access.is_hit() || access.is_pending_hit())
            })
            .map(|(_, count)| count)
            .sum()
    }

    #[must_use]
    pub fn global_hit_rate(&self) -> f32 {
        self.num_global_hits() as f32 / self.num_global_accesses() as f32
    }

    #[must_use]
    pub fn hit_rate(&self) -> f32 {
        self.num_hits() as f32 / self.num_accesses() as f32
    }

    #[must_use]
    pub fn write_hit_rate(&self) -> f32 {
        self.num_write_hits() as f32 / self.num_writes() as f32
    }

    #[must_use]
    pub fn read_hit_rate(&self) -> f32 {
        self.num_read_hits() as f32 / self.num_reads() as f32
    }

    #[must_use]
    pub fn global_write_hit_rate(&self) -> f32 {
        self.num_global_write_hits() as f32 / self.num_global_writes() as f32
    }

    #[must_use]
    pub fn global_read_hit_rate(&self) -> f32 {
        self.num_global_read_hits() as f32 / self.num_global_reads() as f32
    }

    #[must_use]
    pub fn num_pending_hits(&self) -> usize {
        self.inner
            .iter()
            .filter(|((_, access), _)| access.is_pending_hit())
            .map(|(_, count)| count)
            .sum()
    }

    #[must_use]
    pub fn num_reservation_fails(&self) -> usize {
        self.inner
            .iter()
            .filter(|((_, access), _)| access.is_reservation_fail())
            .map(|(_, count)| count)
            .sum()
    }

    // #[inline]
    pub fn inc(
        &mut self,
        alloc_id: Option<usize>,
        kind: impl Into<AccessKind>,
        status: impl Into<AccessStat>,
        count: usize,
    ) {
        let kind = kind.into();
        let status = status.into();
        let access_stat = AccessStatus((kind, status));
        // println!("inc access stat: {access_stat}");
        *self.inner.entry((alloc_id, access_stat)).or_insert(0) += count;
    }
}

#[allow(clippy::module_name_repetitions)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerCache {
    pub kernel_info: super::KernelInfo,
    pub inner: Box<[Cache]>,
}

impl std::ops::Deref for PerCache {
    type Target = Box<[Cache]>;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::ops::DerefMut for PerCache {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl std::ops::AddAssign for PerCache {
    fn add_assign(&mut self, other: Self) {
        for (cache, other_cache) in self.inner.iter_mut().zip(other.iter()) {
            *cache += other_cache.clone();
        }
    }
}

impl<T: Into<Cache>> FromIterator<T> for PerCache {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self {
            inner: iter.into_iter().map(Into::into).collect(),
            kernel_info: super::KernelInfo::default(),
        }
    }
}

impl PerCache {
    #[must_use]
    pub fn from(size: usize) -> Self {
        Self {
            kernel_info: super::KernelInfo::default(),
            inner: utils::box_slice![Cache::default(); size],
        }
    }

    #[must_use]
    pub fn new(size: usize) -> Self {
        Self {
            kernel_info: super::KernelInfo::default(),
            inner: utils::box_slice![Cache::default(); size],
        }
    }

    #[must_use]
    pub fn into_inner(self) -> Box<[Cache]> {
        self.inner
    }

    #[must_use]
    pub fn into_csv_rows(self, full: bool) -> Vec<CsvRow> {
        let mut rows: Vec<_> = Vec::new();
        for (cache_id, cache) in self.inner.iter().cloned().enumerate() {
            for ((allocation_id, access), num_accesses) in cache.inner {
                let need_row = rows.is_empty();
                if !full && !need_row && num_accesses < 1 {
                    continue;
                }
                let (access_kind, access_status) = access.0;
                rows.push(CsvRow {
                    kernel_name: self.kernel_info.name.clone(),
                    kernel_name_mangled: self.kernel_info.mangled_name.clone(),
                    kernel_launch_id: self.kernel_info.launch_id,
                    cache_id,
                    allocation_id,
                    access_kind,
                    is_write: access_kind.is_write(),
                    access_status,
                    num_accesses,
                })
            }
            // rows.extend(
            //     cache
            //         .inner
            //         .into_iter()
            //         // .sort_by_key(|(key, _)| *key)
            //         .map(|((allocation_id, access), num_accesses)| {
            //             // let need_row =
            //             let (access_kind, access_status) = access.0;
            //             CsvRow {
            //                 kernel_name: self.kernel_info.name.clone(),
            //                 kernel_name_mangled: self.kernel_info.mangled_name.clone(),
            //                 kernel_launch_id: self.kernel_info.launch_id,
            //                 cache_id,
            //                 allocation_id,
            //                 access_kind,
            //                 is_write: access_kind.is_write(),
            //                 access_status,
            //                 num_accesses,
            //             }
            //         }),
            // );
        }
        rows
    }

    pub fn shave(&mut self) {
        for stats in &mut *self.inner {
            stats.shave();
        }
    }

    #[must_use]
    pub fn total_accesses(&self) -> usize {
        self.reduce().num_accesses()
    }

    #[must_use]
    pub fn reduce(&self) -> Cache {
        let mut out = Cache::default();
        for stats in &*self.inner {
            out += stats.clone();
        }
        out
    }

    #[must_use]
    pub fn merge_allocations(self) -> PerCache {
        PerCache {
            inner: self
                .inner
                .iter()
                .cloned()
                .map(Cache::merge_allocations)
                .collect(),
            ..self
        }
    }
}
