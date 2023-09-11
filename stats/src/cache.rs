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
pub struct Access(pub (AccessKind, AccessStat));

impl std::fmt::Display for Access {
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

/// Per kernel cache statistics.
///
/// Stats at index `i` correspond to the kernel with launch id `i`.
#[derive(Clone, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerKernel {
    pub inner: Vec<Cache>,
}

impl AsRef<Vec<Cache>> for PerKernel {
    fn as_ref(&self) -> &Vec<Cache> {
        &self.inner
    }
}

impl PerKernel {
    #[inline]
    pub fn get_mut(&mut self, idx: usize) -> &mut Cache {
        self.inner.resize_with(idx + 1, || Cache::default());
        &mut self.inner[idx]
    }

    #[inline]
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
    pub access_stat: AccessStat,
    pub num_accesses: usize,
}

#[derive(Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Cache {
    pub inner: HashMap<(Option<usize>, Access), usize>,
}

impl Default for Cache {
    fn default() -> Self {
        use strum::IntoEnumIterator;
        let mut inner = HashMap::new();
        for access_kind in AccessKind::iter() {
            for status in RequestStatus::iter() {
                inner.insert((None, Access((access_kind, AccessStat::Status(status)))), 0);
            }
            for failure in ReservationFailure::iter() {
                inner.insert(
                    (
                        None,
                        Access((access_kind, AccessStat::ReservationFailure(failure))),
                    ),
                    0,
                );
            }
        }
        Self { inner }
    }
}

impl AsRef<HashMap<(Option<usize>, Access), usize>> for Cache {
    fn as_ref(&self) -> &HashMap<(Option<usize>, Access), usize> {
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
            .map(|((alloc_id, access), count)| {
                // let key = match access_stat {
                //     AccessStat::Status(status) => {
                //         format!("{access_kind:?}[{status:?}]")
                //     }
                //     AccessStat::ReservationFailure(failure) => {
                //         format!("{access_kind:?}[{failure:?}]")
                //     }
                // };
                let key = match alloc_id {
                    None => access.to_string(),
                    Some(id) => format!("{id}@{access}"),
                };
                (key, count)
            })
            .collect();
        accesses.sort_by_key(|(key, _)| key.clone());

        let mut out = f.debug_struct("CacheStats");
        for (access, count) in accesses {
            out.field(&access, count);
        }
        out.finish_non_exhaustive()
    }
}

impl Cache {
    pub fn new(inner: HashMap<(Option<usize>, Access), usize>) -> Self {
        Self { inner }
    }

    pub fn shave(&mut self) {
        self.inner.retain(|_, v| *v > 0);
    }

    pub fn merge_allocations(self) -> Cache {
        let mut inner = HashMap::new();
        for ((_, access), count) in self.inner.into_iter() {
            *inner.entry((None, access)).or_insert(0) += count;
        }
        Cache { inner, ..self }
    }

    #[must_use]
    pub fn num_accesses(&self, access: &Access) -> usize {
        self.inner
            .iter()
            .filter(|((_, acc), _)| acc == access)
            .map(|(_, count)| count)
            .sum()
    }

    #[must_use]
    pub fn total_accesses(&self) -> usize {
        self.inner
            .iter()
            .filter_map(|((_, access), count)| match access {
                Access((_kind, AccessStat::Status(RequestStatus::HIT | RequestStatus::MISS))) => {
                    Some(count)
                }
                _ => None,
            })
            .sum()
    }

    #[deprecated]
    pub fn sub_stats(&self) {
        let mut total_accesses = 0;
        let mut total_misses = 0;
        let mut total_pending_hits = 0;
        let mut total_reservation_fails = 0;
        for ((_alloc_id, access), accesses) in &self.inner {
            let Access((_access_kind, status)) = access;

            if let AccessStat::Status(
                RequestStatus::HIT
                | RequestStatus::MISS
                | RequestStatus::SECTOR_MISS
                | RequestStatus::HIT_RESERVED,
            ) = status
            {
                total_accesses += accesses;
            }

            match status {
                AccessStat::Status(RequestStatus::MISS | RequestStatus::SECTOR_MISS) => {
                    total_misses += accesses;
                }
                AccessStat::Status(RequestStatus::HIT_RESERVED) => {
                    total_pending_hits += accesses;
                }
                AccessStat::Status(RequestStatus::RESERVATION_FAIL) => {
                    total_reservation_fails += accesses;
                }
                _ => {}
            }
        }
    }

    #[inline]
    pub fn inc(
        &mut self,
        alloc_id: Option<usize>,
        kind: impl Into<AccessKind>,
        access: impl Into<AccessStat>,
        count: usize,
    ) {
        *self
            .inner
            .entry((alloc_id, Access((kind.into(), access.into()))))
            .or_insert(0) += count;
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
            *cache += other_cache.clone()
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
    pub fn into_csv_rows(self) -> Vec<CsvRow> {
        let mut rows: Vec<_> = Vec::new();
        for (cache_id, cache) in self.inner.into_iter().cloned().enumerate() {
            rows.extend(
                cache
                    .inner
                    .into_iter()
                    // .sort_by_key(|(key, _)| *key)
                    .map(|((allocation_id, access), num_accesses)| {
                        let (access_kind, access_stat) = access.0;
                        CsvRow {
                            kernel_name: self.kernel_info.name.clone(),
                            kernel_name_mangled: self.kernel_info.mangled_name.clone(),
                            kernel_launch_id: self.kernel_info.launch_id,
                            cache_id,
                            allocation_id,
                            access_kind,
                            is_write: access_kind.is_write(),
                            access_stat,
                            num_accesses,
                        }
                    }),
            );
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
        self.reduce().total_accesses()
    }

    #[must_use]
    pub fn reduce(&self) -> Cache {
        let mut out = Cache::default();
        for stats in &*self.inner {
            out += stats.clone();
        }
        out
    }

    pub fn merge_allocations(self) -> PerCache {
        PerCache {
            inner: self
                .inner
                .to_vec()
                .into_iter()
                .map(Cache::merge_allocations)
                .collect(),
            ..self
        }
    }
}
