use super::mem::AccessKind;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use strum::IntoEnumIterator;

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
#[repr(transparent)]
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

pub type CsvRow = (Option<usize>, Access, usize);

#[derive(Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Cache {
    pub accesses: HashMap<(Option<usize>, Access), usize>,
}

impl Cache {
    #[must_use]
    pub fn flatten(self) -> Vec<CsvRow> {
        let mut flattened: Vec<_> = self
            .accesses
            .into_iter()
            .map(|((alloc_id, access), count)| (alloc_id, access, count))
            .collect();
        flattened.sort_by_key(|(alloc_id, access, _)| (*alloc_id, *access));
        flattened
    }
}

impl std::ops::AddAssign for Cache {
    fn add_assign(&mut self, other: Self) {
        for (k, v) in other.accesses {
            *self.accesses.entry(k).or_insert(0) += v;
        }
    }
}

impl Default for Cache {
    fn default() -> Self {
        let accesses = HashMap::new();
        // for access_kind in AccessKind::iter() {
        //     for status in RequestStatus::iter() {
        //         accesses.insert(Access((access_kind, AccessStat::Status(status))), 0);
        //     }
        //     for failure in ReservationFailure::iter() {
        //         accesses.insert(
        //             Access((access_kind, AccessStat::ReservationFailure(failure))),
        //             0,
        //         );
        //     }
        // }
        Self { accesses }
    }
}

impl std::fmt::Debug for Cache {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut accesses: Vec<_> = self
            .accesses
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
    pub fn shave(&mut self) {
        self.accesses.retain(|_, v| *v > 0);
    }

    pub fn merge_allocations(self) -> Cache {
        let mut accesses = HashMap::new();
        for ((_, access), count) in self.accesses.into_iter() {
            *accesses.entry((None, access)).or_insert(0) += count;
        }
        Cache { accesses }
    }

    #[must_use]
    pub fn num_accesses(&self, access: &Access) -> usize {
        self.accesses
            .iter()
            .filter(|((_, acc), _)| acc == access)
            .map(|(_, count)| count)
            .sum()
    }

    #[must_use]
    pub fn total_accesses(&self) -> usize {
        self.accesses
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
        for ((alloc_id, access), accesses) in &self.accesses {
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
            .accesses
            .entry((alloc_id, Access((kind.into(), access.into()))))
            .or_insert(0) += count;
    }
}

pub type PerCacheCsvRow = (usize, CsvRow);

#[allow(clippy::module_name_repetitions)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerCache(pub Box<[Cache]>);

impl std::ops::Deref for PerCache {
    type Target = Box<[Cache]>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for PerCache {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl std::ops::AddAssign for PerCache {
    fn add_assign(&mut self, other: Self) {
        for (cache, other_cache) in self.0.iter_mut().zip(other.iter()) {
            *cache += other_cache.clone()
        }
    }
}

impl<T: Into<Cache>> FromIterator<T> for PerCache {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(iter.into_iter().map(Into::into).collect())
    }
}

impl PerCache {
    #[must_use]
    pub fn from(size: usize) -> Self {
        Self(utils::box_slice![Cache::default(); size])
    }

    #[must_use]
    pub fn new(size: usize) -> Self {
        Self(utils::box_slice![Cache::default(); size])
    }

    #[must_use]
    pub fn into_inner(self) -> Box<[Cache]> {
        self.0
    }

    #[must_use]
    pub fn flatten(self) -> Vec<PerCacheCsvRow> {
        let mut flattened: Vec<_> = self
            .into_inner()
            .iter()
            .cloned()
            .enumerate()
            .flat_map(|(id, cache)| {
                cache
                    .flatten()
                    .into_iter()
                    .map(move |cache_row| (id, cache_row))
            })
            .collect();
        flattened.sort_by_key(|(id, _)| *id);
        flattened
    }

    pub fn shave(&mut self) {
        for stats in &mut *self.0 {
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
        for stats in &*self.0 {
            out += stats.clone();
        }
        out
    }

    pub fn merge_allocations(self) -> PerCache {
        PerCache(
            self.0
                .to_vec()
                .into_iter()
                .map(Cache::merge_allocations)
                .collect(),
        )
    }
}
