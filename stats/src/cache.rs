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

pub type CacheCsvRow = ((AccessKind, AccessStat), usize);

#[derive(Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Cache {
    pub accesses: HashMap<(AccessKind, AccessStat), usize>,
}

impl Cache {
    pub fn flatten(self) -> Vec<CacheCsvRow> {
        let mut flattened: Vec<_> = self.accesses.into_iter().collect();
        flattened.sort_by_key(|(access, _)| *access);
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
        let mut accesses = HashMap::new();
        for access_kind in AccessKind::iter() {
            for status in RequestStatus::iter() {
                accesses.insert((access_kind, AccessStat::Status(status)), 0);
            }
            for failure in ReservationFailure::iter() {
                accesses.insert((access_kind, AccessStat::ReservationFailure(failure)), 0);
            }
        }
        Self { accesses }
    }
}

impl std::fmt::Debug for Cache {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut accesses: Vec<_> = self
            .accesses
            .iter()
            .filter(|(_, &count)| count > 0)
            .map(|((access_kind, access_stat), count)| {
                let key = match access_stat {
                    AccessStat::Status(status) => {
                        format!("{access_kind:?}[{status:?}]")
                    }
                    AccessStat::ReservationFailure(failure) => {
                        format!("{access_kind:?}[{failure:?}]")
                    }
                };
                (key, count)
            })
            .collect();
        accesses.sort_by_key(|(key, _)| key.clone());

        let mut out = f.debug_struct("CacheStats");
        for (key, count) in accesses {
            out.field(&key, count);
        }
        out.finish_non_exhaustive()
    }
}

impl Cache {
    pub fn shave(&mut self) {
        self.accesses.retain(|_, v| *v > 0);
    }

    #[must_use]
    pub fn total_accesses(&self) -> usize {
        self.accesses.values().sum()
    }

    #[deprecated]
    pub fn sub_stats(&self) {
        let mut total_accesses = 0;
        let mut total_misses = 0;
        let mut total_pending_hits = 0;
        let mut total_reservation_fails = 0;
        for ((_access_kind, status), accesses) in &self.accesses {
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
        kind: impl Into<AccessKind>,
        access: impl Into<AccessStat>,
        count: usize,
    ) {
        *self
            .accesses
            .entry((kind.into(), access.into()))
            .or_insert(0) += count;
    }
}

pub type PerCacheCsvRow = (usize, CacheCsvRow);

#[allow(clippy::module_name_repetitions)]
#[derive(Clone, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerCache(pub HashMap<usize, Cache>);

impl PerCache {
    pub fn into_inner(self) -> HashMap<usize, Cache> {
        self.0
    }

    pub fn flatten(self) -> Vec<PerCacheCsvRow> {
        let mut flattened: Vec<_> = self
            .into_inner()
            .into_iter()
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
        for stats in self.values_mut() {
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
        for stats in self.0.values() {
            out += stats.clone();
        }
        out
    }
}

impl std::ops::Deref for PerCache {
    type Target = HashMap<usize, Cache>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for PerCache {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
