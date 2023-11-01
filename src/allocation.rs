use super::address;
use crate::sync::{Arc, RwLock};

pub type Ref = Arc<RwLock<Allocations>>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Allocation {
    pub id: usize,
    pub name: Option<String>,
    pub start_addr: address,
    pub end_addr: Option<address>,
}

impl Allocation {
    pub fn num_bytes(&self) -> u64 {
        self.end_addr
            .and_then(|end_addr| end_addr.checked_sub(self.start_addr))
            .unwrap_or(0)
    }
}

impl std::cmp::Ord for Allocation {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

impl std::cmp::PartialOrd for Allocation {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl std::fmt::Display for Allocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let num_bytes = self.end_addr.map(|end| end - self.start_addr);
        let num_f32 = num_bytes.map(|num_bytes| num_bytes / 4);
        f.debug_struct("Allocation")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("start_addr", &self.start_addr)
            .field("end_addr", &self.end_addr)
            .field(
                "size",
                &num_bytes.map(|num_bytes| human_bytes::human_bytes(num_bytes as f64)),
            )
            .field("num_f32", &num_f32)
            .finish()
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct Allocations(rangemap::RangeMap<address, Allocation>);

impl std::ops::Deref for Allocations {
    type Target = rangemap::RangeMap<address, Allocation>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Allocations {
    pub fn insert(&mut self, range: std::ops::Range<address>, name: Option<String>) {
        // check for intersections
        assert!(
            !self.0.overlaps(&range),
            "overlapping memory allocation {:?}",
            &range
        );
        let id = self.0.len() + 1; // zero is reserved for instructions
        let start_addr = range.start;
        let end_addr = Some(range.end);
        self.0.insert(
            range,
            Allocation {
                id,
                name,
                start_addr,
                end_addr,
            },
        );
    }
}
