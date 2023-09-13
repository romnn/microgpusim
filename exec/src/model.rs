// pub use trace_model::*;

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemorySpace {
    Local,
    Shared,
    Constant,
    Texture,
    Global,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub enum MemAccessKind {
    Load,
    Store,
}

/// Warp instruction
#[derive(Debug, Clone, Ord, PartialOrd, Hash)]
pub struct MemInstruction {
    pub mem_space: MemorySpace,
    pub kind: MemAccessKind,
    pub addr: u64,
    pub size: u32,
}

impl Eq for MemInstruction {}

impl PartialEq for MemInstruction {
    fn eq(&self, other: &Self) -> bool {
        (self.mem_space, self.kind).eq(&(other.mem_space, other.kind))
    }
}

/// Warp instruction
#[derive(Debug, Clone, Ord, PartialOrd, Hash)]
pub enum ThreadInstruction {
    Access(MemInstruction),
    Inactive,
}

impl Eq for ThreadInstruction {}

impl PartialEq for ThreadInstruction {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ThreadInstruction::Inactive, _) => true,
            (_, ThreadInstruction::Inactive) => true,
            (ThreadInstruction::Access(a), ThreadInstruction::Access(b)) => a.eq(&b),
        }
    }
}
