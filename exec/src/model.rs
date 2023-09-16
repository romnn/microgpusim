pub use trace_model::Dim;

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

impl std::fmt::Display for MemInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}[{:?}]@{}", self.kind, self.mem_space, self.addr)
    }
}

impl Eq for MemInstruction {}

impl PartialEq for MemInstruction {
    fn eq(&self, other: &Self) -> bool {
        (self.mem_space, self.kind).eq(&(other.mem_space, other.kind))
    }
}

/// Warp instruction
#[derive(Debug, Clone, Hash, PartialOrd, Ord)]
pub enum ThreadInstruction {
    Access(MemInstruction),
    Nop,
    Branch(usize),
    TookBranch(usize),
    Reconverge(usize),
}

impl From<MemInstruction> for ThreadInstruction {
    fn from(inst: MemInstruction) -> Self {
        Self::Access(inst)
    }
}

impl std::fmt::Display for ThreadInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Access(inst) => write!(f, "Access({})", inst),
            other => std::fmt::Debug::fmt(other, f),
            // Self::Nop,
            // Self::Branch(usize),
            // Self::TookBranch(usize),
            // Self::Reconvergence(usize),
        }
        // let num_bytes = self.end_addr.map(|end| end - self.start_addr);
        // let num_f32 = num_bytes.map(|num_bytes| num_bytes / 4);
        // f.debug_struct("Allocation")
        //     .field("id", &self.id)
        //     .field("name", &self.name)
        //     .field("start_addr", &self.start_addr)
        //     .field("end_addr", &self.end_addr)
        //     .field(
        //         "size",
        //         &num_bytes.map(|num_bytes| human_bytes::human_bytes(num_bytes as f64)),
        //     )
        //     .field("num_f32", &num_f32)
        //     .finish()
    }
}

impl Eq for ThreadInstruction {}

impl PartialEq for ThreadInstruction {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ThreadInstruction::Nop, _) => true,
            (_, ThreadInstruction::Nop) => true,
            (ThreadInstruction::Access(a), ThreadInstruction::Access(b)) => a.eq(&b),
            (ThreadInstruction::Branch(a), ThreadInstruction::Branch(b)) => a.eq(&b),
            (ThreadInstruction::TookBranch(a), ThreadInstruction::TookBranch(b)) => a.eq(&b),
            (ThreadInstruction::Reconverge(a), ThreadInstruction::Reconverge(b)) => a.eq(&b),
            (_, _) => false,
        }
    }
}
