use super::{
    address,
    instruction::{MemorySpace, WarpInstruction},
    mcu, mem_sub_partition,
};
use bitvec::BitArr;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::sync::atomic;

pub const READ_PACKET_SIZE: u8 = 8;

// bytes: 6 address, 2 miscelaneous.
pub const WRITE_PACKET_SIZE: u8 = 8;

pub const WRITE_MASK_SIZE: u8 = 8;

pub type ByteMask = BitArr!(for mem_sub_partition::MAX_MEMORY_ACCESS_SIZE as usize);
pub type SectorMask = BitArr!(for mem_sub_partition::SECTOR_CHUNK_SIZE as usize, in u8);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Kind {
    READ_REQUEST = 0,
    WRITE_REQUEST,
    READ_REPLY,
    WRITE_ACK,
}

// impl From<Kind> for stats::mem::RequestKind {
// }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Status {
    INITIALIZED,
    IN_L1I_MISS_QUEUE,
    IN_L1D_MISS_QUEUE,
    IN_L1T_MISS_QUEUE,
    IN_L1C_MISS_QUEUE,
    IN_L1TLB_MISS_QUEUE,
    IN_VM_MANAGER_QUEUE,
    IN_ICNT_TO_MEM,
    IN_PARTITION_ROP_DELAY,
    IN_PARTITION_ICNT_TO_L2_QUEUE,
    IN_PARTITION_L2_TO_DRAM_QUEUE,
    IN_PARTITION_DRAM_LATENCY_QUEUE,
    IN_PARTITION_L2_MISS_QUEUE,
    IN_PARTITION_MC_INTERFACE_QUEUE,
    IN_PARTITION_MC_INPUT_QUEUE,
    IN_PARTITION_MC_BANK_ARB_QUEUE,
    IN_PARTITION_DRAM,
    IN_PARTITION_MC_RETURNQ,
    IN_PARTITION_DRAM_TO_L2_QUEUE,
    IN_PARTITION_L2_FILL_QUEUE,
    IN_PARTITION_L2_TO_ICNT_QUEUE,
    IN_ICNT_TO_SHADER,
    IN_CLUSTER_TO_SHADER_QUEUE,
    IN_SHADER_LDST_RESPONSE_FIFO,
    IN_SHADER_FETCHED,
    IN_SHADER_L1T_ROB,
    DELETED,
    NUM_MEM_REQ_STAT,
}

pub mod access {
    use serde::{Deserialize, Serialize};
    use trace_model::ToBitString;

    #[derive(
        Debug,
        strum::EnumIter,
        strum::EnumCount,
        Clone,
        Copy,
        PartialEq,
        Eq,
        Hash,
        PartialOrd,
        Ord,
        Serialize,
        Deserialize,
    )]
    pub enum Kind {
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

    impl Kind {
        pub fn memory_space(self) -> Option<crate::instruction::MemorySpace> {
            self.into()
        }

        pub fn base_addr(self) -> Option<crate::address> {
            Some(self.memory_space()?.base_addr())
        }
    }

    impl From<Kind> for Option<crate::instruction::MemorySpace> {
        fn from(kind: Kind) -> Self {
            match kind {
                Kind::GLOBAL_ACC_R | Kind::GLOBAL_ACC_W => {
                    Some(crate::instruction::MemorySpace::Global)
                }
                Kind::LOCAL_ACC_R | Kind::LOCAL_ACC_W => {
                    Some(crate::instruction::MemorySpace::Local)
                }
                Kind::CONST_ACC_R => Some(crate::instruction::MemorySpace::Constant),
                Kind::TEXTURE_ACC_R => Some(crate::instruction::MemorySpace::Texture),
                Kind::L1_WRBK_ACC
                | Kind::L2_WRBK_ACC
                | Kind::INST_ACC_R
                | Kind::L1_WR_ALLOC_R
                | Kind::L2_WR_ALLOC_R => None,
            }
        }
    }

    impl From<Kind> for stats::mem::AccessKind {
        fn from(kind: Kind) -> Self {
            match kind {
                Kind::GLOBAL_ACC_R => Self::GLOBAL_ACC_R,
                Kind::LOCAL_ACC_R => Self::LOCAL_ACC_R,
                Kind::CONST_ACC_R => Self::CONST_ACC_R,
                Kind::TEXTURE_ACC_R => Self::TEXTURE_ACC_R,
                Kind::GLOBAL_ACC_W => Self::GLOBAL_ACC_W,
                Kind::LOCAL_ACC_W => Self::LOCAL_ACC_W,
                Kind::L1_WRBK_ACC => Self::L1_WRBK_ACC,
                Kind::L2_WRBK_ACC => Self::L2_WRBK_ACC,
                Kind::INST_ACC_R => Self::INST_ACC_R,
                Kind::L1_WR_ALLOC_R => Self::L1_WR_ALLOC_R,
                Kind::L2_WR_ALLOC_R => Self::L2_WR_ALLOC_R,
            }
        }
    }

    impl Kind {
        #[must_use]
        // #[inline]
        pub fn is_global(&self) -> bool {
            matches!(self, Kind::GLOBAL_ACC_R | Kind::GLOBAL_ACC_W)
        }

        #[must_use]
        // #[inline]
        pub fn is_local(&self) -> bool {
            matches!(self, Kind::LOCAL_ACC_R | Kind::LOCAL_ACC_W)
        }

        #[must_use]
        // #[inline]
        pub fn is_texture(&self) -> bool {
            *self == Kind::TEXTURE_ACC_R
        }

        #[must_use]
        // #[inline]
        pub fn is_const(&self) -> bool {
            *self == Kind::CONST_ACC_R
        }

        #[must_use]
        // #[inline]
        pub fn is_write(&self) -> bool {
            match self {
                Kind::GLOBAL_ACC_R
                | Kind::LOCAL_ACC_R
                | Kind::CONST_ACC_R
                | Kind::TEXTURE_ACC_R
                | Kind::INST_ACC_R
                | Kind::L1_WR_ALLOC_R
                | Kind::L2_WR_ALLOC_R => false,
                Kind::GLOBAL_ACC_W | Kind::LOCAL_ACC_W | Kind::L1_WRBK_ACC | Kind::L2_WRBK_ACC => {
                    true
                }
            }
        }
    }

    #[allow(clippy::module_name_repetitions)]
    #[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub struct MemAccess {
        /// Requested address.
        pub addr: super::address,
        /// The allocation that this access corresponds to.
        pub allocation: Option<crate::allocation::Allocation>,
        pub kernel_launch_id: Option<usize>,
        // TODO: is_write could be computed using kind.is_write()
        pub is_write: bool,
        /// Requested number of bytes.
        pub req_size_bytes: u32,
        /// Access kind.
        pub kind: Kind,
        /// Warp active mask of the warp that issued this access.
        pub warp_active_mask: crate::warp::ActiveMask,
        /// Byte mask.
        pub byte_mask: super::ByteMask,
        /// Sector mask of this access.
        pub sector_mask: super::SectorMask,
    }

    impl std::fmt::Debug for MemAccess {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.debug_struct("MemAccess")
                .field("addr", &self.addr)
                .field("relative_addr", &self.relative_addr())
                .field("allocation", &self.allocation)
                .field("kernel_launch_id", &self.kernel_launch_id)
                .field("kind", &self.kind)
                .field("req_size_bytes", &self.req_size_bytes)
                .field("is_write", &self.is_write)
                .field("active_mask", &self.warp_active_mask.to_bit_string())
                .field("byte_mask", &self.byte_mask.to_bit_string())
                .field("sector_mask", &self.sector_mask.to_bit_string())
                .finish()
        }
    }

    impl std::fmt::Display for MemAccess {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            // write!(f, "Access({:?}@", self.kind)?;
            write!(f, "{:?}@", self.kind)?;
            if let Some(ref alloc) = self.allocation {
                write!(f, "{}+{}", &alloc.id, self.addr - alloc.start_addr)
            } else {
                write!(f, "{}", &self.addr)
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct Builder {
        pub kind: Kind,
        pub addr: crate::address,
        pub kernel_launch_id: Option<usize>,
        pub allocation: Option<crate::allocation::Allocation>,
        pub req_size_bytes: u32,
        // TODO: is_write could be computed using kind.is_write()
        pub is_write: bool,
        pub warp_active_mask: crate::warp::ActiveMask,
        pub byte_mask: super::ByteMask,
        pub sector_mask: super::SectorMask,
    }

    impl Builder {
        #[must_use]
        pub fn build(self) -> MemAccess {
            if let Some(ref alloc) = self.allocation {
                debug_assert!(alloc.start_addr <= self.addr);
            }
            assert_eq!(self.kind.is_write(), self.is_write);
            MemAccess {
                addr: self.addr,
                allocation: self.allocation,
                kernel_launch_id: self.kernel_launch_id,
                is_write: self.is_write,
                req_size_bytes: self.req_size_bytes,
                kind: self.kind,
                warp_active_mask: self.warp_active_mask,
                byte_mask: self.byte_mask,
                sector_mask: self.sector_mask,
            }
        }
    }

    impl MemAccess {
        // #[inline]
        #[must_use]
        pub fn relative_addr(&self) -> Option<super::address> {
            self.allocation
                .as_ref()
                .map(|alloc| alloc.start_addr)
                .and_then(|start| self.addr.checked_sub(start))
        }

        // #[inline]
        #[must_use]
        pub fn allocation_id(&self) -> Option<usize> {
            self.allocation.as_ref().map(|alloc| alloc.id)
        }

        #[must_use]
        // #[inline]
        pub fn control_size(&self) -> u32 {
            if self.is_write {
                u32::from(super::WRITE_PACKET_SIZE)
            } else {
                u32::from(super::READ_PACKET_SIZE)
            }
        }

        // #[inline]
        pub fn kernel_launch_id(&self) -> Option<usize> {
            self.kernel_launch_id
        }

        #[must_use]
        // #[inline]
        pub fn is_write(&self) -> bool {
            self.kind.is_write()
        }

        #[must_use]
        // #[inline]
        pub fn data_size(&self) -> u32 {
            self.req_size_bytes
        }

        #[must_use]
        // #[inline]
        pub fn size(&self) -> u32 {
            self.data_size() + self.control_size()
        }
    }
}

#[derive(Clone, Debug, PartialOrd, Ord)]
pub struct MemFetch {
    pub uid: u64,
    pub access: access::MemAccess,
    pub instr: Option<WarpInstruction>,
    pub physical_addr: mcu::PhysicalAddress,
    pub partition_addr: address,
    pub kind: Kind,
    pub warp_id: usize,
    pub core_id: Option<usize>,
    pub cluster_id: Option<usize>,

    pub inject_cycle: Option<u64>,
    pub return_cycle: Option<u64>,

    pub status: Status,
    pub last_status_change: Option<u64>,

    // this pointer is set up when a request is divided into
    // sector requests at L2 cache (if the req size > L2 sector
    // size), so the pointer refers to the original request
    pub original_fetch: Option<Box<MemFetch>>,
    // pub original_fetch: Option<Box<MemFetch>>,

    // this fetch refers to the original write req,
    // when fetch-on-write policy is used
    pub original_write_fetch: Option<Box<MemFetch>>,

    pub latency: u64,
}

impl std::fmt::Display for MemFetch {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let addr = self.addr();
        write!(f, "{:?}({:?}", self.kind, self.access_kind())?;
        if let Some(ref alloc) = self.access.allocation {
            write!(f, "@{}+{})", alloc.id, addr - alloc.start_addr)
        } else {
            write!(f, "@{addr})")
        }
    }
}

impl Eq for MemFetch {}

impl PartialEq for MemFetch {
    fn eq(&self, other: &Self) -> bool {
        self.uid == other.uid
        // self.access == other.access
        //     && self.tlx_addr == other.tlx_addr
        //     && self.partition_addr == other.partition_addr
        //     && self.control_size == other.control_size
        //     && self.data_size == other.data_size
        //     && self.warp_id == other.warp_id
        //     && self.core_id == other.core_id
        //     && self.cluster_id == other.cluster_id
    }
}

impl std::hash::Hash for MemFetch {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.uid.hash(state);
        // self.access.hash(state);
        // self.tlx_addr.hash(state);
        // self.partition_addr.hash(state);
        // self.control_size.hash(state);
        // self.data_size.hash(state);
        // self.warp_id.hash(state);
        // self.core_id.hash(state);
        // self.cluster_id.hash(state);
    }
}

impl From<&MemFetch> for stats::mem::Access {
    fn from(fetch: &MemFetch) -> Self {
        stats::mem::Access {
            addr: fetch.access.addr,
            relative_addr: fetch.access.relative_addr(),
            allocation_id: fetch.access.allocation_id(),
            // /// Requested address.
            // pub addr: super::address,
            // /// The allocation that this access corresponds to.
            // pub allocation: Option<crate::allocation::Allocation>,
            kernel_launch_id: fetch.access.kernel_launch_id(),
            // // TODO: is_write could be computed using kind.is_write()
            // pub is_write: bool,
            // /// Requested number of bytes.
            requested_bytes: fetch.access.req_size_bytes,
            // /// Access kind.
            kind: fetch.access.kind.into(),
            // /// Warp active mask of the warp that issued this access.
            // pub warp_active_mask: crate::warp::ActiveMask,
            // /// Byte mask.
            // byte_mask: fetch.access.byte_mask,
            // sector_mask: fetch.access.sector_mask,

            // uid: u64,
            // access: access::MemAccess,
            // access: access::MemAccess,
            // instr: Option<WarpInstruction>,
            physical_addr: fetch.physical_addr.clone().into(),
            partition_addr: fetch.partition_addr,
            // kind: fetch.kind.into(),
            warp_id: fetch.warp_id,
            core_id: fetch.core_id,
            cluster_id: fetch.cluster_id,
            inject_cycle: fetch.inject_cycle,
            return_cycle: fetch.return_cycle,
            // status: Status,
            // last_status_change: Option<u64>,
            // original_fetch: Option<Box<MemFetch>>,
            // original_write_fetch: Option<Box<MemFetch>>,
            // latency: u64,
        }
    }
}

static MEM_FETCH_UID: Lazy<atomic::AtomicU64> = Lazy::new(|| atomic::AtomicU64::new(0));

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Builder {
    pub instr: Option<WarpInstruction>,
    pub access: access::MemAccess,
    pub warp_id: usize,
    pub core_id: Option<usize>,
    pub cluster_id: Option<usize>,
    pub physical_addr: mcu::PhysicalAddress,
    pub partition_addr: address,
}

/// Generate a unique ID that can be used to identify fetch requests
pub fn generate_uid() -> u64 {
    MEM_FETCH_UID.fetch_add(1, atomic::Ordering::SeqCst)
}

impl Builder {
    #[must_use]
    pub fn build(self) -> MemFetch {
        let kind = if self.access.is_write {
            Kind::WRITE_REQUEST
        } else {
            Kind::READ_REQUEST
        };
        MemFetch {
            uid: generate_uid(),
            access: self.access,
            instr: self.instr,
            warp_id: self.warp_id,
            core_id: self.core_id,
            cluster_id: self.cluster_id,
            physical_addr: self.physical_addr,
            partition_addr: self.partition_addr,
            kind,
            status: Status::INITIALIZED,
            inject_cycle: None,
            return_cycle: None,
            last_status_change: None,
            original_fetch: None,
            original_write_fetch: None,
            latency: 0,
        }
    }
}

impl From<Builder> for MemFetch {
    fn from(builder: Builder) -> Self {
        builder.build()
    }
}

impl MemFetch {
    // #[inline]
    #[must_use]
    pub fn allocation_id(&self) -> Option<usize> {
        self.access.allocation_id()
    }

    // #[inline]
    pub fn is_atomic(&self) -> bool {
        self.instr
            .as_ref()
            .map_or(false, WarpInstruction::is_atomic)
    }

    #[must_use]
    // #[inline]
    pub fn is_texture(&self) -> bool {
        self.instr.as_ref().map_or(false, |inst| {
            inst.memory_space == Some(MemorySpace::Texture)
        })
    }

    #[must_use]
    // #[inline]
    pub fn base_addr(&self) -> Option<address> {
        self.instr
            .as_ref()
            .and_then(|inst| inst.memory_space)
            .map(|space| space.base_addr())
    }

    #[must_use]
    // #[inline]
    pub fn packet_size(&self) -> u32 {
        if self.is_write() || self.is_atomic() {
            self.size()
        } else {
            u32::from(READ_PACKET_SIZE)
        }
    }

    // #[inline]
    pub fn kernel_launch_id(&self) -> Option<usize> {
        self.access.kernel_launch_id()
    }

    #[must_use]
    // #[inline]
    pub fn is_write(&self) -> bool {
        self.access.is_write
    }

    /// Get the sector address
    ///
    /// This address is at the granularity that is used by the memory.
    /// E.g. requesting 4B will result in a request for a 32B sector (sector-aligned).
    #[must_use]
    // #[inline]
    pub fn addr(&self) -> address {
        self.access.addr
    }

    /// Get the address of this fetch at byte-granularity.
    #[must_use]
    pub fn byte_addr(&self) -> address {
        let requested_byte = self.access.byte_mask.first_one().unwrap_or(0) as u64;
        // self.addr() * mem_sub_partition::SECTOR_CHUNK_SIZE as u64 + requested_byte
        self.addr() + (requested_byte % mem_sub_partition::SECTOR_SIZE as u64)
    }

    /// Get the relative address of this fetch at byte-granularity.
    #[must_use]
    pub fn relative_byte_addr(&self) -> address {
        let requested_byte = self.access.byte_mask.first_one().unwrap_or(0) as u64;
        let relative_addr = self.relative_addr().unwrap_or(self.addr());
        // relative_addr * mem_sub_partition::SECTOR_CHUNK_SIZE as u64 + requested_byte
        relative_addr + (requested_byte % mem_sub_partition::SECTOR_SIZE as u64)
    }

    #[must_use]
    // #[inline]
    pub fn relative_addr(&self) -> Option<address> {
        self.access.relative_addr()
    }

    #[must_use]
    // #[inline]
    pub fn data_size(&self) -> u32 {
        self.access.req_size_bytes
    }

    #[must_use]
    // #[inline]
    pub fn control_size(&self) -> u32 {
        self.access.control_size()
    }

    #[must_use]
    // #[inline]
    pub fn size(&self) -> u32 {
        self.data_size() + self.control_size()
    }

    #[must_use]
    // #[inline]
    pub fn sub_partition_id(&self) -> usize {
        self.physical_addr.sub_partition as usize
    }

    #[must_use]
    // #[inline]
    pub fn access_kind(&self) -> access::Kind {
        self.access.kind
    }

    // #[inline]
    pub fn set_status(&mut self, status: Status, time: u64) {
        self.status = status;
        self.last_status_change = Some(time);
    }

    #[must_use]
    // #[inline]
    pub fn is_reply(&self) -> bool {
        matches!(self.kind, Kind::READ_REPLY | Kind::WRITE_ACK)
    }

    // #[inline]
    pub fn set_reply(&mut self) {
        assert!(!matches!(
            self.access.kind,
            access::Kind::L1_WRBK_ACC | access::Kind::L2_WRBK_ACC
        ));
        match self.kind {
            Kind::READ_REQUEST => {
                debug_assert!(!self.is_write());
                self.kind = Kind::READ_REPLY;
            }
            Kind::WRITE_REQUEST => {
                debug_assert!(self.is_write());
                self.kind = Kind::WRITE_ACK;
            }
            Kind::READ_REPLY | Kind::WRITE_ACK => {
                // panic!("cannot set reply for fetch of kind {:?}", self.kind);
            }
        }
    }
}
