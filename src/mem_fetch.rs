use super::{
    address, config,
    instruction::{MemorySpace, WarpInstruction},
    mem_sub_partition, warp, DecodedAddress,
};
use bitvec::{array::BitArray, BitArr};
use serde::{Deserialize, Serialize};

use std::sync::{Mutex, OnceLock};

pub const READ_PACKET_SIZE: u8 = 8;

// bytes: 6 address, 2 miscelaneous.
pub const WRITE_PACKET_SIZE: u8 = 8;

pub const WRITE_MASK_SIZE: u8 = 8;

pub type ByteMask = BitArr!(for mem_sub_partition::MAX_MEMORY_ACCESS_SIZE as usize);
pub type SectorMask = BitArr!(for mem_sub_partition::SECTOR_CHUNCK_SIZE as usize, in u8);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum Kind {
    READ_REQUEST = 0,
    WRITE_REQUEST,
    READ_REPLY,
    WRITE_ACK,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

#[derive(Debug, strum::EnumIter, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
    // NUM_MEM_ACCESS_TYPE,
}

impl From<AccessKind> for stats::mem::AccessKind {
    fn from(kind: AccessKind) -> Self {
        match kind {
            AccessKind::GLOBAL_ACC_R => Self::GLOBAL_ACC_R,
            AccessKind::LOCAL_ACC_R => Self::LOCAL_ACC_R,
            AccessKind::CONST_ACC_R => Self::CONST_ACC_R,
            AccessKind::TEXTURE_ACC_R => Self::TEXTURE_ACC_R,
            AccessKind::GLOBAL_ACC_W => Self::GLOBAL_ACC_W,
            AccessKind::LOCAL_ACC_W => Self::LOCAL_ACC_W,
            AccessKind::L1_WRBK_ACC => Self::L1_WRBK_ACC,
            AccessKind::L2_WRBK_ACC => Self::L2_WRBK_ACC,
            AccessKind::INST_ACC_R => Self::INST_ACC_R,
            AccessKind::L1_WR_ALLOC_R => Self::L1_WR_ALLOC_R,
            AccessKind::L2_WR_ALLOC_R => Self::L2_WR_ALLOC_R,
        }
    }
}

impl AccessKind {
    #[must_use]
    pub fn is_write(&self) -> bool {
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

#[derive(Clone, Hash, Eq, PartialEq)]
pub struct MemAccess {
    // uid: usize,
    /// request address
    pub addr: super::address,
    pub allocation: Option<crate::allocation::Allocation>,
    /// if access is write
    pub is_write: bool,
    /// request size in bytes
    pub req_size_bytes: u32,
    /// access type
    pub kind: AccessKind,
    // active_mask_t m_warp_mask;
    pub warp_mask: warp::ActiveMask,
    pub byte_mask: ByteMask,
    pub sector_mask: SectorMask,
}

pub trait BitString {
    fn to_bit_string(&self) -> String;
}

impl<A, O> BitString for BitArray<A, O>
where
    A: bitvec::view::BitViewSized,
    O: bitvec::order::BitOrder,
{
    fn to_bit_string(&self) -> String {
        self.iter()
            .map(|b| if *b { "1" } else { "0" })
            .collect::<Vec<_>>()
            .join("")
    }
}

impl std::fmt::Debug for MemAccess {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("MemAccess")
            .field("addr", &self.addr)
            .field("relative_addr", &self.relative_addr())
            .field("allocation", &self.allocation)
            .field("kind", &self.kind)
            .field("req_size_bytes", &self.req_size_bytes)
            .field("is_write", &self.is_write)
            .field("active_mask", &self.warp_mask.to_bit_string())
            .field("byte_mask", &self.byte_mask.to_bit_string())
            .field("sector_mask", &self.sector_mask.to_bit_string())
            .finish()
    }
}

impl std::fmt::Display for MemAccess {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Access({:?}@", self.kind)?;
        if let Some(ref alloc) = self.allocation {
            write!(f, "{}+{})", &alloc.id, self.addr - alloc.start_addr)
        } else {
            write!(f, "{})", &self.addr)
        }
    }
}

impl MemAccess {
    /// todo: where is this initialized
    #[must_use]
    pub fn new(
        kind: AccessKind,
        addr: address,
        allocation: Option<crate::allocation::Allocation>,
        req_size_bytes: u32,
        is_write: bool,
        warp_mask: warp::ActiveMask,
        byte_mask: ByteMask,
        sector_mask: SectorMask,
    ) -> Self {
        if let Some(ref alloc) = allocation {
            debug_assert!(alloc.start_addr <= addr);
        }
        Self {
            addr,
            allocation,
            is_write,
            req_size_bytes,
            kind,
            warp_mask,
            byte_mask,
            sector_mask,
        }
    }

    #[inline]
    #[must_use]
    pub fn relative_addr(&self) -> Option<super::address> {
        self.allocation
            .as_ref()
            .map(|alloc| alloc.start_addr)
            .and_then(|start| self.addr.checked_sub(start))
    }

    #[must_use]
    pub fn control_size(&self) -> u32 {
        if self.is_write {
            u32::from(WRITE_PACKET_SIZE)
        } else {
            u32::from(READ_PACKET_SIZE)
        }
    }

    #[must_use]
    pub fn data_size(&self) -> u32 {
        self.req_size_bytes
    }

    // #[must_use]
    // pub fn control_size(&self) -> u32 {
    //     self.access.control_size()
    // }

    #[must_use]
    pub fn size(&self) -> u32 {
        self.data_size() + self.control_size()
    }

    // /// use gen memory accesses
    // #[deprecated]
    // pub fn from_instr(instr: &WarpInstruction) -> Option<Self> {
    //     let Some(kind) = instr.access_kind() else {
    //         return None;
    //     };
    //     let Some(addr) = instr.addr() else {
    //         return None;
    //     };
    //     Some(Self {
    //         warp_mask: instr.active_mask,
    //         byte_mask: BitArray::ZERO,
    //         sector_mask: BitArray::ZERO,
    //         req_size_bytes: instr.data_size,
    //         is_write: instr.is_store(),
    //         kind,
    //         addr,
    //     })
    // }
}

#[derive(Clone, Debug)]
pub struct MemFetch {
    pub uid: u64,
    pub access: MemAccess,
    pub instr: Option<WarpInstruction>,
    pub tlx_addr: DecodedAddress,
    pub partition_addr: address,
    pub chip: u64,
    // pub control_size: u32,
    pub kind: Kind,
    // pub data_size: u32,
    pub warp_id: usize,
    pub core_id: usize,
    pub cluster_id: usize,
    pub pushed_cycle: Option<u64>,

    pub status: Status,
    pub last_status_change: Option<u64>,

    // this pointer is set up when a request is divided into
    // sector requests at L2 cache (if the req size > L2 sector
    // size), so the pointer refers to the original request
    pub original_fetch: Option<Box<MemFetch>>,

    // this fetch refers to the original write req,
    // when fetch-on-write policy is used
    pub original_write_fetch: Option<Box<MemFetch>>,
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
        //     && self.chip == other.chip
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
        // self.chip.hash(state);
        // self.control_size.hash(state);
        // self.data_size.hash(state);
        // self.warp_id.hash(state);
        // self.core_id.hash(state);
        // self.cluster_id.hash(state);
    }
}

impl MemFetch {
    pub fn new(
        instr: Option<WarpInstruction>,
        access: MemAccess,
        config: &config::GPUConfig,
        control_size: u32,
        warp_id: usize,
        core_id: usize,
        cluster_id: usize,
    ) -> Self {
        // let data_size = access.req_size_bytes;
        let kind = if access.is_write {
            Kind::WRITE_REQUEST
        } else {
            Kind::READ_REQUEST
        };

        let tlx_addr = config.address_mapping().tlx(access.addr);
        let partition_addr = config.address_mapping().partition_address(access.addr);

        static MEM_FETCH_UID: OnceLock<Mutex<u64>> = OnceLock::new();
        let mut uid_lock = MEM_FETCH_UID.get_or_init(|| Mutex::new(0)).lock().unwrap();
        let uid = *uid_lock;
        *uid_lock += 1;
        Self {
            uid,
            access,
            instr,
            warp_id,
            core_id,
            cluster_id,
            // data_size,
            // control_size,
            tlx_addr,
            partition_addr,
            chip: 0,
            kind,
            status: Status::INITIALIZED,
            pushed_cycle: None,
            last_status_change: None,
            original_fetch: None,
            original_write_fetch: None,
        }
    }

    pub fn is_atomic(&self) -> bool {
        self.instr
            .as_ref()
            .map_or(false, WarpInstruction::is_atomic)
    }

    #[must_use]
    pub fn is_texture(&self) -> bool {
        self.instr
            .as_ref()
            .map_or(false, |i| i.memory_space == Some(MemorySpace::Texture))
    }

    #[must_use]
    pub fn is_write(&self) -> bool {
        self.access.is_write
    }

    #[must_use]
    pub fn addr(&self) -> address {
        self.access.addr
    }

    #[must_use]
    pub fn relative_addr(&self) -> Option<address> {
        self.access.relative_addr()
    }

    #[must_use]
    pub fn data_size(&self) -> u32 {
        self.access.req_size_bytes
    }

    #[must_use]
    pub fn control_size(&self) -> u32 {
        self.access.control_size()
    }

    #[must_use]
    pub fn size(&self) -> u32 {
        self.data_size() + self.control_size()
    }

    #[must_use]
    pub fn access_byte_mask(&self) -> &ByteMask {
        &self.access.byte_mask
    }

    #[must_use]
    pub fn access_warp_mask(&self) -> &warp::ActiveMask {
        &self.access.warp_mask
    }

    #[must_use]
    pub fn access_sector_mask(&self) -> &SectorMask {
        &self.access.sector_mask
    }

    #[must_use]
    pub fn sub_partition_id(&self) -> usize {
        self.tlx_addr.sub_partition as usize
    }

    #[must_use]
    pub fn access_kind(&self) -> &AccessKind {
        &self.access.kind
    }

    pub fn set_status(&mut self, status: Status, time: u64) {
        self.status = status;
        self.last_status_change = Some(time);
    }

    #[must_use]
    pub fn is_reply(&self) -> bool {
        matches!(self.kind, Kind::READ_REPLY | Kind::WRITE_ACK)
    }

    pub fn set_reply(&mut self) {
        assert!(!matches!(
            self.access.kind,
            AccessKind::L1_WRBK_ACC | AccessKind::L2_WRBK_ACC
        ));
        match self.kind {
            Kind::READ_REQUEST => {
                debug_assert!(!self.is_write());
                self.kind = Kind::READ_REPLY
            }
            Kind::WRITE_REQUEST => {
                debug_assert!(self.is_write());
                self.kind = Kind::WRITE_ACK
            }
            Kind::READ_REPLY | Kind::WRITE_ACK => {
                // panic!("cannot set reply for fetch of kind {:?}", self.kind);
            }
        }
    }
}
