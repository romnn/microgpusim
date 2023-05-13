use super::addrdec::LinearToRawAddressTranslation;
use super::instruction::WarpInstruction;
use super::scheduler::ThreadActiveMask;
use crate::config::GPUConfig;
use crate::ported::{address, DecodedAddress, READ_PACKET_SIZE, WRITE_PACKET_SIZE};
use bitvec::{array::BitArray, field::BitField, BitArr};

pub type MemAccessByteMask = BitArr!(for super::MAX_MEMORY_ACCESS_SIZE);
pub type MemAccessSectorMask = BitArr!(for super::SECTOR_CHUNCK_SIZE, in u8);

pub trait Interconnect {
    fn full(&self, size: usize, write: bool) -> bool;
    fn push(&self, mf: MemFetch);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    READ_REQUEST = 0,
    WRITE_REQUEST,
    READ_REPLY, // send to shader
    WRITE_ACK,
    // Atomic,
    // Const,
    // Tex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(incorrect_ident_case)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemorySpace {
    Undefined,
    Reg,
    Local,
    Shared,
    Sstarr,
    ParamUnclassified,
    ParamKernel, // global to all threads in a kernel : read-only
    ParamLocal,  // local to a thread : read-writable
    Const,
    Tex,
    Surf,
    Global,
    Generic,
    Instruction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    NUM_MEM_ACCESS_TYPE,
}

#[derive(Clone, Debug)]
pub struct MemAccess {
    // uid: usize,
    /// request address
    pub addr: super::address,
    /// if access is write
    pub is_write: bool,
    /// request size in bytes
    pub req_size_bytes: u32,
    /// access type
    pub kind: AccessKind,
    // active_mask_t m_warp_mask;
    pub warp_mask: ThreadActiveMask,
    pub byte_mask: MemAccessByteMask,
    pub sector_mask: MemAccessSectorMask,
}

impl MemAccess {
    /// todo: where is this initialized
    pub fn new(
        kind: AccessKind,
        addr: address,
        req_size_bytes: u32,
        is_write: bool,
        warp_mask: ThreadActiveMask,
        byte_mask: MemAccessByteMask,
        sector_mask: MemAccessSectorMask,
    ) -> Self {
        Self {
            // uid:
            warp_mask,
            byte_mask,
            sector_mask,
            req_size_bytes,
            is_write,
            kind,
            addr,
        }
    }

    /// use gen memory accesses
    #[deprecated]
    pub fn from_instr(instr: &WarpInstruction) -> Option<Self> {
        let Some(kind) = instr.access_kind() else {
            return None;
        };
        let Some(addr) = instr.addr() else {
            return None;
        };
        Some(Self {
            warp_mask: instr.active_mask,
            byte_mask: BitArray::ZERO,
            sector_mask: BitArray::ZERO,
            req_size_bytes: instr.data_size,
            is_write: instr.is_store(),
            kind,
            addr,
        })
    }
}

#[derive(Clone, Debug)]
pub struct MemFetch {
    pub access: MemAccess,
    pub instr: WarpInstruction,
    pub tlx_addr: DecodedAddress,
    pub partition_addr: address,
    pub chip: usize,
    // pub sub_partition_id: usize,
    pub control_size: u32,
    pub kind: Kind,
    pub data_size: u32,
    pub warp_id: usize,
    pub core_id: usize,
    pub cluster_id: usize,

    pub status: Status,
    pub last_status_change: Option<usize>,
}

impl MemFetch {
    pub fn is_atomic(&self) -> bool {
        self.instr.is_atomic()
    }

    pub fn is_write(&self) -> bool {
        self.access.is_write
    }

    pub fn addr(&self) -> address {
        self.access.addr
    }

    // pub fn cache_op(&self) -> super::instruction::CacheOperator {
    //     self.instr.cache_op
    // }

    pub fn access_byte_mask(&self) -> &MemAccessByteMask {
        &self.access.byte_mask
    }

    pub fn access_warp_mask(&self) -> &ThreadActiveMask {
        &self.access.warp_mask
    }

    pub fn access_sector_mask(&self) -> &MemAccessSectorMask {
        &self.access.sector_mask
    }

    pub fn sub_partition_id(&self) -> u64 {
        self.tlx_addr.sub_partition
    }

    pub fn access_kind(&self) -> &AccessKind {
        &self.access.kind
    }

    pub fn set_status(&mut self, status: Status, time: usize) {
        self.status = status;
        self.last_status_change = Some(time);
    }

    pub fn set_addr(&mut self, addr: address) {
        self.access.addr = addr;
    }

    pub fn new(
        instr: WarpInstruction,
        access: MemAccess,
        config: &GPUConfig,
        control_size: u32,
        warp_id: usize,
        core_id: usize,
        cluster_id: usize,
    ) -> Self {
        // m_request_uid = sm_next_mf_request_uid++;
        // let warp_id = instr.warp_id;
        let data_size = access.req_size_bytes;
        let kind = if access.is_write {
            Kind::WRITE_REQUEST
        } else {
            Kind::READ_REQUEST
        };

        let tlx_addr = config.address_mapping().tlx(access.addr);
        let partition_addr = config.address_mapping().partition_address(access.addr);
        Self {
            access,
            instr,
            warp_id,
            core_id,
            cluster_id,
            data_size,
            control_size,
            tlx_addr,
            partition_addr,
            chip: 0,
            // sub_partition_id: 0,
            kind,
            status: Status::INITIALIZED,
            last_status_change: None,
        }
        // if (inst) {
        // m_inst = *inst;
        // assert(wid == m_inst.warp_id());
        // }
        // m_data_size = access.get_size();
        // m_ctrl_size = ctrl_size;
        // m_sid = sid;
        // m_tpc = tpc;
        // m_wid = wid;
        // config->m_address_mapping.addrdec_tlx(access.get_addr(), &m_raw_addr);
        // m_partition_addr =
        //   config->m_address_mapping.partition_address(access.get_addr());
        // m_type = m_access.is_write() ? WRITE_REQUEST : READ_REQUEST;
        // m_timestamp = cycle;
        // m_timestamp2 = 0;
        // m_status = MEM_FETCH_INITIALIZED;
        // m_status_change = cycle;
        // m_mem_config = config;
        // icnt_flit_size = config->icnt_flit_size;
        // original_mf = m_original_mf;
        // original_wr_mf = m_original_wr_mf;
        // if (m_original_mf) {
        // m_raw_addr.chip = m_original_mf->get_tlx_addr().chip;
        // m_raw_addr.sub_partition = m_original_mf->get_tlx_addr().sub_partition;
        // }
    }
}

// class shader_core_mem_fetch_allocator : public mem_fetch_allocator {
//  public:
//   shader_core_mem_fetch_allocator(unsigned core_id, unsigned cluster_id,
//                                   const memory_config *config) {
//     m_core_id = core_id;
//     m_cluster_id = cluster_id;
//     m_memory_config = config;
//   }
//   mem_fetch *alloc(new_addr_type addr, mem_access_type type, unsigned size,
//                    bool wr, unsigned long long cycle) const;
//   mem_fetch *alloc(new_addr_type addr, mem_access_type type,
//                    const active_mask_t &active_mask,
//                    const mem_access_byte_mask_t &byte_mask,
//                    const mem_access_sector_mask_t &sector_mask, unsigned size,
//                    bool wr, unsigned long long cycle, unsigned wid,
//                    unsigned sid, unsigned tpc, mem_fetch *original_mf) const;
//   mem_fetch *alloc(const warp_inst_t &inst, const mem_access_t &access,
//                    unsigned long long cycle) const {
//     warp_inst_t inst_copy = inst;
//     mem_fetch *mf = new mem_fetch(
//         access, &inst_copy,
//         access.is_write() ? WRITE_PACKET_SIZE : READ_PACKET_SIZE,
//         inst.warp_id(), m_core_id, m_cluster_id, m_memory_config, cycle);
//     return mf;
//   }
//
//  private:
//   unsigned m_core_id;
//   unsigned m_cluster_id;
//   const memory_config *m_memory_config;
// };

// class mem_fetch {
//  public:
//   mem_fetch(const mem_access_t &access, const warp_inst_t *inst,
//             unsigned ctrl_size, unsigned wid, unsigned sid, unsigned tpc,
//             const memory_config *config, unsigned long long cycle,
//             mem_fetch *original_mf = NULL, mem_fetch *original_wr_mf = NULL);
//   ~mem_fetch();
//
//   void set_status(enum mem_fetch_status status, unsigned long long cycle);
//   void set_reply() {
//     assert(m_access.get_type() != L1_WRBK_ACC &&
//            m_access.get_type() != L2_WRBK_ACC);
//     if (m_type == READ_REQUEST) {
//       assert(!get_is_write());
//       m_type = READ_REPLY;
//     } else if (m_type == WRITE_REQUEST) {
//       assert(get_is_write());
//       m_type = WRITE_ACK;
//     }
//   }
//   void do_atomic();
//
//   void print(FILE *fp, bool print_inst = true) const;
//
//   const addrdec_t &get_tlx_addr() const { return m_raw_addr; }
//   void set_chip(unsigned chip_id) { m_raw_addr.chip = chip_id; }
//   void set_parition(unsigned sub_partition_id) {
//     m_raw_addr.sub_partition = sub_partition_id;
//   }
//   unsigned get_data_size() const { return m_data_size; }
//   void set_data_size(unsigned size) { m_data_size = size; }
//   unsigned get_ctrl_size() const { return m_ctrl_size; }
//   unsigned size() const { return m_data_size + m_ctrl_size; }
//   bool is_write() { return m_access.is_write(); }
//   void set_addr(new_addr_type addr) { m_access.set_addr(addr); }
//   new_addr_type get_addr() const { return m_access.get_addr(); }
//   unsigned get_access_size() const { return m_access.get_size(); }
//   new_addr_type get_partition_addr() const { return m_partition_addr; }
//   unsigned get_sub_partition_id() const { return m_raw_addr.sub_partition; }
//   bool get_is_write() const { return m_access.is_write(); }
//   unsigned get_request_uid() const { return m_request_uid; }
//   unsigned get_sid() const { return m_sid; }
//   unsigned get_tpc() const { return m_tpc; }
//   unsigned get_wid() const { return m_wid; }
//   bool istexture() const;
//   bool isconst() const;
//   enum mf_type get_type() const { return m_type; }
//   bool isatomic() const;
//
//   void set_return_timestamp(unsigned t) { m_timestamp2 = t; }
//   void set_icnt_receive_time(unsigned t) { m_icnt_receive_time = t; }
//   unsigned get_timestamp() const { return m_timestamp; }
//   unsigned get_return_timestamp() const { return m_timestamp2; }
//   unsigned get_icnt_receive_time() const { return m_icnt_receive_time; }
//
//   enum mem_access_type get_access_type() const { return m_access.get_type(); }
//   const active_mask_t &get_access_warp_mask() const {
//     return m_access.get_warp_mask();
//   }
//   mem_access_byte_mask_t get_access_byte_mask() const {
//     return m_access.get_byte_mask();
//   }
//   mem_access_sector_mask_t get_access_sector_mask() const {
//     return m_access.get_sector_mask();
//   }
//
//   address_type get_pc() const { return m_inst.empty() ? -1 : m_inst.pc; }
//   const warp_inst_t &get_inst() { return m_inst; }
//   enum mem_fetch_status get_status() const { return m_status; }
//
//   const memory_config *get_mem_config() { return m_mem_config; }
//
//   unsigned get_num_flits(bool simt_to_mem);
//
//   mem_fetch *get_original_mf() { return original_mf; }
//   mem_fetch *get_original_wr_mf() { return original_wr_mf; }
//
//  private:
//   // request source information
//   unsigned m_request_uid;
//   unsigned m_sid;
//   unsigned m_tpc;
//   unsigned m_wid;
//
//   // where is this request now?
//   enum mem_fetch_status m_status;
//   unsigned long long m_status_change;
//
//   // request type, address, size, mask
//   mem_access_t m_access;
//   unsigned m_data_size;  // how much data is being written
//   unsigned
//       m_ctrl_size;  // how big would all this meta data be in hardware (does not
//                     // necessarily match actual size of mem_fetch)
//   new_addr_type
//       m_partition_addr;  // linear physical address *within* dram partition
//                          // (partition bank select bits squeezed out)
//   addrdec_t m_raw_addr;  // raw physical address (i.e., decoded DRAM
//                          // chip-row-bank-column address)
//   enum mf_type m_type;
//
//   // statistics
//   unsigned
//       m_timestamp;  // set to gpu_sim_cycle+gpu_tot_sim_cycle at struct creation
//   unsigned m_timestamp2;  // set to gpu_sim_cycle+gpu_tot_sim_cycle when pushed
//                           // onto icnt to shader; only used for reads
//   unsigned m_icnt_receive_time;  // set to gpu_sim_cycle + interconnect_latency
//                                  // when fixed icnt latency mode is enabled
//
//   // requesting instruction (put last so mem_fetch prints nicer in gdb)
//   warp_inst_t m_inst;
//
//   static unsigned sm_next_mf_request_uid;
//
//   const memory_config *m_mem_config;
//   unsigned icnt_flit_size;
//
//   mem_fetch
//       *original_mf;  // this pointer is set up when a request is divided into
//                      // sector requests at L2 cache (if the req size > L2 sector
//                      // size), so the pointer refers to the original request
//   mem_fetch *original_wr_mf;  // this pointer refers to the original write req,
//                               // when fetch-on-write policy is used
// };
