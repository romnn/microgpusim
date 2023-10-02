use playground::types;
use serde::Serialize;
use utils::box_slice;

impl From<types::mem_fetch::mf_type> for crate::mem_fetch::Kind {
    // #[inline]
    fn from(kind: types::mem_fetch::mf_type) -> Self {
        use types::mem_fetch::mf_type;
        match kind {
            mf_type::READ_REQUEST => crate::mem_fetch::Kind::READ_REQUEST,
            mf_type::WRITE_REQUEST => crate::mem_fetch::Kind::WRITE_REQUEST,
            mf_type::READ_REPLY => crate::mem_fetch::Kind::READ_REPLY,
            mf_type::WRITE_ACK => crate::mem_fetch::Kind::WRITE_ACK,
        }
    }
}

impl From<types::mem_fetch::mem_access_type> for crate::mem_fetch::access::Kind {
    // #[inline]
    fn from(kind: types::mem_fetch::mem_access_type) -> Self {
        use crate::mem_fetch::access::Kind as AccessKind;
        use types::mem_fetch::mem_access_type;
        match kind {
            mem_access_type::GLOBAL_ACC_R => AccessKind::GLOBAL_ACC_R,
            mem_access_type::LOCAL_ACC_R => AccessKind::LOCAL_ACC_R,
            mem_access_type::CONST_ACC_R => AccessKind::CONST_ACC_R,
            mem_access_type::TEXTURE_ACC_R => AccessKind::TEXTURE_ACC_R,
            mem_access_type::GLOBAL_ACC_W => AccessKind::GLOBAL_ACC_W,
            mem_access_type::LOCAL_ACC_W => AccessKind::LOCAL_ACC_W,
            mem_access_type::L1_WRBK_ACC => AccessKind::L1_WRBK_ACC,
            mem_access_type::L2_WRBK_ACC => AccessKind::L2_WRBK_ACC,
            mem_access_type::INST_ACC_R => AccessKind::INST_ACC_R,
            mem_access_type::L1_WR_ALLOC_R => AccessKind::L1_WR_ALLOC_R,
            mem_access_type::L2_WR_ALLOC_R => AccessKind::L2_WR_ALLOC_R,
            other @ mem_access_type::NUM_MEM_ACCESS_TYPE => {
                panic!("bad mem access kind: {other:?}")
            }
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Serialize)]
pub struct Cache {
    pub lines: Vec<CacheBlock>,
}

impl std::fmt::Debug for Cache {
    // #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_list()
            .entries(
                self.lines.iter().enumerate(), // we only show valid tags
                                               // .filter(|(idx, line)| line.tag != 0),
            )
            .finish()
    }
}

impl<T> From<&crate::tag_array::TagArray<crate::cache::block::Line, T>> for Cache {
    // #[inline]
    fn from(tag_array: &crate::tag_array::TagArray<crate::cache::block::Line, T>) -> Self {
        Self {
            lines: tag_array.lines.iter().cloned().map(Into::into).collect(),
        }
    }
}

impl<T, const N: usize> From<&crate::tag_array::TagArray<crate::cache::block::sector::Block<N>, T>>
    for Cache
{
    // #[inline]
    fn from(
        tag_array: &crate::tag_array::TagArray<crate::cache::block::sector::Block<N>, T>,
    ) -> Self {
        Self {
            lines: tag_array.lines.iter().cloned().map(Into::into).collect(),
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize)]
pub enum CacheBlockStatus {
    INVALID,
    RESERVED,
    VALID,
    MODIFIED,
}

impl From<crate::cache::block::Status> for CacheBlockStatus {
    // #[inline]
    fn from(status: crate::cache::block::Status) -> Self {
        match status {
            crate::cache::block::Status::INVALID => Self::INVALID,
            crate::cache::block::Status::RESERVED => Self::RESERVED,
            crate::cache::block::Status::VALID => Self::VALID,
            crate::cache::block::Status::MODIFIED => Self::MODIFIED,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Serialize)]
pub struct CacheBlock {
    /// Cache block tag
    pub tag: u64,
    /// Cache block address
    pub block_addr: u64,
    /// Sector status
    pub sector_status: Vec<CacheBlockStatus>,
    /// Last access time per sector
    pub sector_last_accessed: Vec<u64>,
    /// Last access time of the cache block
    pub last_accessed: u64,
}

impl From<crate::cache::block::Line> for CacheBlock {
    // #[inline]
    fn from(block: crate::cache::block::Line) -> Self {
        Self {
            tag: block.tag,
            block_addr: block.block_addr,
            sector_status: vec![block.status.into()],
            sector_last_accessed: vec![block.last_access_time],
            last_accessed: block.last_access_time,
        }
    }
}

impl<const N: usize> From<crate::cache::block::sector::Block<N>> for CacheBlock {
    // #[inline]
    fn from(block: crate::cache::block::sector::Block<N>) -> Self {
        Self {
            tag: block.tag,
            block_addr: block.block_addr,
            sector_status: block.status.into_iter().map(Into::into).collect(),
            sector_last_accessed: block.last_sector_access_time.to_vec(),
            last_accessed: block.last_access_time,
        }
    }
}

impl From<playground::cache::cache_block_state> for CacheBlockStatus {
    // #[inline]
    fn from(state: playground::cache::cache_block_state) -> Self {
        use playground::cache::cache_block_state;
        match state {
            cache_block_state::MODIFIED => CacheBlockStatus::MODIFIED,
            cache_block_state::INVALID => CacheBlockStatus::INVALID,
            cache_block_state::RESERVED => CacheBlockStatus::RESERVED,
            cache_block_state::VALID => CacheBlockStatus::VALID,
        }
    }
}

impl<'a> From<playground::cache::CacheBlock<'a>> for CacheBlock {
    // #[inline]
    fn from(block: playground::cache::CacheBlock<'a>) -> Self {
        Self {
            tag: block.get_tag(),
            block_addr: block.get_block_addr(),
            sector_status: block.sector_status().into_iter().map(Into::into).collect(),
            last_accessed: block.get_last_access_time(),
            sector_last_accessed: block.last_sector_access_time(),
        }
    }
}

impl CacheBlock {
    // #[inline]
    pub fn sectored(&self) -> bool {
        assert!(self.sector_status.len() > 0);
        self.sector_status.len() > 1
    }
}

impl std::fmt::Debug for CacheBlock {
    // #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.sectored() {
            write!(
                f,
                "(sectors={:?}, tag={}, block={}, accessed={})",
                self.sector_status, self.tag, self.block_addr, self.last_accessed
            )
        } else {
            write!(
                f,
                "{:?}(tag={}, block={}, accessed={})",
                self.sector_status[0], self.tag, self.block_addr, self.last_accessed
            )
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Serialize)]
pub struct WarpInstruction {
    pub opcode: String,
    pub pc: usize,
    pub latency: usize,
    pub initiation_interval: usize,
    pub dispatch_delay_cycles: usize,
    pub warp_id: usize,
}

impl std::fmt::Debug for WarpInstruction {
    // #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}[pc={},warp={}]", self.opcode, self.pc, self.warp_id)
    }
}

impl From<crate::instruction::WarpInstruction> for WarpInstruction {
    // #[inline]
    fn from(instr: crate::instruction::WarpInstruction) -> Self {
        WarpInstruction {
            opcode: instr.opcode.to_string(),
            pc: instr.pc,
            latency: instr.latency,
            initiation_interval: instr.initiation_interval,
            dispatch_delay_cycles: instr.dispatch_delay_cycles,
            warp_id: instr.warp_id,
        }
    }
}

#[derive(Clone, Default, PartialEq, Eq, Hash, Serialize)]
pub struct RegisterSet {
    pub name: String,
    pub pipeline: Vec<Option<WarpInstruction>>,
}

impl RegisterSet {
    // #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.num_instructions_in_pipeline() == 0
    }

    // #[inline]
    #[must_use]
    pub fn num_instructions_in_pipeline(&self) -> usize {
        self.pipeline
            .iter()
            .filter_map(std::option::Option::as_ref)
            .count()
    }
}

impl std::fmt::Debug for RegisterSet {
    // #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}={:?}", self.name, self.pipeline)
    }
}

impl From<crate::register_set::RegisterSet> for RegisterSet {
    // #[inline]
    fn from(reg: crate::register_set::RegisterSet) -> Self {
        let pipeline = reg
            .regs
            .into_iter()
            .map(|instr| instr.map(std::convert::Into::into))
            .collect();
        Self {
            name: format!("{:?}", &reg.stage),
            pipeline,
        }
    }
}

impl<'a> From<playground::warp_inst::WarpInstr<'a>> for WarpInstruction {
    // #[inline]
    fn from(instr: playground::warp_inst::WarpInstr<'a>) -> Self {
        let opcode = instr.opcode_str().trim_start_matches("OP_").to_string();
        Self {
            opcode,
            pc: instr.get_pc() as usize,
            latency: instr.get_latency() as usize,
            initiation_interval: instr.get_initiation_interval() as usize,
            dispatch_delay_cycles: instr.get_dispatch_delay_cycles() as usize,
            warp_id: instr.warp_id() as usize,
        }
    }
}

impl<'a> From<playground::register_set::RegisterSet<'a>> for RegisterSet {
    // #[inline]
    fn from(reg: playground::register_set::RegisterSet<'a>) -> Self {
        Self {
            name: reg.name(),
            pipeline: reg
                .registers()
                .into_iter()
                .map(|instr| {
                    if instr.empty() {
                        None
                    } else {
                        Some(instr.into())
                    }
                })
                .collect(),
        }
    }
}

#[derive(strum::FromRepr, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[repr(u32)]
pub enum OperandCollectorUnitKind {
    SP_CUS,
    DP_CUS,
    SFU_CUS,
    TENSOR_CORE_CUS,
    INT_CUS,
    MEM_CUS,
    GEN_CUS,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct CollectorUnit {
    pub warp_id: Option<usize>,
    pub warp_instr: Option<WarpInstruction>,
    pub output_register: Option<RegisterSet>,
    pub not_ready: String,
    pub reg_id: Option<usize>,
    pub kind: OperandCollectorUnitKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct DispatchUnit {
    pub last_cu: usize,
    pub next_cu: usize,
    pub kind: OperandCollectorUnitKind,
}

impl From<&playground::operand_collector::dispatch_unit_t> for DispatchUnit {
    // #[inline]
    fn from(unit: &playground::operand_collector::dispatch_unit_t) -> Self {
        Self {
            last_cu: unit.get_last_cu() as usize,
            next_cu: unit.get_next_cu() as usize,
            kind: OperandCollectorUnitKind::from_repr(unit.get_set_id()).unwrap(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize)]
pub struct Port {
    pub in_ports: Vec<RegisterSet>,
    pub out_ports: Vec<RegisterSet>,
    pub ids: Vec<OperandCollectorUnitKind>,
}

impl Port {
    // #[inline]
    pub fn is_empty(&self) -> bool {
        self.in_ports.iter().all(RegisterSet::is_empty)
            && self.in_ports.iter().all(RegisterSet::is_empty)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize)]
pub struct Arbiter {
    pub last_cu: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize)]
pub struct OperandCollector {
    pub ports: Vec<Port>,
    pub collector_units: Vec<CollectorUnit>,
    pub dispatch_units: Vec<DispatchUnit>,
    pub arbiter: Arbiter,
}

impl<'a> From<playground::port::Port<'a>> for Port {
    // #[inline]
    fn from(port: playground::port::Port<'a>) -> Self {
        let in_ports = port.in_ports().into_iter();
        let out_ports = port.out_ports().into_iter();
        let ids = port
            .cu_sets()
            .map(|&set| OperandCollectorUnitKind::from_repr(set).unwrap())
            .collect();
        Self {
            in_ports: in_ports.map(Into::into).collect(),
            out_ports: out_ports.map(Into::into).collect(),
            ids,
        }
    }
}

impl<'a> From<playground::collector_unit::CollectorUnit<'a>> for CollectorUnit {
    // #[inline]
    fn from(cu: playground::collector_unit::CollectorUnit<'a>) -> Self {
        Self {
            kind: OperandCollectorUnitKind::from_repr(cu.set_id()).unwrap(),
            warp_id: cu.warp_id(),
            warp_instr: cu.warp_instruction().map(Into::into),
            output_register: cu.output_register().map(Into::into),
            not_ready: cu.not_ready_mask(),
            reg_id: cu.reg_id(),
        }
    }
}

impl From<&playground::operand_collector::arbiter_t> for Arbiter {
    // #[inline]
    fn from(arbiter: &playground::operand_collector::arbiter_t) -> Self {
        Self {
            last_cu: arbiter.get_last_cu() as usize,
        }
    }
}

impl<'a> From<playground::operand_collector::OperandCollector<'a>> for OperandCollector {
    // #[inline]
    fn from(opcoll: playground::operand_collector::OperandCollector<'a>) -> Self {
        use std::collections::HashSet;
        let skip: HashSet<_> = [
            OperandCollectorUnitKind::TENSOR_CORE_CUS,
            // OperandCollectorUnitKind::SFU_CUS,
        ]
        .into_iter()
        .collect();
        let ports = opcoll
            .ports()
            .into_iter()
            .map(Port::from)
            .filter(|port| {
                let ids = port.ids.clone().into_iter();
                let ids: HashSet<_> = ids.collect();
                if ids.intersection(&skip).count() > 0 {
                    // skip and make sure that they are empty anyways
                    // dbg!(&port.in_ports);
                    // dbg!(&port.out_ports);
                    assert!(port.is_empty());
                    false
                } else {
                    true
                }
            })
            .collect();
        let collector_units = opcoll
            .collector_units()
            .into_iter()
            .map(CollectorUnit::from)
            .filter(|unit| !skip.contains(&unit.kind))
            .collect();
        let dispatch_units = opcoll
            .dispatch_units()
            .map(DispatchUnit::from)
            .filter(|unit| !skip.contains(&unit.kind))
            .collect();
        let arbiter = opcoll.arbiter().into();
        Self {
            ports,
            collector_units,
            dispatch_units,
            arbiter,
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct Scheduler {
    pub prioritized_warp_ids: Vec<(usize, usize)>,
}

impl<'a> From<playground::scheduler_unit::SchedulerUnit<'a>> for Scheduler {
    // #[inline]
    fn from(scheduler: playground::scheduler_unit::SchedulerUnit<'a>) -> Self {
        let prioritized_warp_ids = scheduler.prioritized_warp_ids();
        let prioritized_dynamic_warp_ids = scheduler.prioritized_dynamic_warp_ids();
        assert_eq!(
            prioritized_warp_ids.len(),
            prioritized_dynamic_warp_ids.len()
        );
        Self {
            prioritized_warp_ids: prioritized_warp_ids
                .into_iter()
                .zip(prioritized_dynamic_warp_ids)
                .collect(),
        }
    }
}

#[derive(Clone, Serialize)]
pub struct MemFetch {
    pub kind: crate::mem_fetch::Kind,
    pub access_kind: crate::mem_fetch::access::Kind,
    // cannot compare addr because its different between runs
    // addr: crate::address,
    pub relative_addr: Option<(usize, crate::address)>,
}

impl Eq for MemFetch {}

impl PartialEq for MemFetch {
    // #[inline]
    fn eq(&self, other: &Self) -> bool {
        let mut equal = self.kind == other.kind;
        equal &= self.access_kind == other.access_kind;

        if let (Some(alloc), Some(other_alloc)) = (self.relative_addr, other.relative_addr) {
            // playground does not track allocations which have not been copied to.
            // Hence, we only compare alloations where possible.
            equal &= alloc == other_alloc;
        }
        equal
    }
}

impl std::hash::Hash for MemFetch {
    // #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.kind.hash(state);
        self.access_kind.hash(state);
    }
}

impl std::fmt::Debug for MemFetch {
    // #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}({:?}", self.kind, self.access_kind)?;
        if let Some((alloc_id, rel_addr)) = self.relative_addr {
            write!(f, "@{alloc_id}+{rel_addr}")?;
        }
        write!(f, ")")
    }
}

impl<'a> From<playground::mem_fetch::MemFetch<'a>> for MemFetch {
    // #[inline]
    fn from(fetch: playground::mem_fetch::MemFetch<'a>) -> Self {
        let addr = fetch.get_addr();
        let relative_addr = fetch.get_relative_addr();
        Self {
            kind: fetch.get_type().into(),
            access_kind: fetch.get_access_type().into(),
            relative_addr: if addr == relative_addr {
                None
            } else {
                Some((fetch.get_alloc_id() as usize, relative_addr))
            },
        }
    }
}

impl From<crate::mem_fetch::MemFetch> for MemFetch {
    // #[inline]
    fn from(fetch: crate::mem_fetch::MemFetch) -> Self {
        let addr = fetch.addr();
        Self {
            kind: fetch.kind,
            access_kind: fetch.access_kind(),
            relative_addr: match fetch.access.allocation {
                Some(alloc) => Some((alloc.id, addr - alloc.start_addr)),
                None => None,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct PendingRegisterWrites {
    pub warp_id: usize,
    pub reg_num: u32,
    pub pending: usize,
}

impl From<&playground::core::pending_register_writes> for PendingRegisterWrites {
    // #[inline]
    fn from(writes: &playground::core::pending_register_writes) -> Self {
        Self {
            warp_id: writes.warp_id as usize,
            reg_num: writes.reg_num,
            pending: writes.pending as usize,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Arbitration {
    pub last_borrower: usize,
    pub shared_credit: usize,
    pub private_credit: Box<[usize]>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Simulation {
    pub last_cluster_issue: usize,
    // per sub partition
    pub interconn_to_l2_queue_per_sub: Box<[Vec<MemFetch>]>,
    pub l2_to_interconn_queue_per_sub: Box<[Vec<MemFetch>]>,
    pub l2_to_dram_queue_per_sub: Box<[Vec<MemFetch>]>,
    pub dram_to_l2_queue_per_sub: Box<[Vec<MemFetch>]>,
    pub rop_queue_per_sub: Box<[Vec<(u64, MemFetch)>]>,
    pub l2_cache_per_sub: Box<[Option<Cache>]>,
    // per partition
    pub dram_latency_queue_per_partition: Box<[Vec<MemFetch>]>,
    pub dram_arbitration_per_partition: Box<[Arbitration]>,
    // per cluster
    pub core_sim_order_per_cluster: Box<[Box<[usize]>]>,
    // per core
    pub functional_unit_occupied_slots_per_core: Box<[String]>,
    pub functional_unit_pipelines_per_core: Box<[Vec<RegisterSet>]>,
    pub operand_collector_per_core: Box<[Option<OperandCollector>]>,
    pub scheduler_per_core: Box<[Box<[Scheduler]>]>,
    pub pending_register_writes_per_core: Box<[Vec<PendingRegisterWrites>]>,
    pub l1_latency_queue_per_core: Box<[Vec<(usize, Vec<Option<MemFetch>>)>]>,
    pub l1_cache_per_core: Box<[Option<Cache>]>,
}

impl Simulation {
    #[must_use]
    pub fn new(
        num_clusters: usize,
        cores_per_cluster: usize,
        num_mem_partitions: usize,
        num_sub_partitions: usize,
        num_schedulers: usize,
    ) -> Self {
        let total_cores = num_clusters * cores_per_cluster;
        Self {
            last_cluster_issue: 0,

            // per sub partition
            interconn_to_l2_queue_per_sub: box_slice![vec![]; num_sub_partitions],
            l2_to_interconn_queue_per_sub: box_slice![vec![]; num_sub_partitions],
            l2_to_dram_queue_per_sub: box_slice![vec![]; num_sub_partitions],
            dram_to_l2_queue_per_sub: box_slice![vec![]; num_sub_partitions],
            rop_queue_per_sub: box_slice![vec![]; num_sub_partitions],
            l2_cache_per_sub: box_slice![None; num_sub_partitions],

            // per partition
            dram_latency_queue_per_partition: box_slice![vec![]; num_mem_partitions],
            dram_arbitration_per_partition: box_slice![
                Arbitration {
                    last_borrower: 0,
                    shared_credit: 0,
                    private_credit: box_slice![0; num_sub_partitions],
                };
                num_mem_partitions
            ],

            // per cluster
            core_sim_order_per_cluster: box_slice![
                box_slice![0; cores_per_cluster];
                num_clusters
            ],

            // per core
            functional_unit_occupied_slots_per_core: box_slice![String::new(); total_cores],
            functional_unit_pipelines_per_core: box_slice![vec![]; total_cores],
            scheduler_per_core: box_slice![
                box_slice![Scheduler::default(); num_schedulers];
                total_cores
            ],
            operand_collector_per_core: box_slice![None; total_cores],
            pending_register_writes_per_core: box_slice![vec![]; total_cores],
            l1_latency_queue_per_core: box_slice![vec![]; total_cores],
            l1_cache_per_core: box_slice![None; total_cores],
        }
    }
}
