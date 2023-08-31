use playground::types;
use serde::Serialize;
use utils::box_slice;

impl From<types::mem_fetch::mf_type> for crate::mem_fetch::Kind {
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

impl From<types::mem_fetch::mem_access_type> for crate::mem_fetch::AccessKind {
    fn from(kind: types::mem_fetch::mem_access_type) -> Self {
        use crate::mem_fetch::AccessKind;
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
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_list()
            .entries(
                self.lines.iter().enumerate(), // we only show valid tags
                                               // .filter(|(idx, line)| line.tag != 0),
            )
            .finish()
    }
}

impl From<crate::tag_array::TagArray<crate::cache::block::Line>> for Cache {
    fn from(tag_array: crate::tag_array::TagArray<crate::cache::block::Line>) -> Self {
        Self {
            lines: tag_array.lines.into_iter().map(Into::into).collect(),
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
    pub tag: u64,
    pub block_addr: u64,
    pub status: CacheBlockStatus,
    pub last_accessed: u64,
}

impl From<crate::cache::block::Line> for CacheBlock {
    fn from(block: crate::cache::block::Line) -> Self {
        Self {
            tag: block.tag,
            block_addr: block.block_addr,
            status: block.status.into(),
            last_accessed: block.last_access_time,
        }
    }
}

impl From<&playground::cache::cache_block_t> for CacheBlock {
    fn from(block: &playground::cache::cache_block_t) -> Self {
        let status = if block.is_valid_line() {
            CacheBlockStatus::VALID
        } else if block.is_invalid_line() {
            CacheBlockStatus::INVALID
        } else if block.is_reserved_line() {
            CacheBlockStatus::RESERVED
        } else if block.is_modified_line() {
            CacheBlockStatus::MODIFIED
        } else {
            unreachable!()
        };
        Self {
            status,
            tag: block.get_tag(),
            block_addr: block.get_block_addr(),
            last_accessed: block.get_last_access_time(),
        }
    }
}

impl std::fmt::Debug for CacheBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{:?}(tag={}, block={}, accessed={})",
            self.status, self.tag, self.block_addr, self.last_accessed
        )
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
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}[pc={},warp={}]", self.opcode, self.pc, self.warp_id)
    }
}

impl From<crate::instruction::WarpInstruction> for WarpInstruction {
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
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.num_instructions_in_pipeline() == 0
    }

    #[must_use]
    pub fn num_instructions_in_pipeline(&self) -> usize {
        self.pipeline
            .iter()
            .filter_map(std::option::Option::as_ref)
            .count()
    }
}

impl std::fmt::Debug for RegisterSet {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}={:?}", self.name, self.pipeline)
    }
}

impl From<crate::register_set::RegisterSet> for RegisterSet {
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
    fn from(arbiter: &playground::operand_collector::arbiter_t) -> Self {
        Self {
            last_cu: arbiter.get_last_cu() as usize,
        }
    }
}

impl<'a> From<playground::operand_collector::OperandCollector<'a>> for OperandCollector {
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
                .zip(prioritized_dynamic_warp_ids.into_iter())
                .collect(),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Serialize)]
pub struct MemFetch {
    pub kind: crate::mem_fetch::Kind,
    pub access_kind: crate::mem_fetch::AccessKind,
    // cannot compare addr because its different between runs
    // addr: crate::address,
    pub relative_addr: Option<(usize, crate::address)>,
}

impl std::fmt::Debug for MemFetch {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}({:?}", self.kind, self.access_kind)?;
        if let Some((alloc_id, rel_addr)) = self.relative_addr {
            write!(f, "@{alloc_id}+{rel_addr}")?;
        }
        write!(f, ")")
    }
}

impl<'a> From<playground::mem_fetch::MemFetch<'a>> for MemFetch {
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
    fn from(fetch: crate::mem_fetch::MemFetch) -> Self {
        let addr = fetch.addr();
        Self {
            kind: fetch.kind,
            access_kind: *fetch.access_kind(),
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
        }
    }
}
