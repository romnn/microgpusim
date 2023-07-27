use crate::ported;
use playground::types;

impl From<types::mf_type> for ported::mem_fetch::Kind {
    fn from(kind: types::mf_type) -> Self {
        use types::mf_type;
        match kind {
            mf_type::READ_REQUEST => ported::mem_fetch::Kind::READ_REQUEST,
            mf_type::WRITE_REQUEST => ported::mem_fetch::Kind::WRITE_REQUEST,
            mf_type::READ_REPLY => ported::mem_fetch::Kind::READ_REPLY,
            mf_type::WRITE_ACK => ported::mem_fetch::Kind::WRITE_ACK,
        }
    }
}

impl From<types::mem_access_type> for ported::mem_fetch::AccessKind {
    fn from(kind: types::mem_access_type) -> Self {
        use ported::mem_fetch::AccessKind;
        match kind {
            types::mem_access_type::GLOBAL_ACC_R => AccessKind::GLOBAL_ACC_R,
            types::mem_access_type::LOCAL_ACC_R => AccessKind::LOCAL_ACC_R,
            types::mem_access_type::CONST_ACC_R => AccessKind::CONST_ACC_R,
            types::mem_access_type::TEXTURE_ACC_R => AccessKind::TEXTURE_ACC_R,
            types::mem_access_type::GLOBAL_ACC_W => AccessKind::GLOBAL_ACC_W,
            types::mem_access_type::LOCAL_ACC_W => AccessKind::LOCAL_ACC_W,
            types::mem_access_type::L1_WRBK_ACC => AccessKind::L1_WRBK_ACC,
            types::mem_access_type::L2_WRBK_ACC => AccessKind::L2_WRBK_ACC,
            types::mem_access_type::INST_ACC_R => AccessKind::INST_ACC_R,
            types::mem_access_type::L1_WR_ALLOC_R => AccessKind::L1_WR_ALLOC_R,
            types::mem_access_type::L2_WR_ALLOC_R => AccessKind::L2_WR_ALLOC_R,
            other @ types::mem_access_type::NUM_MEM_ACCESS_TYPE => {
                panic!("bad mem access kind: {:?}", other)
            }
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Cache {
    pub lines: Vec<CacheBlock>,
}

impl std::fmt::Debug for Cache {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_list()
            .entries(
                self.lines
                    .iter()
                    .enumerate()
                    .filter(|(idx, line)| line.tag != 0), // .filter(|(idx, line)| line.status == CacheBlockStatus::VALID),
            )
            .finish()
    }
}

// impl From<ported::TagArray<ported::cache_block::LineCacheBlock>> for Cache {
impl<T> From<ported::TagArray<T>> for Cache {
    fn from(tag_array: ported::TagArray<T>) -> Self {
        Self {
            lines: tag_array.lines.into_iter().map(Into::into).collect(),
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum CacheBlockStatus {
    INVALID,
    RESERVED,
    VALID,
    MODIFIED,
}

impl From<ported::cache_block::Status> for CacheBlockStatus {
    fn from(status: ported::cache_block::Status) -> Self {
        use crate::ported::cache_block;
        match status {
            cache_block::Status::INVALID => Self::INVALID,
            cache_block::Status::RESERVED => Self::RESERVED,
            cache_block::Status::VALID => Self::VALID,
            cache_block::Status::MODIFIED => Self::MODIFIED,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct CacheBlock {
    pub tag: u64,
    pub block_addr: u64,
    pub status: CacheBlockStatus,
}

impl From<ported::cache_block::LineCacheBlock> for CacheBlock {
    fn from(block: ported::cache_block::LineCacheBlock) -> Self {
        Self {
            tag: block.tag,
            block_addr: block.block_addr,
            status: block.status.into(),
        }
    }
}

impl<'a> From<&'a playground::cache::cache_block_t> for CacheBlock {
    fn from(block: &'a playground::cache::cache_block_t) -> Self {
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
        }
    }
}

impl std::fmt::Debug for CacheBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}({}, {})", self.status, self.tag, self.block_addr)
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct WarpInstruction {
    pub opcode: String,
    pub pc: usize,
    pub warp_id: usize,
}

impl std::fmt::Debug for WarpInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}[pc={},warp={}]", self.opcode, self.pc, self.warp_id)
    }
}

impl From<ported::instruction::WarpInstruction> for WarpInstruction {
    fn from(instr: ported::instruction::WarpInstruction) -> Self {
        WarpInstruction {
            opcode: instr.opcode.to_string(),
            pc: instr.pc,
            warp_id: instr.warp_id,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct RegisterSet {
    // pub stage: ported::core::PipelineStage,
    pub stage: String,
    pub pipeline: Vec<Option<WarpInstruction>>,
}

impl RegisterSet {
    pub fn is_empty(&self) -> bool {
        self.num_instructions_in_pipeline() == 0
    }

    pub fn num_instructions_in_pipeline(&self) -> usize {
        self.pipeline.iter().filter_map(|x| x.as_ref()).count()
    }
}

impl std::fmt::Debug for RegisterSet {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}={:?}", self.stage, self.pipeline)
    }
}

impl From<ported::register_set::RegisterSet> for RegisterSet {
    fn from(reg: ported::register_set::RegisterSet) -> Self {
        let pipeline = reg
            .regs
            .into_iter()
            .map(|instr| match instr {
                Some(instr) => Some(instr.into()),
                None => None,
            })
            .collect();
        Self {
            stage: format!("{:?}", &reg.stage),
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
            warp_id: instr.warp_id() as usize,
        }
    }
}

impl<'a> From<playground::register_set::RegisterSet<'a>> for RegisterSet {
    fn from(reg: playground::register_set::RegisterSet<'a>) -> Self {
        Self {
            stage: reg.name(),
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

#[derive(strum::FromRepr, Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CollectorUnit {
    // pub in_ports: Vec<RegisterSet>,
    // pub out_ports: Vec<RegisterSet>,
    // pub ids: Vec<OperandCollectorUnitKind>,
    pub warp_id: Option<usize>,
    pub warp_instr: Option<WarpInstruction>,
    pub output_register: Option<RegisterSet>,
    // pub src_operands: Vec<Option<Operand>>, // ; MAX_REG_OPERANDS * 2],
    // pub not_ready: BitArr!(for MAX_REG_OPERANDS * 2),
    pub not_ready: String,
    pub reg_id: Option<usize>,
    pub kind: OperandCollectorUnitKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DispatchUnit {
    pub last_cu: usize,
    pub next_cu: usize,
    // pub sub_core_model: bool,
    // pub num_warp_schedulers: usize,
    pub kind: OperandCollectorUnitKind,
}

impl<'a> From<&playground::operand_collector::dispatch_unit_t> for DispatchUnit {
    fn from(unit: &playground::operand_collector::dispatch_unit_t) -> Self {
        Self {
            last_cu: unit.get_last_cu() as usize,
            next_cu: unit.get_next_cu() as usize,
            kind: OperandCollectorUnitKind::from_repr(unit.get_set_id()).unwrap(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
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

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct OperandCollector {
    pub ports: Vec<Port>,
    pub collector_units: Vec<CollectorUnit>,
    pub dispatch_units: Vec<DispatchUnit>,
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
            // fix: default endianness is different for rust bitvec and c++ std::bitset
            not_ready: cu.not_ready_mask().chars().rev().collect::<String>(),
            reg_id: cu.reg_id(),
        }
    }
}

impl<'a> From<playground::operand_collector::OperandCollector<'a>> for OperandCollector {
    fn from(opcoll: playground::operand_collector::OperandCollector<'a>) -> Self {
        use std::collections::HashSet;
        let skip: HashSet<_> = [
            OperandCollectorUnitKind::TENSOR_CORE_CUS,
            OperandCollectorUnitKind::SFU_CUS,
        ]
        .into_iter()
        .collect();
        let mut ports = opcoll
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

        Self {
            ports,
            collector_units,
            dispatch_units,
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct Scheduler {
    pub prioritized_warp_ids: Vec<(usize, usize)>,
    // pub prioritized_warp_ids: Vec<usize>,
    // pub prioritized_dynamic_warp_ids: Vec<usize>,
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
            // prioritized_dynamic_warp_ids,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct MemFetch {
    pub kind: ported::mem_fetch::Kind,
    pub access_kind: ported::mem_fetch::AccessKind,
    // cannot compare addr because its different between runs
    // addr: ported::address,
    pub relative_addr: Option<(usize, ported::address)>,
}

impl std::fmt::Debug for MemFetch {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}({:?}", self.kind, self.access_kind)?;
        if let Some((alloc_id, rel_addr)) = self.relative_addr {
            write!(f, "@{}+{}", alloc_id, rel_addr)?;
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

impl From<ported::mem_fetch::MemFetch> for MemFetch {
    fn from(fetch: ported::mem_fetch::MemFetch) -> Self {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Simulation {
    pub interconn_to_l2_queue: Vec<Vec<MemFetch>>,
    pub l2_to_interconn_queue: Vec<Vec<MemFetch>>,
    pub l2_to_dram_queue: Vec<Vec<MemFetch>>,
    pub dram_to_l2_queue: Vec<Vec<MemFetch>>,
    pub l2_cache: Vec<Option<Cache>>,
    pub dram_latency_queue: Vec<Vec<MemFetch>>,
    pub functional_unit_pipelines: Vec<Vec<RegisterSet>>,
    pub operand_collectors: Vec<Option<OperandCollector>>,
    pub schedulers: Vec<Vec<Scheduler>>,
}

impl Simulation {
    pub fn new(total_cores: usize, num_mem_partitions: usize, num_sub_partitions: usize) -> Self {
        Self {
            // per sub partition
            interconn_to_l2_queue: vec![vec![]; num_sub_partitions],
            l2_to_interconn_queue: vec![vec![]; num_sub_partitions],
            l2_to_dram_queue: vec![vec![]; num_sub_partitions],
            dram_to_l2_queue: vec![vec![]; num_sub_partitions],
            l2_cache: vec![None; num_sub_partitions],
            // per partition
            dram_latency_queue: vec![vec![]; num_mem_partitions],
            // per core
            functional_unit_pipelines: vec![vec![]; total_cores],
            schedulers: vec![vec![]; total_cores],
            operand_collectors: vec![None; total_cores],
        }
    }
}
