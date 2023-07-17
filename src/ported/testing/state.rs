use crate::ported;

impl From<playground::mem_fetch::mf_type> for ported::mem_fetch::Kind {
    fn from(kind: playground::mem_fetch::mf_type) -> Self {
        use playground::mem_fetch::mf_type;
        match kind {
            mf_type::READ_REQUEST => ported::mem_fetch::Kind::READ_REQUEST,
            mf_type::WRITE_REQUEST => ported::mem_fetch::Kind::WRITE_REQUEST,
            mf_type::READ_REPLY => ported::mem_fetch::Kind::READ_REPLY,
            mf_type::WRITE_ACK => ported::mem_fetch::Kind::WRITE_ACK,
        }
    }
}

impl From<playground::mem_fetch::mem_access_type> for ported::mem_fetch::AccessKind {
    fn from(kind: playground::mem_fetch::mem_access_type) -> Self {
        use playground::mem_fetch::mem_access_type;
        match kind {
            mem_access_type::GLOBAL_ACC_R => ported::mem_fetch::AccessKind::GLOBAL_ACC_R,
            mem_access_type::LOCAL_ACC_R => ported::mem_fetch::AccessKind::LOCAL_ACC_R,
            mem_access_type::CONST_ACC_R => ported::mem_fetch::AccessKind::CONST_ACC_R,
            mem_access_type::TEXTURE_ACC_R => ported::mem_fetch::AccessKind::TEXTURE_ACC_R,
            mem_access_type::GLOBAL_ACC_W => ported::mem_fetch::AccessKind::GLOBAL_ACC_W,
            mem_access_type::LOCAL_ACC_W => ported::mem_fetch::AccessKind::LOCAL_ACC_W,
            mem_access_type::L1_WRBK_ACC => ported::mem_fetch::AccessKind::L1_WRBK_ACC,
            mem_access_type::L2_WRBK_ACC => ported::mem_fetch::AccessKind::L2_WRBK_ACC,
            mem_access_type::INST_ACC_R => ported::mem_fetch::AccessKind::INST_ACC_R,
            mem_access_type::L1_WR_ALLOC_R => ported::mem_fetch::AccessKind::L1_WR_ALLOC_R,
            mem_access_type::L2_WR_ALLOC_R => ported::mem_fetch::AccessKind::L2_WR_ALLOC_R,
            other @ mem_access_type::NUM_MEM_ACCESS_TYPE => {
                panic!("bad mem access kind: {:?}", other)
            }
        }
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
                Some(instr) => Some(WarpInstruction {
                    opcode: instr.opcode.to_string(),
                    pc: instr.pc,
                    warp_id: instr.warp_id,
                }),
                None => None,
            })
            .collect();
        Self {
            stage: format!("{:?}", &reg.stage),
            pipeline,
        }
    }
}

// impl<'a> From<playground::main::pipeline_stage_name_t> for ported::core::PipelineStage {
//     fn from(stage: playground::main::pipeline_stage_name_t) -> Self {
//         use playground::main::pipeline_stage_name_t;
//         match stage {
//             // pipeline_stage_name_t::ID_OC_SP => Self::ID_OC_SP,
//             // pipeline_stage_name_t::ID_OC_DP => Self::ID_OC_DP,
//             // pipeline_stage_name_t::ID_OC_INT => Self::ID_OC_INT,
//             // pipeline_stage_name_t::ID_OC_SFU => Self::ID_OC_SFU,
//             // pipeline_stage_name_t::ID_OC_MEM => Self::ID_OC_MEM,
//             pipeline_stage_name_t::OC_EX_SP => Self::OC_EX_SP,
//             // pipeline_stage_name_t::OC_EX_DP => Self::OC_EX_DP,
//             // pipeline_stage_name_t::OC_EX_INT => Self::OC_EX_INT,
//             // pipeline_stage_name_t::OC_EX_SFU => Self::OC_EX_SFU,
//             pipeline_stage_name_t::OC_EX_MEM => Self::OC_EX_MEM,
//             pipeline_stage_name_t::EX_WB => Self::EX_WB,
//             // pipeline_stage_name_t::ID_OC_TENSOR_CORE => Self::ID_OC_TENSOR_CORE,
//             // pipeline_stage_name_t::OC_EX_TENSOR_CORE => Self::OC_EX_TENSOR_CORE,
//             other => panic!("bad pipeline stage {:?}", other),
//         }
//     }
// }

impl<'a> From<playground::RegisterSet<'a>> for RegisterSet {
    fn from(reg: playground::RegisterSet<'a>) -> Self {
        Self {
            stage: reg.name(),
            pipeline: reg
                .registers()
                .into_iter()
                .map(|instr| {
                    if instr.empty() {
                        None
                    } else {
                        let opcode = instr.opcode_str().trim_start_matches("OP_").to_string();
                        Some(WarpInstruction {
                            opcode,
                            pc: instr.get_pc() as usize,
                            warp_id: instr.warp_id() as usize,
                        })
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
    // pub warp_instr: Option<WarpInstruction>,
    pub output_register: Option<RegisterSet>,
    // pub src_operands: Vec<Option<Operand>>, // ; MAX_REG_OPERANDS * 2],
    // pub not_ready: BitArr!(for MAX_REG_OPERANDS * 2),
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

impl<'a> From<&playground::main::dispatch_unit_t> for DispatchUnit {
    fn from(unit: &playground::main::dispatch_unit_t) -> Self {
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

impl<'a> From<playground::Port<'a>> for Port {
    fn from(port: playground::Port<'a>) -> Self {
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

impl<'a> From<playground::CollectorUnit<'a>> for CollectorUnit {
    fn from(cu: playground::CollectorUnit<'a>) -> Self {
        Self {
            kind: OperandCollectorUnitKind::from_repr(cu.set_id()).unwrap(),
            warp_id: cu.warp_id(),
            output_register: cu.output_register().map(Into::into),
            reg_id: cu.reg_id(),
        }
    }
}

impl<'a> From<playground::OperandCollector<'a>> for OperandCollector {
    fn from(opcoll: playground::OperandCollector<'a>) -> Self {
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

impl<'a> From<playground::MemFetch<'a>> for MemFetch {
    fn from(fetch: playground::MemFetch<'a>) -> Self {
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
    pub dram_latency_queue: Vec<Vec<MemFetch>>,
    pub functional_unit_pipelines: Vec<Vec<RegisterSet>>,
    // pub operand_collectors: Vec<Vec<OperandCollector>>,
    pub operand_collectors: Vec<Option<OperandCollector>>,
}

impl Simulation {
    pub fn new(total_cores: usize, num_mem_partitions: usize, num_sub_partitions: usize) -> Self {
        Self {
            // per sub partition
            interconn_to_l2_queue: vec![vec![]; num_sub_partitions],
            l2_to_interconn_queue: vec![vec![]; num_sub_partitions],
            l2_to_dram_queue: vec![vec![]; num_sub_partitions],
            dram_to_l2_queue: vec![vec![]; num_sub_partitions],
            // per partition
            dram_latency_queue: vec![vec![]; num_mem_partitions],
            // per core
            functional_unit_pipelines: vec![vec![]; total_cores],
            operand_collectors: vec![None; total_cores],
        }
    }
}
