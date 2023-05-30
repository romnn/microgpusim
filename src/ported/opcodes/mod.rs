mod ampere;

use color_eyre::eyre;
use std::collections::HashMap;
use trace_model::KernelLaunch;

#[derive(strum::FromRepr, Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[repr(i32)]
enum BinaryVersion {
    AMPERE_RTX = 86,
    AMPERE_A100 = 80,
    VOLTA = 70,
    PASCAL_TITANX = 61,
    PASCAL_P100 = 60,
    KEPLER = 35,
    TURING = 75,
}

#[derive(strum::FromRepr, Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Op {
    LD,
    LDC,
    LDG,
    LDL,
    LDS,
    LDSM,
    ST,
    STG,
    STL,
    STS,
    ATOM,
    ATOMS,
    ATOMG,
    RED,
    MEMBAR,
    LDGSTS,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum ArchOpKind {
    UN_OP,
    INT_OP,
    FP_OP,
}

// pub fn get_operand_kind(op_type op, special_ops sp_op) -> ArchOpKind {
//   match op {
//     case SP_OP:
//     case SFU_OP:
//     case SPECIALIZED_UNIT_2_OP:
//     case SPECIALIZED_UNIT_3_OP:
//     case DP_OP:
//     case LOAD_OP:
//     case STORE_OP:
//       return FP_OP;
//     case INTP_OP:
//     case SPECIALIZED_UNIT_4_OP:
//       return INT_OP;
//     case ALU_OP:
//       if ((sp_op == FP__OP) || (sp_op == TEX__OP) || (sp_op == OTHER_OP))
//         return FP_OP;
//       else if (sp_op == INT__OP)
//         return INT_OP;
//     default:
//       return UN_OP;
//   }
// }

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum ArchOp {
    NO_OP,
    ALU_OP,
    SFU_OP,
    TENSOR_CORE_OP,
    DP_OP,
    SP_OP,
    INTP_OP,
    ALU_SFU_OP,
    LOAD_OP,
    TENSOR_CORE_LOAD_OP,
    TENSOR_CORE_STORE_OP,
    STORE_OP,
    BRANCH_OP,
    BARRIER_OP,
    MEMORY_BARRIER_OP,
    CALL_OPS,
    RET_OPS,
    EXIT_OPS,
}

#[derive(Clone, Copy, Debug)]
pub struct Opcode {
    pub op: Op,
    pub category: ArchOp,
}

pub type OpcodeMap = phf::Map<&'static str, Opcode>;

pub fn get_opcode_map(config: &KernelLaunch) -> eyre::Result<&'static OpcodeMap> {
    type BV = BinaryVersion;
    let version = BV::from_repr(config.binary_version);
    let version = version.ok_or(eyre::eyre!(
        "unknown binary version {}",
        config.binary_version
    ))?;
    match version {
        BV::AMPERE_RTX | BV::AMPERE_A100 => Ok(&ampere::OPCODES),
        BV::PASCAL_P100 | BV::PASCAL_TITANX => Ok(&ampere::OPCODES),
        _ => unimplemented!(),
    }
}
