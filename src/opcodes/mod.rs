pub mod ampere;
pub mod kepler;
pub mod pascal;
pub mod turing;

use color_eyre::eyre;

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

/// Trace instruction opcodes for all hardware generations.
#[derive(strum::AsRefStr, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Op {
    NOP,
    // memory ops
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

    // alu ops
    FADD,
    FADD32I,
    FCHK,
    FFMA32I,
    FFMA,
    FMNMX,
    FMUL,
    FMUL32I,
    FSEL,
    FSET,
    FSETP,
    FSWZADD,
    MUFU,
    HADD2,
    HADD2_32I,
    HFMA2,
    HFMA2_32I,
    HMUL2,
    HMUL2_32I,
    HSET2,
    HSETP2,
    HMMA,
    DADD,
    DFMA,
    DMUL,
    DSETP,
    BMSK,
    BREV,
    FLO,
    IABS,
    IADD,
    IADD3,
    IADD32I,
    IDP,
    IDP4A,
    IMAD,
    IMMA,
    IMNMX,
    IMUL,
    IMUL32I,
    ISCADD,
    ISCADD32I,
    ISETP,
    LEA,
    LOP,
    LOP3,
    LOP32I,
    POPC,
    SHF,
    SHR,
    VABSDIFF,
    VABSDIFF4,
    VADD,
    F2F,
    F2I,
    I2F,
    I2I,
    I2IP,
    FRND,
    MOV,
    MOV32I,
    PRMT,
    SEL,
    SGXT,
    SHFL,
    PLOP3,
    PSETP,
    P2R,
    R2P,
    MATCH,
    QSPC,
    CCTL,
    CCTLL,
    ERRBAR,
    CCTLT,
    TEX,
    TLD,
    TLD4,
    TMML,
    TXD,
    TXQ,
    BMOV,
    BPT,
    BRA,
    BREAK,
    BRX,
    BSSY,
    BSYNC,
    CALL,
    EXIT,
    JMP,
    JMX,
    KILL,
    NANOSLEEP,
    RET,
    RPCMOV,
    RTT,
    WARPSYNC,
    YIELD,
    B2R,
    BAR,
    CS2R,
    CSMTEST,
    DEPBAR,
    GETLMEMBASE,
    LEPC,
    PMTRIG,
    R2B,
    S2R,
    SETCTAID,
    SETLMEMBASE,
    VOTE,
    VOTE_VTG,
    Pascal(pascal::op::Op),
    Turing(turing::op::Op),
    Kepler(kepler::op::Op),
    Ampere(ampere::op::Op),
}

impl std::fmt::Debug for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Op::Pascal(op) => op.fmt(f),
            Op::Turing(op) => op.fmt(f),
            Op::Kepler(op) => op.fmt(f),
            Op::Ampere(op) => op.fmt(f),
            op => write!(f, "{}", op.as_ref()),
        }
    }
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

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[repr(usize)]
pub enum ArchOp {
    /// No-op
    NO_OP,
    ALU_OP,
    SFU_OP,
    TENSOR_CORE_OP,
    /// Double precision
    DP_OP,
    /// Single precision
    SP_OP,
    INT_OP,
    ALU_SFU_OP,
    /// Load operation
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
    SPECIALIZED_UNIT_1_OP,
    SPECIALIZED_UNIT_2_OP,
    SPECIALIZED_UNIT_3_OP,
    SPECIALIZED_UNIT_4_OP,
    SPECIALIZED_UNIT_5_OP,
    SPECIALIZED_UNIT_6_OP,
    SPECIALIZED_UNIT_7_OP,
    SPECIALIZED_UNIT_8_OP,
}

pub const SPEC_UNIT_START_ID: usize = ArchOp::SPECIALIZED_UNIT_1_OP as usize;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Opcode {
    pub op: Op,
    pub category: ArchOp,
}

impl std::fmt::Display for Opcode {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self.op)
    }
}

pub type OpcodeMap = phf::Map<&'static str, Opcode>;

pub fn get_opcode_map(
    config: &trace_model::command::KernelLaunch,
) -> eyre::Result<&'static OpcodeMap> {
    let version = BinaryVersion::from_repr(config.binary_version);
    let version = version.ok_or(eyre::eyre!(
        "unknown binary version {}",
        config.binary_version
    ))?;
    #[allow(clippy::match_same_arms)]
    match version {
        BinaryVersion::AMPERE_RTX | BinaryVersion::AMPERE_A100 => Ok(&ampere::OPCODES),
        BinaryVersion::PASCAL_P100 | BinaryVersion::PASCAL_TITANX => Ok(&pascal::OPCODES),
        other => unimplemented!("binary version {other:?} not supported"),
    }
}
