mod ampere;

use super::instruction::ArchOp;
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
