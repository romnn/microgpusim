use super::{ArchOp, Op, Opcode, OpcodeMap};

pub static OPCODES: OpcodeMap = phf::phf_map! {
    "LD" => Opcode { op: Op::LD, category: ArchOp::LOAD_OP },
    // For now, we ignore constant loads, consider it as ALU_OP, TO DO
    "LDC" => Opcode { op: Op::LDC, category: ArchOp::ALU_OP},
    "LDG" => Opcode { op: Op::LDG, category: ArchOp::LOAD_OP },
    "LDL" => Opcode { op: Op::LDL, category: ArchOp::LOAD_OP },
    "LDS" => Opcode { op: Op::LDS, category: ArchOp::LOAD_OP },
    "LDSM" => Opcode { op: Op::LDSM, category: ArchOp::LOAD_OP },
    "ST" => Opcode { op: Op::ST, category: ArchOp::STORE_OP },
    "STG" => Opcode { op: Op::STG, category: ArchOp::STORE_OP },
    "STL" => Opcode { op: Op::STL, category: ArchOp::STORE_OP },
    "STS" => Opcode { op: Op::STS, category: ArchOp::STORE_OP },
    "ATOM" => Opcode { op: Op::ATOM, category: ArchOp::STORE_OP },
    "ATOMS" => Opcode { op: Op::ATOMS, category: ArchOp::STORE_OP },
    "ATOMG" => Opcode { op: Op::ATOMG, category: ArchOp::STORE_OP },
    "RED" => Opcode { op: Op::RED, category: ArchOp::STORE_OP },
    "MEMBAR" => Opcode { op: Op::MEMBAR, category: ArchOp::MEMORY_BARRIER_OP },
    "LDGSTS" => Opcode { op: Op::LDGSTS, category: ArchOp::LOAD_OP },
};
