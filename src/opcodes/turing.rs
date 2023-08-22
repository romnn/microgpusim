use super::Op;

pub mod op {
    /// Unique trace instruction opcodes for turing.
    #[derive(strum::FromRepr, Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
    pub enum Op {
        BMMA,
        MOVM,
        LDSM,
        R2UR,
        S2UR,
        UBMSK,
        UBREV,
        UCLEA,
        UFLO,
        UIADD3,
        UIMAD,
        UISETP,
        ULDC,
        ULEA,
        ULOP,
        ULOP3,
        ULOP32I,
        UMOV,
        UP2UR,
        UPLOP3,
        UPOPC,
        UPRMT,
        UPSETP,
        UR2UP,
        USEL,
        USGXT,
        USHF,
        USHL,
        USHR,
        VOTEU,
        SUATOM,
        SULD,
        SURED,
        SUST,
        BRXU,
        JMXU,
    }

    impl From<Op> for super::Op {
        fn from(op: Op) -> Self {
            Self::Turing(op)
        }
    }
}
