use super::{Op};

pub mod op {
    /// Unique trace instruction opcodes for kepler.
    #[derive(strum::FromRepr, Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
    pub enum Op {
        FCMP,
        FSWZ,
        ISAD,
        LDSLK,
        STSCUL,
        SUCLAMP,
        SUBFM,
        SUEAU,
        SULDGA,
        SUSTGA,
        ISUB,
    }

    impl From<Op> for super::Op {
        fn from(op: Op) -> Self {
            Self::Kepler(op)
        }
    }
}
