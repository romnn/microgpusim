use bitvec::field::BitField;
use bitvec::BitArr;
use serde::{Deserialize, Serialize};

pub type Inner = BitArr!(for super::WARP_SIZE, in u32);

/// Thread active mask.
///
/// Bitmask where a 1 at position i means that thread i is active for the current instruction.
#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct ActiveMask(Inner);

impl ActiveMask {
    /// Active mask with all threads inactive
    pub const ZERO: Self = ActiveMask(Inner::ZERO);

    #[must_use] pub fn all_ones() -> Self {
        Self::ZERO.inverted()
    }

    #[must_use] pub fn as_u32(&self) -> u32 {
        self.0.load()
    }

    #[must_use] pub fn inverted(mut self) -> Self {
        self.0 = !self.0;
        self
    }
}

impl<I> From<I> for ActiveMask
where
    I: funty::Integral,
{
    fn from(value: I) -> Self {
        let mut active_mask = Inner::ZERO;
        active_mask.store(value);
        Self(active_mask)
    }
}

impl std::ops::Deref for ActiveMask {
    type Target = Inner;
    fn deref(&self) -> &Inner {
        &self.0
    }
}

impl std::ops::DerefMut for ActiveMask {
    fn deref_mut(&mut self) -> &mut Inner {
        &mut self.0
    }
}

impl Serialize for ActiveMask {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_u32(self.0.load())
    }
}

struct Visitor;

impl<'de> serde::de::Visitor<'de> for Visitor {
    type Value = ActiveMask;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("an integer between 0 and 2^31")
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(ActiveMask::from(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(ActiveMask::from(value))
    }

    fn visit_u32<E>(self, value: u32) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(ActiveMask::from(value))
    }

    fn visit_i32<E>(self, value: i32) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(ActiveMask::from(value))
    }
}

impl<'de> Deserialize<'de> for ActiveMask {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_u32(Visitor)
    }
}

impl std::fmt::Display for ActiveMask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.to_bit_string())
    }
}

/// Format as a binary string.
pub trait ToBitString {
    fn to_bit_string(&self) -> String;
}

impl<A, O> ToBitString for bitvec::slice::BitSlice<A, O>
where
    A: bitvec::store::BitStore,
    O: bitvec::order::BitOrder,
{
    fn to_bit_string(&self) -> String {
        self.iter()
            .rev()
            .map(|b| if *b { "1" } else { "0" })
            .collect::<Vec<_>>()
            .join("")
    }
}
