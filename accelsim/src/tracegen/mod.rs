#![allow(warnings)]

pub mod reader;
pub mod writer;

pub const WARP_SIZE: usize = 32;

#[derive(strum::FromRepr, Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum AddressFormat {
    ListAll = 0,
    BaseStride = 1,
    BaseDelta = 2,
}

type ActiveMask = bitvec::BitArr!(for 32, in u32);

fn parse_active_mask(raw_mask: u32) -> ActiveMask {
    use bitvec::{access, array::BitArray, field::BitField, BitArr};
    let mut active_mask = BitArray::ZERO;
    active_mask.store(raw_mask);
    active_mask
}

fn is_number(s: &str) -> bool {
    !s.is_empty() && s.chars().all(char::is_numeric)
}

fn get_data_width_from_opcode(opcode: &str) -> Result<u32, std::num::ParseIntError> {
    let opcode_tokens: Vec<_> = opcode
        .split(".")
        .map(|t| t.trim())
        .filter(|t| !t.is_empty())
        .collect();

    for token in opcode_tokens {
        assert!(!token.is_empty());

        if is_number(token) {
            return Ok(token.parse::<u32>()? / 8);
        } else if let Some('U') = token.chars().nth(0) {
            if is_number(&token[1..token.len()]) {
                // handle the U* case
                return Ok(token[1..token.len()].parse::<u32>()? / 8);
            }
        }
    }
    // default is 4 bytes
    Ok(4)
}
