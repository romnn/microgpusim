#pragma once

struct OpcodeChar {
  OpcodeChar(unsigned m_opcode, unsigned m_opcode_category) {
    opcode = m_opcode;
    opcode_category = m_opcode_category;
  }
  unsigned opcode;
  unsigned opcode_category;
};
