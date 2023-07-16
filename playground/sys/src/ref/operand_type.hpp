#pragma once

enum operand_type {
  reg_t,
  vector_t,
  builtin_t,
  address_t,
  memory_t,
  float_op_t,
  double_op_t,
  int_t,
  unsigned_t,
  symbolic_t,
  label_t,
  v_reg_t,
  v_float_op_t,
  v_double_op_t,
  v_int_t,
  v_unsigned_t,
  undef_t
};
