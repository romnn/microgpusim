#pragma once

#include <cstddef>

class symbol_table;
class function_info;
class symbol;

struct stack_entry {
  stack_entry() {
    m_symbol_table = NULL;
    m_func_info = NULL;
    m_PC = 0;
    m_RPC = -1;
    m_return_var_src = NULL;
    m_return_var_dst = NULL;
    m_call_uid = 0;
    m_valid = false;
  }
  stack_entry(symbol_table *s, function_info *f, unsigned pc, unsigned rpc,
              const symbol *return_var_src, const symbol *return_var_dst,
              unsigned call_uid) {
    m_symbol_table = s;
    m_func_info = f;
    m_PC = pc;
    m_RPC = rpc;
    m_return_var_src = return_var_src;
    m_return_var_dst = return_var_dst;
    m_call_uid = call_uid;
    m_valid = true;
  }

  bool m_valid;
  symbol_table *m_symbol_table;
  function_info *m_func_info;
  unsigned m_PC;
  unsigned m_RPC;
  const symbol *m_return_var_src;
  const symbol *m_return_var_dst;
  unsigned m_call_uid;
};
