#pragma once

#include <cstdio>

// #include "ptx_thread_info.hpp"
#include "ptx_instruction.hpp"

class watchpoint_event {
public:
  watchpoint_event() {
    m_thread = NULL;
    m_inst = NULL;
  }
  watchpoint_event(const ptx_thread_info *thd, const ptx_instruction *pI) {
    m_thread = thd;
    m_inst = pI;
  }
  const ptx_thread_info *thread() const { return m_thread; }
  const ptx_instruction *inst() const { return m_inst; }

private:
  const ptx_thread_info *m_thread;
  const ptx_instruction *m_inst;
};
