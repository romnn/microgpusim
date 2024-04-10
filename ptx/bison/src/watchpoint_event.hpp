#pragma once

#include <cstddef>

class ptx_thread_info;
class ptx_instruction;

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
