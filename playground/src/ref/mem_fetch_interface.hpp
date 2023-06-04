#pragma once

class mem_fetch;

class mem_fetch_interface {
public:
  virtual bool full(unsigned size, bool write) const = 0;
  virtual void push(mem_fetch *mf) = 0;
};
