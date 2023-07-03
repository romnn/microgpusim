#pragma once

#include "type_info_key.hpp"

class symbol_table;

class type_info {
 public:
  type_info(symbol_table *scope, type_info_key t) { m_type_info = t; }
  const type_info_key &get_key() const { return m_type_info; }

 private:
  symbol_table *m_scope;
  type_info_key m_type_info;
};
