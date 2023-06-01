#pragma once

#include <string>

#include "command_type.hpp"

struct trace_command {
  std::string command_string;
  command_type m_type;
};
