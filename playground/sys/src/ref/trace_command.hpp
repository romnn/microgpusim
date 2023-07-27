#pragma once

#include <string>

#include "command_type.hpp"

struct trace_command {
  std::string command_string;
  command_type m_type;

  const std::string &get_command() const { return command_string; };
  command_type get_type() const { return m_type; };
};
