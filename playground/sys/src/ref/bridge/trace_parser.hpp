#pragma once

#include "rust/cxx.h"
#include "../trace_parser.hpp"

std::unique_ptr<trace_parser> new_trace_parser(
    rust::String kernellist_filepath);
// std::unique_ptr<trace_parser> new_trace_parser(std::string
// kernellist_filepath);
