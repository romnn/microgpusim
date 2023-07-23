#include "trace_parser.hpp"

std::unique_ptr<trace_parser> new_trace_parser(
    rust::String kernellist_filepath) {
  // std::string kernellist_filepath) {
  return std::make_unique<trace_parser>(kernellist_filepath.c_str());
}
