#include "trace_parser.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

void split(const std::string &str, std::vector<std::string> &cont,
           char delimi = ' ') {
  std::stringstream ss(str);
  std::string token;
  while (std::getline(ss, token, delimi)) {
    cont.push_back(token);
  }
}

trace_parser::trace_parser(const char *kernellist_filepath) {
  kernellist_filename = kernellist_filepath;
}

std::vector<trace_command> trace_parser::parse_commandlist_file() {
  std::ifstream fs;
  fs.open(kernellist_filename);

  if (!fs.is_open()) {
    std::cout << "Unable to open file: " << kernellist_filename << std::endl;
    exit(1);
  }

  std::string directory(kernellist_filename);
  const size_t last_slash_idx = directory.rfind('/');
  if (std::string::npos != last_slash_idx) {
    directory = directory.substr(0, last_slash_idx);
  }

  std::string line, filepath;
  std::vector<trace_command> commandlist;
  while (!fs.eof()) {
    getline(fs, line);
    if (line.empty())
      continue;
    else if (line.substr(0, 10) == "MemcpyHtoD") {
      trace_command command;
      command.command_string = line;
      command.m_type = command_type::cpu_gpu_mem_copy;
      commandlist.push_back(command);
    } else if (line.substr(0, 6) == "kernel") {
      trace_command command;
      command.m_type = command_type::kernel_launch;
      filepath = directory + "/" + line;
      command.command_string = filepath;
      commandlist.push_back(command);
    }
    // ignore gpu_to_cpu_memory_cpy
  }

  fs.close();
  return commandlist;
}

void trace_parser::parse_memcpy_info(const std::string &memcpy_command,
                                     size_t &address, size_t &count) {
  std::vector<std::string> params;
  split(memcpy_command, params, ',');
  assert(params.size() == 3);
  std::stringstream ss;
  ss.str(params[1]);
  ss >> std::hex >> address;
  ss.clear();
  ss.str(params[2]);
  ss >> std::dec >> count;
}

kernel_trace_t *
trace_parser::parse_kernel_info(const std::string &kerneltraces_filepath) {
  kernel_trace_t *kernel_info = new kernel_trace_t;
  kernel_info->enable_lineinfo = 0; // default disabled
  kernel_info->ifs = new std::ifstream;
  std::ifstream *ifs = kernel_info->ifs;
  ifs->open(kerneltraces_filepath.c_str());

  if (!ifs->is_open()) {
    std::cout << "Unable to open file: " << kerneltraces_filepath << std::endl;
    exit(1);
  }

  std::cout << "Processing kernel " << kerneltraces_filepath << std::endl;

  std::string line;

  while (!ifs->eof()) {
    getline(*ifs, line);

    if (line.length() == 0) {
      continue;
    } else if (line[0] == '#') {
      // the trace format, ignore this and assume fixed format for now
      break; // the begin of the instruction stream
    } else if (line[0] == '-') {
      std::stringstream ss;
      std::string string1, string2;

      ss.str(line);
      ss.ignore();
      ss >> string1 >> string2;

      if (string1 == "kernel" && string2 == "name") {
        const size_t equal_idx = line.find('=');
        kernel_info->kernel_name = line.substr(equal_idx + 2);
      } else if (string1 == "kernel" && string2 == "id") {
        sscanf(line.c_str(), "-kernel id = %d", &kernel_info->kernel_id);
      } else if (string1 == "grid" && string2 == "dim") {
        sscanf(line.c_str(), "-grid dim = (%d,%d,%d)", &kernel_info->grid_dim_x,
               &kernel_info->grid_dim_y, &kernel_info->grid_dim_z);
      } else if (string1 == "block" && string2 == "dim") {
        sscanf(line.c_str(), "-block dim = (%d,%d,%d)", &kernel_info->tb_dim_x,
               &kernel_info->tb_dim_y, &kernel_info->tb_dim_z);
      } else if (string1 == "shmem" && string2 == "=") {
        sscanf(line.c_str(), "-shmem = %d", &kernel_info->shmem);
      } else if (string1 == "nregs") {
        sscanf(line.c_str(), "-nregs = %d", &kernel_info->nregs);
      } else if (string1 == "cuda" && string2 == "stream") {
        sscanf(line.c_str(), "-cuda stream id = %lu",
               &kernel_info->cuda_stream_id);
      } else if (string1 == "binary" && string2 == "version") {
        sscanf(line.c_str(), "-binary version = %d",
               &kernel_info->binary_verion);
      } else if (string1 == "enable" && string2 == "lineinfo") {
        sscanf(line.c_str(), "-enable lineinfo = %d",
               &kernel_info->enable_lineinfo);
      } else if (string1 == "nvbit" && string2 == "version") {
        const size_t equal_idx = line.find('=');
        kernel_info->nvbit_verion = line.substr(equal_idx + 1);

      } else if (string1 == "accelsim" && string2 == "tracer") {
        sscanf(line.c_str(), "-accelsim tracer version = %d",
               &kernel_info->trace_verion);

      } else if (string1 == "shmem" && string2 == "base_addr") {
        const size_t equal_idx = line.find('=');
        ss.str(line.substr(equal_idx + 1));
        ss >> std::hex >> kernel_info->shmem_base_addr;

      } else if (string1 == "local" && string2 == "mem") {
        const size_t equal_idx = line.find('=');
        ss.str(line.substr(equal_idx + 1));
        ss >> std::hex >> kernel_info->local_base_addr;
      }
      std::cout << line << std::endl;
      continue;
    }
  }

  // do not close the file ifs, the kernel_finalizer will close it
  return kernel_info;
}

void trace_parser::kernel_finalizer(kernel_trace_t *trace_info) {
  assert(trace_info);
  assert(trace_info->ifs);
  if (trace_info->ifs->is_open())
    trace_info->ifs->close();
  delete trace_info->ifs;
  delete trace_info;
}

void trace_parser::get_next_threadblock_traces(
    std::vector<std::vector<inst_trace_t> *> threadblock_traces,
    unsigned trace_version, unsigned enable_lineinfo, std::ifstream *ifs) {
  for (unsigned i = 0; i < threadblock_traces.size(); ++i) {
    threadblock_traces[i]->clear();
  }

  unsigned block_id_x = 0, block_id_y = 0, block_id_z = 0;
  bool start_of_tb_stream_found = false;

  unsigned warp_id = 0;
  unsigned insts_num = 0;
  unsigned inst_count = 0;

  while (!ifs->eof()) {
    std::string line;
    std::stringstream ss;
    std::string string1, string2;

    getline(*ifs, line);

    if (line.length() == 0) {
      continue;
    } else {
      ss.str(line);
      ss >> string1 >> string2;
      if (string1 == "#BEGIN_TB") {
        if (!start_of_tb_stream_found) {
          start_of_tb_stream_found = true;
        } else
          assert(0 &&
                 "Parsing error: thread block start before the previous one "
                 "finishes");
      } else if (string1 == "#END_TB") {
        assert(start_of_tb_stream_found);
        break; // end of TB stream
      } else if (string1 == "thread" && string2 == "block") {
        assert(start_of_tb_stream_found);
        sscanf(line.c_str(), "thread block = %d,%d,%d", &block_id_x,
               &block_id_y, &block_id_z);
        std::cout << line << std::endl;
      } else if (string1 == "warp") {
        // the start of new warp stream
        assert(start_of_tb_stream_found);
        sscanf(line.c_str(), "warp = %d", &warp_id);
      } else if (string1 == "insts") {
        assert(start_of_tb_stream_found);
        sscanf(line.c_str(), "insts = %d", &insts_num);
        threadblock_traces[warp_id]->resize(
            insts_num); // allocate all the space at once
        inst_count = 0;
      } else {
        assert(start_of_tb_stream_found);
        threadblock_traces[warp_id]
            ->at(inst_count)
            .parse_from_string(line, trace_version, enable_lineinfo);
        inst_count++;
      }
    }
  }
}
