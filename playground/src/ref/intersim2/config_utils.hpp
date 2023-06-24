// $Id: config_utils.hpp 5188 2012-08-30 00:31:31Z dub $

/*
 Copyright (c) 2007-2012, Trustees of The Leland Stanford Junior University
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.
 Redistributions in binary form must reproduce the above copyright notice, this
 list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef _CONFIG_UTILS_HPP_
#define _CONFIG_UTILS_HPP_

#include "booksim.hpp"

#include <cstdio>
#include <map>
#include <string>
#include <vector>

// TODO: ptx replace with config
// extern "C" int yyparse();

class Configuration {
  // static Configuration *theConfig;
  // FILE *_config_file;
  // std::string _config_string;

protected:
  std::map<std::string, std::string> _str_map;
  std::map<std::string, int> _int_map;
  std::map<std::string, double> _float_map;

public:
  Configuration();

  void AddStrField(std::string const &field, std::string const &value);

  void Assign(std::string const &field, std::string const &value);
  void Assign(std::string const &field, int value);
  void Assign(std::string const &field, double value);

  const std::string &GetStr(std::string const &field) const;
  int GetInt(std::string const &field) const;
  double GetFloat(std::string const &field) const;

  std::vector<std::string> GetStrArray(const std::string &field) const;
  std::vector<int> GetIntArray(const std::string &field) const;
  std::vector<double> GetFloatArray(const std::string &field) const;

  void ParseFile(std::string const &filename);
  void ParseString(std::string const &str);
  // int Input(char *line, int max_size);
  void ParseError(std::string const &msg, unsigned int lineno = 0) const;

  void WriteFile(std::string const &filename);
  void WriteMatlabFile(std::ostream *o) const;

  inline const std::map<std::string, std::string> &GetStrMap() const {
    return _str_map;
  }
  inline const std::map<std::string, int> &GetIntMap() const {
    return _int_map;
  }
  inline const std::map<std::string, double> &GetFloatMap() const {
    return _float_map;
  }

  // static Configuration *GetTheConfig();
};

bool ParseArgs(Configuration *cf, int argc, char **argv);

std::vector<std::string> tokenize_str(std::string const &data);
std::vector<int> tokenize_int(std::string const &data);
std::vector<double> tokenize_float(std::string const &data);

#endif
