// $Id: config_utils.cpp 5188 2012-08-30 00:31:31Z dub $
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

/*config_utils.cpp
 *
 *The configuration object which contained the parsed data from the
 *configuration file
 */

#include "config_utils.hpp"
#include "booksim.hpp"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

#include "config.parser.tab.h"  // parser

#include "config.lex.h"  // lexer

Configuration::Configuration() {}

void Configuration::AddStrField(std::string const &field,
                                std::string const &value) {
  _str_map[field] = value;
}

void Configuration::Assign(std::string const &field, std::string const &value) {
  std::map<std::string, std::string>::const_iterator match;

  match = _str_map.find(field);
  if (match != _str_map.end()) {
    _str_map[field] = value;
  } else {
    ParseError("Unknown string field: " + field);
  }
}

void Configuration::Assign(std::string const &field, int value) {
  std::map<std::string, int>::const_iterator match;

  match = _int_map.find(field);
  if (match != _int_map.end()) {
    _int_map[field] = value;
  } else {
    ParseError("Unknown integer field: " + field);
  }
}

void Configuration::Assign(std::string const &field, double value) {
  std::map<std::string, double>::const_iterator match;

  match = _float_map.find(field);
  if (match != _float_map.end()) {
    _float_map[field] = value;
  } else {
    ParseError("Unknown double field: " + field);
  }
}

const std::string &Configuration::GetStr(std::string const &field) const {
  std::map<std::string, std::string>::const_iterator match;

  match = _str_map.find(field);
  if (match != _str_map.end()) {
    return match->second;
  } else {
    ParseError("Unknown string field: " + field);
    exit(-1);
  }
}

int Configuration::GetInt(std::string const &field) const {
  std::map<std::string, int>::const_iterator match;

  match = _int_map.find(field);
  if (match != _int_map.end()) {
    return match->second;
  } else {
    ParseError("Unknown integer field: " + field);
    exit(-1);
  }
}

double Configuration::GetFloat(std::string const &field) const {
  std::map<std::string, double>::const_iterator match;

  match = _float_map.find(field);
  if (match != _float_map.end()) {
    return match->second;
  } else {
    ParseError("Unknown double field: " + field);
    exit(-1);
  }
}

std::vector<std::string> Configuration::GetStrArray(
    std::string const &field) const {
  std::string const param_str = GetStr(field);
  return tokenize_str(param_str);
}

std::vector<int> Configuration::GetIntArray(std::string const &field) const {
  std::string const param_str = GetStr(field);
  return tokenize_int(param_str);
}

std::vector<double> Configuration::GetFloatArray(
    std::string const &field) const {
  std::string const param_str = GetStr(field);
  return tokenize_float(param_str);
}

int Configuration::ParseFile(std::string const &filename) {
  FILE *config_file;
  if ((config_file = fopen(filename.c_str(), "r")) == 0) {
    std::cerr << "Could not open configuration file " << filename << std::endl;
    exit(-1);
  }

  yyscan_t scanner;
  yylex_init(&scanner);
  yyset_in(config_file, scanner);

  int res = yyparse(scanner, *this);
  yylex_destroy(scanner);

  fclose(config_file);

  return res;
}

int Configuration::ParseString(std::string const &str) {
  std::string config_string = str + ';';
  yyscan_t scanner;
  yylex_init(&scanner);

  YY_BUFFER_STATE buf = yy_scan_string(config_string.c_str(), scanner);

  int res = yyparse(scanner, *this);
  yy_delete_buffer(buf, scanner);
  yylex_destroy(scanner);

  return res;
}

void Configuration::ParseError(std::string const &msg,
                               unsigned int lineno) const {
  if (lineno) {
    std::cerr << "Parse error on line " << lineno << " : " << msg << std::endl;
  } else {
    std::cerr << "Parse error : " << msg << std::endl;
  }

  exit(-1);
}

bool ParseArgs(Configuration *cf, int argc, char **argv) {
  bool rc = false;

  // all dashed variables are ignored by the arg parser
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    size_t pos = arg.find('=');
    bool dash = (argv[i][0] == '-');
    if (pos == std::string::npos && !dash) {
      // parse config file
      cf->ParseFile(argv[i]);
      std::ifstream in(argv[i]);
      std::cout << "BEGIN Configuration File: " << argv[i] << std::endl;
      while (!in.eof()) {
        char c;
        in.get(c);
        std::cout << c;
      }
      std::cout << "END Configuration File: " << argv[i] << std::endl;
      rc = true;
    } else if (pos != std::string::npos) {
      // override individual parameter
      std::cout << "OVERRIDE Parameter: " << arg << std::endl;
      cf->ParseString(argv[i]);
    }
  }

  return rc;
}

// helpful for the GUI, write out nearly all variables contained in a config
// file. However, it can't and won't write out  empty strings since the booksim
// yacc parser won't be abled to parse blank strings
void Configuration::WriteFile(std::string const &filename) {
  std::ostream *config_out = new std::ofstream(filename.c_str());

  for (std::map<std::string, std::string>::const_iterator i = _str_map.begin();
       i != _str_map.end(); i++) {
    // the parser won't read empty strings
    if (i->second[0] != '\0') {
      *config_out << i->first << " = " << i->second << ";" << std::endl;
    }
  }

  for (std::map<std::string, int>::const_iterator i = _int_map.begin();
       i != _int_map.end(); i++) {
    *config_out << i->first << " = " << i->second << ";" << std::endl;
  }

  for (std::map<std::string, double>::const_iterator i = _float_map.begin();
       i != _float_map.end(); i++) {
    *config_out << i->first << " = " << i->second << ";" << std::endl;
  }
  config_out->flush();
  delete config_out;
}

void Configuration::WriteMatlabFile(std::ostream *config_out) const {
  for (std::map<std::string, std::string>::const_iterator i = _str_map.begin();
       i != _str_map.end(); i++) {
    // the parser won't read blanks lolz
    if (i->second[0] != '\0') {
      *config_out << "%" << i->first << " = \'" << i->second << "\';"
                  << std::endl;
    }
  }

  for (std::map<std::string, int>::const_iterator i = _int_map.begin();
       i != _int_map.end(); i++) {
    *config_out << "%" << i->first << " = " << i->second << ";" << std::endl;
  }

  for (std::map<std::string, double>::const_iterator i = _float_map.begin();
       i != _float_map.end(); i++) {
    *config_out << "%" << i->first << " = " << i->second << ";" << std::endl;
  }
  config_out->flush();
}

std::vector<std::string> tokenize_str(std::string const &data) {
  std::vector<std::string> values;

  // no elements, no braces --> empty list
  if (data.empty()) {
    return values;
  }

  // doesn't start with an opening brace --> treat as single element
  // note that this element can potentially contain nested lists
  if (data[0] != '{') {
    values.push_back(data);
    return values;
  }

  size_t start = 1;
  int nested = 0;

  size_t curr = start;

  while (std::string::npos != (curr = data.find_first_of("{,}", curr))) {
    if (data[curr] == '{') {
      ++nested;
    } else if ((data[curr] == '}') && nested) {
      --nested;
    } else if (!nested) {
      if (curr > start) {
        std::string token = data.substr(start, curr - start);
        values.push_back(token);
      }
      start = curr + 1;
    }
    ++curr;
  }
  assert(!nested);

  return values;
}

std::vector<int> tokenize_int(std::string const &data) {
  std::vector<int> values;

  // no elements, no braces --> empty list
  if (data.empty()) {
    return values;
  }

  // doesn't start with an opening brace --> treat as single element
  // note that this element can potentially contain nested lists
  if (data[0] != '{') {
    values.push_back(atoi(data.c_str()));
    return values;
  }

  size_t start = 1;
  int nested = 0;

  size_t curr = start;

  while (std::string::npos != (curr = data.find_first_of("{,}", curr))) {
    if (data[curr] == '{') {
      ++nested;
    } else if ((data[curr] == '}') && nested) {
      --nested;
    } else if (!nested) {
      if (curr > start) {
        std::string token = data.substr(start, curr - start);
        values.push_back(atoi(token.c_str()));
      }
      start = curr + 1;
    }
    ++curr;
  }
  assert(!nested);

  return values;
}

std::vector<double> tokenize_float(std::string const &data) {
  std::vector<double> values;

  // no elements, no braces --> empty list
  if (data.empty()) {
    return values;
  }

  // doesn't start with an opening brace --> treat as single element
  // note that this element can potentially contain nested lists
  if (data[0] != '{') {
    values.push_back(atof(data.c_str()));
    return values;
  }

  size_t start = 1;
  int nested = 0;

  size_t curr = start;

  while (std::string::npos != (curr = data.find_first_of("{,}", curr))) {
    if (data[curr] == '{') {
      ++nested;
    } else if ((data[curr] == '}') && nested) {
      --nested;
    } else if (!nested) {
      if (curr > start) {
        std::string token = data.substr(start, curr - start);
        values.push_back(atof(token.c_str()));
      }
      start = curr + 1;
    }
    ++curr;
  }
  assert(!nested);

  return values;
}
