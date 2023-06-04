#include "option_parser.hpp"

#include <assert.h>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// A generic option registry regardless of data type
class OptionRegistryInterface {
public:
  OptionRegistryInterface(const std::string optionName,
                          const std::string optionDesc)
      : m_optionName(optionName), m_optionDesc(optionDesc), m_isParsed(false) {}

  virtual ~OptionRegistryInterface() {}

  const std::string &GetName() { return m_optionName; }
  const std::string &GetDesc() { return m_optionDesc; }
  const bool isParsed() { return m_isParsed; }
  virtual std::string toString() = 0;
  virtual bool fromString(const std::string str) = 0;
  virtual bool isFlag() = 0;
  virtual bool assignDefault(const char *str) = 0;

protected:
  std::string m_optionName;
  std::string m_optionDesc;
  bool m_isParsed; // true if the target variable has been updated by
                   // fromString()
};

// Template for option registry - class T = specify data type of the option
template <class T> class OptionRegistry : public OptionRegistryInterface {
public:
  OptionRegistry(const std::string name, const std::string desc, T &variable)
      : OptionRegistryInterface(name, desc), m_variable(variable) {}

  virtual ~OptionRegistry() {}

  virtual std::string toString() {
    std::stringstream ss;
    ss << m_variable;
    return ss.str();
  }

  virtual bool fromString(const std::string str) {
    std::stringstream ss(str);
    ss.exceptions(std::stringstream::failbit | std::stringstream::badbit);
    ss << std::setbase(10);
    if (str.size() > 1 && str[0] == '0') {
      if (str.size() > 2 && str[1] == 'x') {
        ss.ignore(2);
        ss << std::setbase(16);
      } else {
        ss.ignore(1);
        ss << std::setbase(8);
      }
    }
    try {
      ss >> m_variable;
    } catch (std::exception &e) {
      return false;
    }
    m_isParsed = true;
    return true;
  }

  virtual bool isFlag() { return false; }
  virtual bool assignDefault(const char *str) { return fromString(str); }

  operator T() { return m_variable; }

private:
  T &m_variable;
};

// specialized parser for string-type options
template <>
bool OptionRegistry<std::string>::fromString(const std::string str) {
  m_variable = str;
  m_isParsed = true;
  return true;
}

// specialized parser for c-string type options
template <> bool OptionRegistry<char *>::fromString(const std::string str) {
  m_variable = new char[str.size() + 1];
  strcpy(m_variable, str.c_str());
  m_isParsed = true;
  return true;
}

// specialized default assignment for c-string type option to allow NULL default
template <> bool OptionRegistry<char *>::assignDefault(const char *str) {
  m_variable = const_cast<char *>(
      str); // c-string options are not meant to be edited anyway
  m_isParsed = true;
  return true;
}

// specialized default assignment for c-string type option to allow NULL default
template <> std::string OptionRegistry<char *>::toString() {
  std::stringstream ss;
  if (m_variable != NULL) {
    ss << m_variable;
  } else {
    ss << "NULL";
  }
  return ss.str();
}

// specialized parser for boolean options
template <> bool OptionRegistry<bool>::fromString(const std::string str) {
  int value = 1;
  bool parsed = true;
  std::stringstream ss(str);
  ss.exceptions(std::stringstream::failbit | std::stringstream::badbit);
  try {
    ss >> value;
  } catch (std::stringstream::failure &ep) {
    parsed = false;
  }
  assert(value == 0 or
         value ==
             1); // sanity check for boolean options (it can only be 1 or 0)
  m_variable = (value != 0);
  m_isParsed = true;
  return parsed;
}

// specializing a flag query function to identify boolean option
template <> bool OptionRegistry<bool>::isFlag() { return true; }

// class holding a collection of options and parse them from command
// line/configfile
class OptionParser {
public:
  OptionParser() {}
  ~OptionParser() {
    OptionCollection::iterator i_option;
    for (i_option = m_optionReg.begin(); i_option != m_optionReg.end();
         ++i_option) {
      delete (*i_option);
    }
  }

  template <class T>
  void Register(const std::string optionName, const std::string optionDesc,
                T &optionVariable, const char *optionDefault) {
    OptionRegistry<T> *p_option =
        new OptionRegistry<T>(optionName, optionDesc, optionVariable);
    m_optionReg.push_back(p_option);
    m_optionMap[optionName] = p_option;
    p_option->assignDefault(optionDefault);
  }

  void ParseCommandLine(int argc, const char *const argv[]) {
    for (int i = 1; i < argc; i++) {
      OptionMap::iterator i_option;
      bool optionFound = false;

      i_option = m_optionMap.find(argv[i]);
      if (i_option != m_optionMap.end()) {
        const char *argstr = (i + 1 < argc) ? argv[i + 1] : "";
        OptionRegistryInterface *p_option = i_option->second;
        if (p_option->isFlag()) {
          if (p_option->fromString(argstr) == true) {
            i += 1;
          }
        } else {
          if (p_option->fromString(argstr) == false) {
            fprintf(stderr,
                    "\n\nGPGPU-Sim ** ERROR: Cannot parse value '%s' for "
                    "option '%s'.\n",
                    argstr, argv[i]);
            exit(1);
          }
          i += 1;
        }
        optionFound = true;
      } else if (std::string(argv[i]) == "-config") {
        if (i + 1 >= argc) {
          fprintf(stderr, "\n\nGPGPU-Sim ** ERROR: Missing filename for option "
                          "'-config'.\n");
          exit(1);
        }

        ParseFile(argv[i + 1]);
        i += 1;
        optionFound = true;
      }
      if (optionFound == false) {
        fprintf(stderr, "\n\nGPGPU-Sim ** ERROR: Unknown Option: '%s' \n",
                argv[i]);
        exit(1);
      }
    }
  }

  void ParseFile(const char *filename) {
    std::ifstream inputFile;
    std::stringstream args;

    // open config file, stream every line into a continuous buffer
    // get rid of comments in the process
    inputFile.open(filename);
    if (!inputFile.good()) {
      fprintf(stderr, "\n\nGPGPU-Sim ** ERROR: Cannot open config file '%s'\n",
              filename);
      exit(1);
    }
    while (inputFile.good()) {
      std::string line;
      getline(inputFile, line);
      size_t commentStart = line.find_first_of("#");
      if (commentStart != line.npos) {
        line.erase(commentStart);
      }
      args << line << ' ';
    }
    inputFile.close();

    ParseStringStream(args);
  }

  // parse the given string as tokens separated by a set of given delimiters
  void ParseString(std::string inputString,
                   const std::string delimiters = std::string(" ;")) {
    // convert all delimiter characters into whitespaces
    for (unsigned t = 0; t < inputString.size(); t++) {
      for (unsigned d = 0; d < delimiters.size(); d++) {
        if (inputString[t] == delimiters.at(d)) {
          inputString[t] = ' ';
          break;
        }
      }
    }
    std::stringstream args(inputString);
    ParseStringStream(args);
  }

  // parse the given stringstream as whitespace-separated tokens. drain the
  // stream in the process
  void ParseStringStream(std::stringstream &args) {
    // extract non-whitespace string tokens
    std::vector<char *> argv;
    argv.push_back(new char[6]);
    strcpy(argv[0], "dummy");
    while (args.good()) {
      std::string argNew;
      args >> argNew;

      if (argNew.size() == 0)
        continue; // this is probably the last token

      if (argNew[0] == '"') {
        while (args.good() && argNew[argNew.size() - 1] != '"') {
          std::string argCont;
          args >> argCont;
          argNew += " " + argCont;
        }
        argNew.erase(0, 1);
        argNew.erase(argNew.size() - 1);
      }

      char *c_argNew = new char[argNew.size() + 1];
      strcpy(c_argNew, argNew.c_str());
      argv.push_back(c_argNew);
    }

    // pass the string token into normal commandline parser
    char **targv = (char **)calloc(argv.size(), sizeof(char *));
    for (unsigned k = 0; k < argv.size(); k++)
      targv[k] = argv[k];
    ParseCommandLine(argv.size(), targv);
    free(targv);
    for (size_t i = 0; i < argv.size(); i++) {
      delete[] argv[i];
    }
  }

  void Print(FILE *fout) {
    OptionCollection::iterator i_option;
    for (i_option = m_optionReg.begin(); i_option != m_optionReg.end();
         ++i_option) {
      std::stringstream sout;
      if ((*i_option)->isParsed() == false) {
        std::cerr << "\n\nGPGPU-Sim ** ERROR: Missing option '"
                  << (*i_option)->GetName() << "'\n";
        assert(0);
      }
      sout << std::setw(20) << std::left << (*i_option)->GetName() << " ";
      sout << std::setw(20) << std::right << (*i_option)->toString() << " # ";
      sout << std::left << (*i_option)->GetDesc();
      sout << std::endl;
      fprintf(fout, "%s", sout.str().c_str());
    }
  }

private:
  typedef std::list<OptionRegistryInterface *> OptionCollection;
  OptionCollection m_optionReg;
  typedef std::map<std::string, OptionRegistryInterface *> OptionMap;
  OptionMap m_optionMap;
};

option_parser_t option_parser_create() {
  OptionParser *p_opr = new OptionParser();
  return reinterpret_cast<option_parser_t>(p_opr);
}

void option_parser_destroy(option_parser_t opp) {
  OptionParser *p_opr = reinterpret_cast<OptionParser *>(opp);
  delete p_opr;
}

void option_parser_register(option_parser_t opp, const char *name,
                            enum option_dtype type, void *variable,
                            const char *desc, const char *defaultvalue) {
  OptionParser *p_opr = reinterpret_cast<OptionParser *>(opp);
  switch (type) {
  case OPT_INT32:
    p_opr->Register<int>(name, desc, *(int *)variable, defaultvalue);
    break;
  case OPT_UINT32:
    p_opr->Register<unsigned int>(name, desc, *(unsigned int *)variable,
                                  defaultvalue);
    break;
  case OPT_INT64:
    p_opr->Register<long long>(name, desc, *(long long *)variable,
                               defaultvalue);
    break;
  case OPT_UINT64:
    p_opr->Register<unsigned long long>(
        name, desc, *(unsigned long long *)variable, defaultvalue);
    break;
  case OPT_BOOL:
    p_opr->Register<bool>(name, desc, *(bool *)variable, defaultvalue);
    break;
  case OPT_FLOAT:
    p_opr->Register<float>(name, desc, *(float *)variable, defaultvalue);
    break;
  case OPT_DOUBLE:
    p_opr->Register<double>(name, desc, *(double *)variable, defaultvalue);
    break;
  case OPT_CHAR:
    p_opr->Register<char>(name, desc, *(char *)variable, defaultvalue);
    break;
  case OPT_CSTR:
    p_opr->Register<char *>(name, desc, *(char **)variable, defaultvalue);
    break;
  default:
    fprintf(stderr,
            "\n\nGPGPU-Sim ** ERROR: option data type (%d) not supported!\n",
            type);
    exit(1);
    break;
  }
}

void option_parser_cmdline(option_parser_t opp,
                           const std::vector<const char *> &argv) {
  int argc = argv.size();
  const char **c_argv = const_cast<const char **>(&argv[0]);
  option_parser_cmdline(opp, argc, c_argv);
}

void option_parser_cmdline(option_parser_t opp, int argc, const char *argv[]) {
  OptionParser *p_opr = reinterpret_cast<OptionParser *>(opp);
  return p_opr->ParseCommandLine(argc, argv);
}

void option_parser_cfgfile(option_parser_t opp, const char *filename) {
  OptionParser *p_opr = reinterpret_cast<OptionParser *>(opp);
  p_opr->ParseFile(filename);
}

void option_parser_delimited_string(option_parser_t opp,
                                    const char *inputstring,
                                    const char *delimiters) {
  OptionParser *p_opr = reinterpret_cast<OptionParser *>(opp);
  p_opr->ParseString(inputstring, delimiters);
}

void option_parser_print(option_parser_t opp, FILE *fout) {
  OptionParser *p_opr = reinterpret_cast<OptionParser *>(opp);
  p_opr->Print(fout);
}
