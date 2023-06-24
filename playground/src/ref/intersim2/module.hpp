// $Id: module.hpp 5188 2012-08-30 00:31:31Z dub $

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

#ifndef _MODULE_HPP_
#define _MODULE_HPP_

#include "booksim.hpp"
// #include "interconnect_interface.hpp"

#include <iostream>
#include <string>
#include <vector>

class InterconnectInterface;

class Module {
public:
  // todo maybe the order here
  Module(Module *parent, const std::string &name, InterconnectInterface *icnt)
      : Module(parent, name) {
    m_icnt = icnt;
  };
  Module(Module *parent, const std::string &name);
  virtual ~Module() {}

  inline const std::string &Name() const { return _name; }
  inline const std::string &FullName() const { return _fullname; }

  void DisplayHierarchy(int level = 0, std::ostream &os = std::cout) const;
  int GetSimTime() const;

  void Error(const std::string &msg) const;
  void Debug(const std::string &msg) const;

  virtual void Display(std::ostream &os = std::cout) const;

  InterconnectInterface *m_icnt;

private:
  std::string _name;
  std::string _fullname;

  std::vector<Module *> _children;

protected:
  void _AddChild(Module *child);
};

#endif
