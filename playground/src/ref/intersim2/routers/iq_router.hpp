// $Id: iq_router.hpp 5263 2012-09-20 23:40:33Z dub $

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

#ifndef _IQ_ROUTER_HPP_
#define _IQ_ROUTER_HPP_

#include <deque>
#include <map>
#include <queue>
#include <set>
#include <string>

#include "../routefunc.hpp"
#include "router.hpp"

class VC;
class Flit;
class Credit;
class Buffer;
class BufferState;
class Allocator;
class SwitchMonitor;
class BufferMonitor;

class IQRouter : public Router {
  int _vcs;

  bool _vc_busy_when_full;
  bool _vc_prioritize_empty;
  bool _vc_shuffle_requests;

  bool _speculative;
  bool _spec_check_elig;
  bool _spec_check_cred;
  bool _spec_mask_by_reqs;

  bool _active;

  int _routing_delay;
  int _vc_alloc_delay;
  int _sw_alloc_delay;

  std::map<int, Flit *> _in_queue_flits;

  std::deque<std::pair<int, std::pair<Credit *, int>>> _proc_credits;

  std::deque<std::pair<int, std::pair<int, int>>> _route_vcs;
  std::deque<std::pair<int, std::pair<std::pair<int, int>, int>>> _vc_alloc_vcs;
  std::deque<std::pair<int, std::pair<std::pair<int, int>, int>>> _sw_hold_vcs;
  std::deque<std::pair<int, std::pair<std::pair<int, int>, int>>> _sw_alloc_vcs;

  std::deque<std::pair<int, std::pair<Flit *, std::pair<int, int>>>>
      _crossbar_flits;

  std::map<int, Credit *> _out_queue_credits;

  std::vector<Buffer *> _buf;
  std::vector<BufferState *> _next_buf;

  Allocator *_vc_allocator;
  Allocator *_sw_allocator;
  Allocator *_spec_sw_allocator;

  std::vector<int> _vc_rr_offset;
  std::vector<int> _sw_rr_offset;

  tRoutingFunction _rf;

  int _output_buffer_size;
  std::vector<std::queue<Flit *>> _output_buffer;

  std::vector<std::queue<Credit *>> _credit_buffer;

  bool _hold_switch_for_packet;
  std::vector<int> _switch_hold_in;
  std::vector<int> _switch_hold_out;
  std::vector<int> _switch_hold_vc;

  bool _noq;
  std::vector<std::vector<int>> _noq_next_output_port;
  std::vector<std::vector<int>> _noq_next_vc_start;
  std::vector<std::vector<int>> _noq_next_vc_end;

#ifdef TRACK_FLOWS
  std::vector<std::vector<std::queue<int>>> _outstanding_classes;
#endif

  bool _ReceiveFlits();
  bool _ReceiveCredits();

  virtual void _InternalStep();

  bool _SWAllocAddReq(int input, int vc, int output);

  void _InputQueuing();

  void _RouteEvaluate();
  void _VCAllocEvaluate();
  void _SWHoldEvaluate();
  void _SWAllocEvaluate();
  void _SwitchEvaluate();

  void _RouteUpdate();
  void _VCAllocUpdate();
  void _SWHoldUpdate();
  void _SWAllocUpdate();
  void _SwitchUpdate();

  void _OutputQueuing();

  void _SendFlits();
  void _SendCredits();

  void _UpdateNOQ(int input, int vc, Flit const *f);

  // ----------------------------------------
  //
  //   Router Power Modellingyes
  //
  // ----------------------------------------

  SwitchMonitor *_switchMonitor;
  BufferMonitor *_bufferMonitor;

 public:
  IQRouter(Configuration const &config, Module *parent, std::string const &name,
           int id, int inputs, int outputs);

  virtual ~IQRouter();

  virtual void AddOutputChannel(FlitChannel *channel,
                                CreditChannel *backchannel);

  virtual void ReadInputs();
  virtual void WriteOutputs();

  void Display(std::ostream &os = std::cout) const;

  virtual int GetUsedCredit(int o) const;
  virtual int GetBufferOccupancy(int i) const;

#ifdef TRACK_BUFFERS
  virtual int GetUsedCreditForClass(int output, int cl) const;
  virtual int GetBufferOccupancyForClass(int input, int cl) const;
#endif

  virtual std::vector<int> UsedCredits() const;
  virtual std::vector<int> FreeCredits() const;
  virtual std::vector<int> MaxCredits() const;

  SwitchMonitor const *const GetSwitchMonitor() const { return _switchMonitor; }
  BufferMonitor const *const GetBufferMonitor() const { return _bufferMonitor; }
};

#endif
