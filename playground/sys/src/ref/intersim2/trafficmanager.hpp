// $Id: trafficmanager.hpp 5365 2012-11-25 02:09:59Z qtedq $

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

#ifndef _TRAFFICMANAGER_HPP_
#define _TRAFFICMANAGER_HPP_

#include <cassert>
#include <list>
#include <map>
#include <set>

#include "./networks/network.hpp"
#include "buffer_state.hpp"
#include "config_utils.hpp"
#include "flit.hpp"
#include "injection.hpp"
#include "module.hpp"
#include "outputset.hpp"
#include "routefunc.hpp"
#include "stats.hpp"
#include "traffic.hpp"

// register the requests to a node
class PacketReplyInfo;

class TrafficManager : public Module {
 private:
  std::vector<std::vector<int>> _packet_size;
  std::vector<std::vector<int>> _packet_size_rate;
  std::vector<int> _packet_size_max_val;

 protected:
  int _nodes;
  int _routers;
  int _vcs;

  std::vector<Network *> _net;
  std::vector<std::vector<Router *>> _router;

  // ============ Traffic ============

  int _classes;

  std::vector<double> _load;

  std::vector<int> _use_read_write;
  std::vector<double> _write_fraction;

  std::vector<int> _read_request_size;
  std::vector<int> _read_reply_size;
  std::vector<int> _write_request_size;
  std::vector<int> _write_reply_size;

  std::vector<std::string> _traffic;

  std::vector<int> _class_priority;

  std::vector<std::vector<int>> _last_class;

  std::vector<TrafficPattern *> _traffic_pattern;
  std::vector<InjectionProcess *> _injection_process;

  // ============ Message priorities ============

  enum ePriority {
    class_based,
    age_based,
    network_age_based,
    local_age_based,
    queue_length_based,
    hop_count_based,
    sequence_based,
    none
  };

  ePriority _pri_type;

  // ============ Injection VC states  ============

  std::vector<std::vector<BufferState *>> _buf_states;
#ifdef TRACK_FLOWS
  std::vector<std::vector<std::vector<int>>> _outstanding_credits;
  sdt::vector<std::vector<std::vector<std::queue<int>>>> _outstanding_classes;
#endif
  std::vector<std::vector<std::vector<int>>> _last_vc;

  // ============ Routing ============

  tRoutingFunction _rf;
  bool _lookahead_routing;
  bool _noq;

  // ============ Injection queues ============

  std::vector<std::vector<int>> _qtime;
  std::vector<std::vector<bool>> _qdrained;
  std::vector<std::vector<std::list<Flit *>>> _partial_packets;

  std::vector<std::map<unsigned long long, Flit *>> _total_in_flight_flits;
  std::vector<std::map<unsigned long long, Flit *>> _measured_in_flight_flits;
  std::vector<std::map<unsigned long long, Flit *>> _retired_packets;
  bool _empty_network;

  bool _hold_switch_for_packet;

  // ============ physical sub-networks ==========

  int _subnets;

  std::vector<int> _subnet;

  // ============ deadlock ==========

  int _deadlock_timer;
  int _deadlock_warn_timeout;

  // ============ request & replies ==========================

  std::vector<int> _packet_seq_no;
  std::vector<std::list<PacketReplyInfo *>> _repliesPending;
  std::vector<int> _requestsOutstanding;

  // ============ Statistics ============

  std::vector<Stats *> _plat_stats;
  std::vector<double> _overall_min_plat;
  std::vector<double> _overall_avg_plat;
  std::vector<double> _overall_max_plat;

  std::vector<Stats *> _nlat_stats;
  std::vector<double> _overall_min_nlat;
  std::vector<double> _overall_avg_nlat;
  std::vector<double> _overall_max_nlat;

  std::vector<Stats *> _flat_stats;
  std::vector<double> _overall_min_flat;
  std::vector<double> _overall_avg_flat;
  std::vector<double> _overall_max_flat;

  std::vector<Stats *> _frag_stats;
  std::vector<double> _overall_min_frag;
  std::vector<double> _overall_avg_frag;
  std::vector<double> _overall_max_frag;

  std::vector<std::vector<Stats *>> _pair_plat;
  std::vector<std::vector<Stats *>> _pair_nlat;
  std::vector<std::vector<Stats *>> _pair_flat;

  std::vector<Stats *> _hop_stats;
  std::vector<double> _overall_hop_stats;

  std::vector<std::vector<int>> _sent_packets;
  std::vector<double> _overall_min_sent_packets;
  std::vector<double> _overall_avg_sent_packets;
  std::vector<double> _overall_max_sent_packets;
  std::vector<std::vector<int>> _accepted_packets;
  std::vector<double> _overall_min_accepted_packets;
  std::vector<double> _overall_avg_accepted_packets;
  std::vector<double> _overall_max_accepted_packets;
  std::vector<std::vector<int>> _sent_flits;
  std::vector<double> _overall_min_sent;
  std::vector<double> _overall_avg_sent;
  std::vector<double> _overall_max_sent;
  std::vector<std::vector<int>> _accepted_flits;
  std::vector<double> _overall_min_accepted;
  std::vector<double> _overall_avg_accepted;
  std::vector<double> _overall_max_accepted;

#ifdef TRACK_STALLS
  std::vector<std::vector<int>> _buffer_busy_stalls;
  std::vector<std::vector<int>> _buffer_conflict_stalls;
  std::vector<std::vector<int>> _buffer_full_stalls;
  std::vector<std::vector<int>> _buffer_reserved_stalls;
  std::vector<std::vector<int>> _crossbar_conflict_stalls;
  std::vector<double> _overall_buffer_busy_stalls;
  std::vector<double> _overall_buffer_conflict_stalls;
  std::vector<double> _overall_buffer_full_stalls;
  std::vector<double> _overall_buffer_reserved_stalls;
  std::vector<double> _overall_crossbar_conflict_stalls;
#endif

  std::vector<int> _slowest_packet;
  std::vector<int> _slowest_flit;

  std::map<std::string, Stats *> _stats;

  // ============ Simulation parameters ============

  enum eSimState { warming_up, running, draining, done };
  eSimState _sim_state;

  bool _measure_latency;

  int _reset_time;
  int _drain_time;

  int _total_sims;
  int _sample_period;
  int _max_samples;
  int _warmup_periods;

  int _include_queuing;

  std::vector<int> _measure_stats;
  bool _pair_stats;

  std::vector<double> _latency_thres;

  std::vector<double> _stopping_threshold;
  std::vector<double> _acc_stopping_threshold;

  std::vector<double> _warmup_threshold;
  std::vector<double> _acc_warmup_threshold;

  unsigned long long _cur_id;
  unsigned long long _cur_pid;
  int _time;

  std::set<unsigned long long> _flits_to_watch;
  std::set<unsigned long long> _packets_to_watch;

  bool _print_csv_results;

  // flits to watch
  std::ostream *_stats_out;

#ifdef TRACK_FLOWS
  std::vector<std::vector<int>> _injected_flits;
  std::vector<std::vector<int>> _ejected_flits;
  std::ostream *_injected_flits_out;
  std::ostream *_received_flits_out;
  std::ostream *_stored_flits_out;
  std::ostream *_sent_flits_out;
  std::ostream *_outstanding_credits_out;
  std::ostream *_ejected_flits_out;
  std::ostream *_active_packets_out;
#endif

#ifdef TRACK_CREDITS
  std::ostream *_used_credits_out;
  std::ostream *_free_credits_out;
  std::ostream *_max_credits_out;
#endif

  // ============ Internal methods ============
 protected:
  virtual void _RetireFlit(Flit *f, int dest);

  void _Inject();
  virtual void _Step();

  bool _PacketsOutstanding() const;

  virtual int _IssuePacket(int source, int cl);
  virtual void _GeneratePacket(int source, int size, int cl, int time);

  virtual void _ClearStats();

  void _ComputeStats(const std::vector<int> &stats, int *sum, int *min = NULL,
                     int *max = NULL, int *min_pos = NULL,
                     int *max_pos = NULL) const;

  virtual bool _SingleSim();

  void _DisplayRemaining(std::ostream &os = std::cout) const;

  void _LoadWatchList(const std::string &filename);

  virtual void _UpdateOverallStats();

  virtual std::string _OverallStatsCSV(int c = 0) const;

  int _GetNextPacketSize(int cl) const;
  double _GetAveragePacketSize(int cl) const;

 public:
  static TrafficManager *New(Configuration const &config,
                             std::vector<Network *> const &net,
                             InterconnectInterface *icnt);

  TrafficManager(const Configuration &config, const std::vector<Network *> &net,
                 InterconnectInterface *icnt);
  virtual ~TrafficManager();

  bool Run();

  virtual void WriteStats(std::ostream &os = std::cout) const;
  virtual void UpdateStats();
  virtual void DisplayStats(std::ostream &os = std::cout) const;
  virtual void DisplayOverallStats(std::ostream &os = std::cout) const;
  virtual void DisplayOverallStatsCSV(std::ostream &os = std::cout) const;

  inline int getTime() { return _time; }
  Stats *getStats(const std::string &name) { return _stats[name]; }
};

template <class T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
  for (size_t i = 0; i < v.size() - 1; ++i) {
    os << v[i] << ",";
  }
  os << v[v.size() - 1];
  return os;
}

#endif
