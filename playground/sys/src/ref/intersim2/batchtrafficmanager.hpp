// $Id: batchtrafficmanager.hpp 5188 2012-08-30 00:31:31Z dub $

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

#ifndef _BATCHTRAFFICMANAGER_HPP_
#define _BATCHTRAFFICMANAGER_HPP_

#include <iostream>

#include "config_utils.hpp"
#include "stats.hpp"
#include "trafficmanager.hpp"

class BatchTrafficManager : public TrafficManager {
 protected:
  int _max_outstanding;
  int _batch_size;
  int _batch_count;
  int _last_id;
  int _last_pid;

  Stats *_batch_time;
  double _overall_min_batch_time;
  double _overall_avg_batch_time;
  double _overall_max_batch_time;

  std::ostream *_sent_packets_out;

  virtual void _RetireFlit(Flit *f, int dest) override;

  virtual int _IssuePacket(int source, int cl) override;
  virtual void _ClearStats() override;
  virtual bool _SingleSim() override;

  virtual void _UpdateOverallStats() override;

  virtual std::string _OverallStatsCSV(int c = 0) const override;

 public:
  BatchTrafficManager(const Configuration &config,
                      const std::vector<Network *> &net,
                      InterconnectInterface *icnt);
  virtual ~BatchTrafficManager();

  virtual void WriteStats(std::ostream &os = std::cout) const override;
  // virtual void DisplayStats(std::ostream &os = std::cout) const override;
  virtual void DisplayStats(FILE *fp = stdout) const override;
  // virtual void DisplayOverallStats(std::ostream &os = std::cout) const
  // override;
  virtual void DisplayOverallStats(FILE *fp = stdout) const override;
};

#endif
