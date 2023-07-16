#pragma once

#include <cstdio>
#include <list>
#include <map>

class memory_config;
class dram_req_t;
class memory_stats_t;
class dram_t;

enum memory_mode { READ_MODE = 0, WRITE_MODE };

class frfcfs_scheduler {
 public:
  frfcfs_scheduler(const memory_config *config, dram_t *dm,
                   memory_stats_t *stats);
  void add_req(dram_req_t *req);
  void data_collection(unsigned bank);
  dram_req_t *schedule(unsigned bank, unsigned curr_row);
  void print(FILE *fp);
  unsigned num_pending() const { return m_num_pending; }
  unsigned num_write_pending() const { return m_num_write_pending; }

 private:
  const memory_config *m_config;
  dram_t *m_dram;
  unsigned m_num_pending;
  unsigned m_num_write_pending;
  std::list<dram_req_t *> *m_queue;
  std::map<unsigned, std::list<std::list<dram_req_t *>::iterator>> *m_bins;
  std::list<std::list<dram_req_t *>::iterator> **m_last_row;
  unsigned *curr_row_service_time;  // one set of variables for each bank.
  unsigned *row_service_timestamp;  // tracks when scheduler began servicing
                                    // current row

  std::list<dram_req_t *> *m_write_queue;
  std::map<unsigned, std::list<std::list<dram_req_t *>::iterator>>
      *m_write_bins;
  std::list<std::list<dram_req_t *>::iterator> **m_last_write_row;

  enum memory_mode m_mode;
  memory_stats_t *m_stats;
};
