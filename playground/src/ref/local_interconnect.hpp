#pragma once

#include <iostream>
#include <map>
#include <queue>
#include <vector>

enum Interconnect_type { REQ_NET = 0, REPLY_NET = 1 };

enum Arbiteration_type { NAIVE_RR = 0, iSLIP = 1 };

struct inct_config {
  // config for local interconnect
  unsigned in_buffer_limit;
  unsigned out_buffer_limit;
  unsigned subnets;
  Arbiteration_type arbiter_algo;
  unsigned verbose;
  unsigned grant_cycles;
};

class xbar_router {
public:
  xbar_router(unsigned router_id, enum Interconnect_type m_type,
              unsigned n_shader, unsigned n_mem,
              const struct inct_config &m_localinct_config);
  ~xbar_router();
  void Push(unsigned input_deviceID, unsigned output_deviceID, void *data,
            unsigned int size);
  void *Pop(unsigned ouput_deviceID);
  void Advance();

  bool Busy() const;
  bool Has_Buffer_In(unsigned input_deviceID, unsigned size,
                     bool update_counter = false);
  bool Has_Buffer_Out(unsigned output_deviceID, unsigned size);

  // some stats
  unsigned long long cycles;
  unsigned long long conflicts;
  unsigned long long conflicts_util;
  unsigned long long cycles_util;
  unsigned long long reqs_util;
  unsigned long long out_buffer_full;
  unsigned long long out_buffer_util;
  unsigned long long in_buffer_full;
  unsigned long long in_buffer_util;
  unsigned long long packets_num;

private:
  void iSLIP_Advance();
  void RR_Advance();

  struct Packet {
    Packet(void *m_data, unsigned m_output_deviceID) {
      data = m_data;
      output_deviceID = m_output_deviceID;
    }
    void *data;
    unsigned output_deviceID;
  };
  std::vector<std::queue<Packet>> in_buffers;
  std::vector<std::queue<Packet>> out_buffers;
  unsigned _n_shader, _n_mem, total_nodes;
  unsigned in_buffer_limit, out_buffer_limit;
  std::vector<unsigned> next_node; // used for iSLIP arbit
  unsigned next_node_id;           // used for RR arbit
  unsigned m_id;
  enum Interconnect_type router_type;
  unsigned active_in_buffers, active_out_buffers;
  Arbiteration_type arbit_type;
  unsigned verbose;

  unsigned grant_cycles;
  unsigned grant_cycles_count;

  friend class LocalInterconnect;
};

class LocalInterconnect {
public:
  LocalInterconnect(const struct inct_config &m_localinct_config);
  ~LocalInterconnect();
  static LocalInterconnect *New(const struct inct_config &m_inct_config);
  void CreateInterconnect(unsigned n_shader, unsigned n_mem);

  // node side functions
  void Init();
  void Push(unsigned input_deviceID, unsigned output_deviceID, void *data,
            unsigned int size);
  void *Pop(unsigned ouput_deviceID);
  void Advance();
  bool Busy() const;
  bool HasBuffer(unsigned deviceID, unsigned int size) const;
  void DisplayStats() const;
  void DisplayOverallStats() const;
  unsigned GetFlitSize() const;

  void DisplayState(FILE *fp) const;

protected:
  const inct_config &m_inct_config;

  unsigned n_shader, n_mem;
  unsigned n_subnets;
  std::vector<xbar_router *> net;
};

extern LocalInterconnect *g_localicnt_interface;

static void LocalInterconnect_create(unsigned int n_shader,
                                     unsigned int n_mem) {
  g_localicnt_interface->CreateInterconnect(n_shader, n_mem);
}

static void LocalInterconnect_init() { g_localicnt_interface->Init(); }

static bool LocalInterconnect_has_buffer(unsigned input, unsigned int size) {
  return g_localicnt_interface->HasBuffer(input, size);
}

static void LocalInterconnect_push(unsigned input, unsigned output, void *data,
                                   unsigned int size) {
  g_localicnt_interface->Push(input, output, data, size);
}

static void *LocalInterconnect_pop(unsigned output) {
  return g_localicnt_interface->Pop(output);
}

static void LocalInterconnect_transfer() { g_localicnt_interface->Advance(); }

static bool LocalInterconnect_busy() { return g_localicnt_interface->Busy(); }

static void LocalInterconnect_display_stats() {
  g_localicnt_interface->DisplayStats();
}

static void LocalInterconnect_display_overall_stats() {
  g_localicnt_interface->DisplayOverallStats();
}

static void LocalInterconnect_display_state(FILE *fp) {
  g_localicnt_interface->DisplayState(fp);
}

static unsigned LocalInterconnect_get_flit_size() {
  return g_localicnt_interface->GetFlitSize();
}
