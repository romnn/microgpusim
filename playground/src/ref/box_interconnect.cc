#include "box_interconnect.hpp"

#include "intersim2/intersim_config.hpp"
#include "intersim2/networks/network.hpp"
#include "mem_fetch.hpp"

bool BoxInterconnect::HasBuffer(unsigned deviceID, unsigned int size) const {
  int icntID = _node_map.find(deviceID)->second;
  assert(icntID == deviceID);

  // request is subnet 0 and reply is subnet 1
  bool is_memory_node = ((_subnets > 1) && deviceID >= _n_shader);
  unsigned subnet = is_memory_node ? 1 : 0;
  bool has_buffer =
      simple_input_queue[subnet][icntID][0].size() <= _input_buffer_capacity;

  // printf("InterconnectInterface::HasBuffer(dev=%u, size=%u): "
  //        "_input_buffer_capacity = %u\n",
  //        deviceID, size, _input_buffer_capacity);

  return has_buffer;
}

void BoxInterconnect::Advance() {
  // printf("ROMAN INTERCONN ADVANCE\n");
  // do nothing
}

bool BoxInterconnect::Busy() const { return false; }

void *BoxInterconnect::Pop(unsigned deviceID) {
  int icntID = _node_map[deviceID];
  assert(icntID == deviceID);

  // bool is_memory_node = ((_subnets > 1) && deviceID >= _n_shader);
  // unsigned subnet = is_memory_node ? 1 : 0;

  // request is subnet 0 and reply is subnet 1
  // int subnet = (deviceID < _n_shader) ? 1 : 0;
  int subnet = (deviceID >= _n_shader) ? 1 : 0;

  void *data = NULL;

  int turn = _round_robin_turn[subnet][icntID];

  printf("INTERCONN POP FROM %d (device=%u, id=%u, subnet=%d, turn=%d)\n",
         deviceID, icntID, deviceID, subnet, turn);

  for (int vc = 0; (vc < _vcs) && (data == NULL); vc++) {
    // printf("ROMAN INTERCONN POP from (%d, %d, %d) vc=%d turn=%d size=%lu\n",
    //        subnet, icntID, turn, vc, turn,
    //        simple_output_queue[subnet][icntID][turn].size());

    if (!simple_output_queue[subnet][icntID][turn].empty()) {
      data = simple_output_queue[subnet][icntID][turn].front();
      assert(data != NULL);
      simple_output_queue[subnet][icntID][turn].pop_front();
    }
    // if (_boundary_buffer[subnet][icntID][turn].HasPacket()) {
    //   data = _boundary_buffer[subnet][icntID][turn].PopPacket();
    // }
    turn++;
    if (turn == _vcs)
      turn = 0;
  }
  if (data != NULL) {
    _round_robin_turn[subnet][icntID] = turn;
  }

  return data;
}

void BoxInterconnect::Push(unsigned input_deviceID, unsigned output_deviceID,
                           void *data, unsigned int size) {
  // sanity check
  // no, traffic manager uses nk ary from config file
  // assert(_n_shader + _n_mem == _traffic_manager->_nodes);
  assert(output_deviceID < _n_shader + _n_mem);

  // it should have free buffer
  assert(HasBuffer(input_deviceID, size));

  // request is subnet 0 and reply is subnet 1
  // bool is_memory_node = ((_subnets > 1) && (input_deviceID >= _n_shader));
  bool is_memory_node = ((_subnets > 1) && (output_deviceID >= _n_shader));
  unsigned subnet = is_memory_node ? 1 : 0;

  int input_icntID = _node_map[input_deviceID];
  int output_icntID = _node_map[output_deviceID];
  for (auto const &x : _node_map) {
    // std::cout << x.first << ':' << x.second << std::endl;
    assert(x.first == x.second);
  }
  // printf("output icnt id %u output device id %u\n", output_icntID,
  //        output_deviceID);

  assert(input_icntID == input_deviceID);
  assert(output_icntID == output_deviceID);

  // mem_fetch *mf = static_cast<mem_fetch *>(data);
  printf("INTERCONN PUSH ");
  ((mem_fetch *)data)->print(stdout);
  printf(": %d bytes from device %d to %d (subnet %d)\n", size, input_icntID,
         output_icntID, subnet);

  // simple_input_queue[subnet][input_icntID][0].push_back(data);
  simple_output_queue[subnet][output_icntID][0].push_back(data);
  // printf("output queue size of (%d, %d, 0) is now %lu\n", subnet,
  // output_icntID,
  //        simple_output_queue[subnet][output_icntID][0].size());
}

void BoxInterconnect::Init() {
  // _traffic_manager->Init();

  unsigned nodes = _net[0]->NumNodes();
  unsigned classes = _icnt_config->GetInt("classes");

  simple_input_queue.resize(_subnets);
  simple_output_queue.resize(_subnets);
  for (int subnet = 0; subnet < _subnets; ++subnet) {
    simple_input_queue[subnet].resize(nodes);
    simple_output_queue[subnet].resize(nodes);
    for (int node = 0; node < nodes; ++node) {
      simple_input_queue[subnet][node].resize(classes);
      simple_output_queue[subnet][node].resize(classes);
    }
  }
}

std::unique_ptr<BoxInterconnect>
new_box_interconnect(const char *config_filename) {
  std::unique_ptr<BoxInterconnect> box_interconnect =
      std::make_unique<BoxInterconnect>();
  box_interconnect->ParseConfigFile(config_filename);
  return box_interconnect;
}
