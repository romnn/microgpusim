#pragma once

#include "../trace_simt_core_cluster.hpp"

class cluster_bridge {
 public:
  cluster_bridge(const trace_simt_core_cluster *ptr) : ptr(ptr) {}

  std::unique_ptr<std::vector<unsigned>> get_core_sim_order() const {
    std::vector<unsigned> out;
    std::list<unsigned>::const_iterator iter;
    for (iter = (ptr->m_core_sim_order).begin();
         iter != (ptr->m_core_sim_order).end(); iter++) {
      out.push_back(*iter);
    }
    return std::make_unique<std::vector<unsigned>>(out);
  }

 private:
  const class trace_simt_core_cluster *ptr;
};
