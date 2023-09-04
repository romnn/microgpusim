#include "visualizer.hpp"

#include <iostream>
#include <map>
#include <vector>
#include <zlib.h>

#include "memory_partition_unit.hpp"
#include "memory_stats.hpp"
#include "shader_core_stats.hpp"
#include "stats/tool.hpp"
#include "trace_gpgpu_sim.hpp"

class my_time_vector {
 private:
  std::map<unsigned int, std::vector<long int>> ld_time_map;
  std::map<unsigned int, std::vector<long int>> st_time_map;
  unsigned ld_vector_size;
  unsigned st_vector_size;
  std::vector<double> ld_time_dist;
  std::vector<double> st_time_dist;

  std::vector<double> overal_ld_time_dist;
  std::vector<double> overal_st_time_dist;
  int overal_ld_count;
  int overal_st_count;

 public:
  my_time_vector(int ld_size, int st_size) {
    ld_vector_size = ld_size;
    st_vector_size = st_size;
    ld_time_dist.resize(ld_size);
    st_time_dist.resize(st_size);
    overal_ld_time_dist.resize(ld_size);
    overal_st_time_dist.resize(st_size);
    overal_ld_count = 0;
    overal_st_count = 0;
  }
  void update_ld(unsigned int uid, unsigned int slot, long int time) {
    if (ld_time_map.find(uid) != ld_time_map.end()) {
      ld_time_map[uid][slot] = time;
    } else if (slot < NUM_MEM_REQ_STAT) {
      std::vector<long int> time_vec;
      time_vec.resize(ld_vector_size);
      time_vec[slot] = time;
      ld_time_map[uid] = time_vec;
    } else {
      // It's a merged mshr! forget it
    }
  }
  void update_st(unsigned int uid, unsigned int slot, long int time) {
    if (st_time_map.find(uid) != st_time_map.end()) {
      st_time_map[uid][slot] = time;
    } else {
      std::vector<long int> time_vec;
      time_vec.resize(st_vector_size);
      time_vec[slot] = time;
      st_time_map[uid] = time_vec;
    }
  }
  void check_ld_update(unsigned int uid, unsigned int slot, long int latency) {
    if (ld_time_map.find(uid) != ld_time_map.end()) {
      int our_latency =
          ld_time_map[uid][slot] - ld_time_map[uid][IN_ICNT_TO_MEM];
      assert(our_latency == latency);
    } else if (slot < NUM_MEM_REQ_STAT) {
      abort();
    }
  }
  void check_st_update(unsigned int uid, unsigned int slot, long int latency) {
    if (st_time_map.find(uid) != st_time_map.end()) {
      int our_latency =
          st_time_map[uid][slot] - st_time_map[uid][IN_ICNT_TO_MEM];
      assert(our_latency == latency);
    } else {
      abort();
    }
  }

 private:
  void calculate_ld_dist(void) {
    unsigned i, first;
    long int last_update, diff;
    int finished_count = 0;
    ld_time_dist.clear();
    ld_time_dist.resize(ld_vector_size);
    std::map<unsigned int, std::vector<long int>>::iterator iter, iter_temp;
    iter = ld_time_map.begin();
    while (iter != ld_time_map.end()) {
      last_update = 0;
      first = -1;
      if (!iter->second[IN_SHADER_FETCHED]) {
        // this request is not done yet skip it!
        ++iter;
        continue;
      }
      while (!last_update) {
        first++;
        assert(first < iter->second.size());
        last_update = iter->second[first];
      }

      for (i = first; i < ld_vector_size; i++) {
        diff = iter->second[i] - last_update;
        if (diff > 0) {
          ld_time_dist[i] += diff;
          last_update = iter->second[i];
        }
      }
      iter_temp = iter;
      iter++;
      ld_time_map.erase(iter_temp);
      finished_count++;
    }
    if (finished_count) {
      for (i = 0; i < ld_vector_size; i++) {
        overal_ld_time_dist[i] =
            (overal_ld_time_dist[i] * overal_ld_count + ld_time_dist[i]) /
            (overal_ld_count + finished_count);
      }
      overal_ld_count += finished_count;
      for (i = 0; i < ld_vector_size; i++) {
        ld_time_dist[i] /= finished_count;
      }
    }
  }

  void calculate_st_dist(void) {
    unsigned i, first;
    long int last_update, diff;
    int finished_count = 0;
    st_time_dist.clear();
    st_time_dist.resize(st_vector_size);
    std::map<unsigned int, std::vector<long int>>::iterator iter, iter_temp;
    iter = st_time_map.begin();
    while (iter != st_time_map.end()) {
      last_update = 0;
      first = -1;
      if (!iter->second[IN_SHADER_FETCHED]) {
        // this request is not done yet skip it!
        ++iter;
        continue;
      }
      while (!last_update) {
        first++;
        assert(first < iter->second.size());
        last_update = iter->second[first];
      }

      for (i = first; i < st_vector_size; i++) {
        diff = iter->second[i] - last_update;
        if (diff > 0) {
          st_time_dist[i] += diff;
          last_update = iter->second[i];
        }
      }
      iter_temp = iter;
      iter++;
      st_time_map.erase(iter_temp);
      finished_count++;
    }
    if (finished_count) {
      for (i = 0; i < st_vector_size; i++) {
        overal_st_time_dist[i] =
            (overal_st_time_dist[i] * overal_st_count + st_time_dist[i]) /
            (overal_st_count + finished_count);
      }
      overal_st_count += finished_count;
      for (i = 0; i < st_vector_size; i++) {
        st_time_dist[i] /= finished_count;
      }
    }
  }

 public:
  void clear_time_map_vectors(void) {
    ld_time_map.clear();
    st_time_map.clear();
  }
  void print_all_ld(void) {
    unsigned i;
    std::map<unsigned int, std::vector<long int>>::iterator iter;
    for (iter = ld_time_map.begin(); iter != ld_time_map.end(); ++iter) {
      std::cout << "ld_uid" << iter->first;
      for (i = 0; i < ld_vector_size; i++) {
        std::cout << " " << iter->second[i];
      }
      std::cout << std::endl;
    }
  }

  void print_all_st(void) {
    unsigned i;
    std::map<unsigned int, std::vector<long int>>::iterator iter;

    for (iter = st_time_map.begin(); iter != st_time_map.end(); ++iter) {
      std::cout << "st_uid" << iter->first;
      for (i = 0; i < st_vector_size; i++) {
        std::cout << " " << iter->second[i];
      }
      std::cout << std::endl;
    }
  }

  void calculate_dist() {
    calculate_ld_dist();
    calculate_st_dist();
  }
  void print_dist(FILE *fp) {
    unsigned i;
    calculate_dist();
    fprintf(fp, "LD_mem_lat_dist ");
    for (i = 0; i < ld_vector_size; i++) {
      fprintf(fp, " %d", (int)overal_ld_time_dist[i]);
    }
    fprintf(fp, "\n");
    fprintf(fp, "ST_mem_lat_dist ");
    for (i = 0; i < st_vector_size; i++) {
      fprintf(fp, " %d", (int)overal_st_time_dist[i]);
    }
    fprintf(fp, "\n");
  }

  void print_to_file(FILE *outfile) {
    unsigned i;
    calculate_dist();
    fprintf(outfile, "LDmemlatdist:");
    for (i = 0; i < ld_vector_size; i++) {
      fprintf(outfile, " %d", (int)ld_time_dist[i]);
    }
    fprintf(outfile, "\n");
    fprintf(outfile, "STmemlatdist:");
    for (i = 0; i < st_vector_size; i++) {
      fprintf(outfile, " %d", (int)st_time_dist[i]);
    }
    fprintf(outfile, "\n");
  }
  void print_to_gzfile(gzFile outfile) {
    unsigned i;
    calculate_dist();
    gzprintf(outfile, "LDmemlatdist:");
    for (i = 0; i < ld_vector_size; i++) {
      gzprintf(outfile, " %d", (int)ld_time_dist[i]);
    }
    gzprintf(outfile, "\n");
    gzprintf(outfile, "STmemlatdist:");
    for (i = 0; i < st_vector_size; i++) {
      gzprintf(outfile, " %d", (int)st_time_dist[i]);
    }
    gzprintf(outfile, "\n");
  }
};

my_time_vector *g_my_time_vector;

void time_vector_create(int size) {
  g_my_time_vector = new my_time_vector(size, size);
}

void time_vector_print(FILE *fp) { g_my_time_vector->print_dist(fp); }

void time_vector_print_interval2gzfile(gzFile outfile) {
  g_my_time_vector->print_to_gzfile(outfile);
}

void time_vector_update(unsigned int uid, int slot, long int cycle, int type) {
  if ((type == READ_REQUEST) || (type == READ_REPLY)) {
    g_my_time_vector->update_ld(uid, slot, cycle);
  } else if ((type == WRITE_REQUEST) || (type == WRITE_ACK)) {
    g_my_time_vector->update_st(uid, slot, cycle);
  } else {
    abort();
  }
}

void check_time_vector_update(unsigned int uid, int slot, long int latency,
                              int type) {
  if ((type == READ_REQUEST) || (type == READ_REPLY)) {
    g_my_time_vector->check_ld_update(uid, slot, latency);
  } else if ((type == WRITE_REQUEST) || (type == WRITE_ACK)) {
    g_my_time_vector->check_st_update(uid, slot, latency);
  } else {
    abort();
  }
}
