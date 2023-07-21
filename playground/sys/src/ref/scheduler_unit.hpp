#pragma once

#include <algorithm>
#include <memory>
#include <vector>

#include "spdlog/logger.h"
#include "register_set.hpp"
#include "scoreboard.hpp"
#include "trace_shd_warp.hpp"
#include "trace_shader_core_ctx.hpp"

class shader_core_stats;
class simt_stack;
class trace_shader_core_ctx;

// this can be copied freely, so can be used in std containers.
class scheduler_unit {
 public:
  scheduler_unit(shader_core_stats *stats, trace_shader_core_ctx *shader,
                 Scoreboard *scoreboard, simt_stack **simt,
                 std::vector<trace_shd_warp_t *> *warp, register_set *sp_out,
                 register_set *dp_out, register_set *sfu_out,
                 register_set *int_out, register_set *tensor_core_out,
                 std::vector<register_set *> &spec_cores_out,
                 register_set *mem_out, int id)
      : logger(shader->logger),
        m_supervised_warps(),
        m_stats(stats),
        m_shader(shader),
        m_scoreboard(scoreboard),
        m_simt_stack(simt),
        /*m_pipeline_reg(pipe_regs),*/ m_warp(warp),
        m_sp_out(sp_out),
        m_dp_out(dp_out),
        m_sfu_out(sfu_out),
        m_int_out(int_out),
        m_tensor_core_out(tensor_core_out),
        m_mem_out(mem_out),
        m_spec_cores_out(spec_cores_out),
        m_id(id) {}
  virtual ~scheduler_unit() {}
  virtual void add_supervised_warp_id(int i) {
    m_supervised_warps.push_back(&warp(i));
  }
  virtual void done_adding_supervised_warps() {
    m_last_supervised_issued = m_supervised_warps.end();
  }

  virtual const char *name() = 0;

  // The core scheduler cycle method is meant to be common between
  // all the derived schedulers.  The scheduler's behaviour can be
  // modified by changing the contents of the m_next_cycle_prioritized_warps
  // list.
  void cycle();

  /**
   * A general function to order things in a Loose Round Robin way. The simplist
   * use of this function would be to implement a loose RR scheduler between all
   * the warps assigned to this core. A more sophisticated usage would be to
   * order a set of "fetch groups" in a RR fashion. In the first case, the
   * templated class variable would be a simple unsigned int representing the
   * warp_id.  In the 2lvl case, T could be a struct or a list representing a
   * set of warp_ids.
   * @param result_list: The resultant list the caller wants returned.  This
   * list is cleared and then populated in a loose round robin way
   * @param input_list: The list of things that should be put into the
   * result_list. For a simple scheduler this can simply be the
   * m_supervised_warps list.
   * @param last_issued_from_input:  An iterator pointing the last member in the
   * input_list that issued. Since this function orders in a RR fashion, the
   * object pointed to by this iterator will be last in the prioritization list
   * @param num_warps_to_add: The number of warps you want the scheudler to pick
   * between this cycle. Normally, this will be all the warps availible on the
   * core, i.e. m_supervised_warps.size(). However, a more sophisticated
   * scheduler may wish to limit this number. If the number if <
   * m_supervised_warps.size(), then only the warps with highest RR priority
   * will be placed in the result_list.
   */
  template <class T>
  void order_lrr(
      std::vector<T> &result_list, const typename std::vector<T> &input_list,
      const typename std::vector<T>::const_iterator &last_issued_from_input,
      unsigned num_warps_to_add) {
    assert(num_warps_to_add <= input_list.size());
    result_list.clear();
    typename std::vector<T>::const_iterator iter =
        (last_issued_from_input == input_list.end())
            ? input_list.begin()
            : last_issued_from_input + 1;

    for (unsigned count = 0; count < num_warps_to_add; ++iter, ++count) {
      if (iter == input_list.end()) {
        iter = input_list.begin();
      }
      result_list.push_back(*iter);
    }
  }

  template <class T>
  void order_rrr(
      std::vector<T> &result_list, const typename std::vector<T> &input_list,
      const typename std::vector<T>::const_iterator &last_issued_from_input,
      unsigned num_warps_to_add) {
    result_list.clear();

    if (m_num_issued_last_cycle > 0 || warp(m_current_turn_warp).done_exit() ||
        warp(m_current_turn_warp).waiting()) {
      std::vector<trace_shd_warp_t *>::const_iterator iter =
          (last_issued_from_input == input_list.end())
              ? input_list.begin()
              : last_issued_from_input + 1;
      for (unsigned count = 0; count < num_warps_to_add; ++iter, ++count) {
        if (iter == input_list.end()) {
          iter = input_list.begin();
        }
        unsigned warp_id = (*iter)->get_warp_id();
        if (!(*iter)->done_exit() && !(*iter)->waiting()) {
          result_list.push_back(*iter);
          m_current_turn_warp = warp_id;
          break;
        }
      }
    } else {
      result_list.push_back(&warp(m_current_turn_warp));
    }
  }

  enum OrderingType {
    // The item that issued last is prioritized first then the sorted result
    // of the priority_function
    ORDERING_GREEDY_THEN_PRIORITY_FUNC = 0,
    // No greedy scheduling based on last to issue. Only the priority function
    // determines priority
    ORDERED_PRIORITY_FUNC_ONLY,
    NUM_ORDERING,
  };

  /**
   * A general function to order things in an priority-based way.
   * The core usage of the function is similar to order_lrr.
   * The explanation of the additional parameters (beyond order_lrr) explains
   * the further extensions.
   * @param ordering: An enum that determines how the age function will be
   * treated in prioritization see the definition of OrderingType.
   * @param priority_function: This function is used to sort the input_list.  It
   * is passed to stl::sort as the sorting fucntion. So, if you wanted to sort a
   * list of integer warp_ids with the oldest warps having the most priority,
   * then the priority_function would compare the age of the two warps.
   */
  template <class T>
  void order_by_priority(
      std::vector<T> &result_list, const typename std::vector<T> &input_list,
      const typename std::vector<T>::const_iterator &last_issued_from_input,
      unsigned num_warps_to_add, OrderingType ordering,
      bool (*priority_func)(T lhs, T rhs)) {
    assert(num_warps_to_add <= input_list.size());
    result_list.clear();
    typename std::vector<T> temp = input_list;

    if (ORDERING_GREEDY_THEN_PRIORITY_FUNC == ordering) {
      T greedy_value = *last_issued_from_input;
      result_list.push_back(greedy_value);

      logger->trace("added greedy warp: {}",
                    greedy_value->get_dynamic_warp_id());

      // std::sort(temp.begin(), temp.end(), priority_func);
      std::stable_sort(temp.begin(), temp.end(), priority_func);
      typename std::vector<T>::iterator iter = temp.begin();
      for (unsigned count = 0; count < num_warps_to_add; ++count, ++iter) {
        if (*iter != greedy_value) {
          result_list.push_back(*iter);
        }
      }
    } else if (ORDERED_PRIORITY_FUNC_ONLY == ordering) {
      // std::sort(temp.begin(), temp.end(), priority_func);
      std::stable_sort(temp.begin(), temp.end(), priority_func);
      typename std::vector<T>::iterator iter = temp.begin();
      for (unsigned count = 0; count < num_warps_to_add; ++count, ++iter) {
        result_list.push_back(*iter);
      }
    } else {
      fprintf(stderr, "Unknown ordering - %d\n", ordering);
      abort();
    }
    assert(result_list.size() == num_warps_to_add);
  }
  // // These are some common ordering fucntions that the
  // // higher order schedulers can take advantage of
  // template <typename T>
  // void order_lrr(
  //     typename std::vector<T> &result_list,
  //     const typename std::vector<T> &input_list,
  //     const typename std::vector<T>::const_iterator &last_issued_from_input,
  //     unsigned num_warps_to_add);
  // template <typename T>
  // void order_rrr(
  //     typename std::vector<T> &result_list,
  //     const typename std::vector<T> &input_list,
  //     const typename std::vector<T>::const_iterator &last_issued_from_input,
  //     unsigned num_warps_to_add);

  // template <typename U>
  // void order_by_priority(
  //     std::vector<U> &result_list, const typename std::vector<U> &input_list,
  //     const typename std::vector<U>::const_iterator &last_issued_from_input,
  //     unsigned num_warps_to_add, OrderingType age_ordering,
  //     bool (*priority_func)(U lhs, U rhs));

  static bool sort_warps_by_oldest_dynamic_id(trace_shd_warp_t *lhs,
                                              trace_shd_warp_t *rhs);

  // Derived classes can override this function to populate
  // m_supervised_warps with their scheduling policies
  virtual void order_warps() = 0;

  int get_schd_id() const { return m_id; }

  std::shared_ptr<spdlog::logger> logger;

 protected:
  virtual void do_on_warp_issued(
      unsigned warp_id, unsigned num_issued,
      const std::vector<trace_shd_warp_t *>::const_iterator &prioritized_iter);
  inline int get_sid() const;

 protected:
  trace_shd_warp_t &warp(int i);

  // This is the prioritized warp list that is looped over each cycle to
  // determine which warp gets to issue.
  std::vector<trace_shd_warp_t *> m_next_cycle_prioritized_warps;
  // The m_supervised_warps list is all the warps this scheduler is supposed to
  // arbitrate between.  This is useful in systems where there is more than
  // one warp scheduler. In a single scheduler system, this is simply all
  // the warps assigned to this core.
  std::vector<trace_shd_warp_t *> m_supervised_warps;
  // This is the iterator pointer to the last supervised warp you issued
  std::vector<trace_shd_warp_t *>::const_iterator m_last_supervised_issued;
  shader_core_stats *m_stats;
  trace_shader_core_ctx *m_shader;
  // these things should become accessors: but would need a bigger rearchitect
  // of how shader_core_ctx interacts with its parts.
  Scoreboard *m_scoreboard;
  simt_stack **m_simt_stack;
  // warp_inst_t** m_pipeline_reg;
  std::vector<trace_shd_warp_t *> *m_warp;
  register_set *m_sp_out;
  register_set *m_dp_out;
  register_set *m_sfu_out;
  register_set *m_int_out;
  register_set *m_tensor_core_out;
  register_set *m_mem_out;
  std::vector<register_set *> &m_spec_cores_out;
  unsigned m_num_issued_last_cycle;
  unsigned m_current_turn_warp;

  int m_id;

  friend class scheduler_unit_bridge;
};

std::unique_ptr<scheduler_unit> new_scheduler_unit();
