#pragma once

// logger snapshot trigger:
// automate the snap_shot part of loggers to avoid modifying simulation
// loop everytime a new time-dependent stat is added
class snap_shot_trigger {
 public:
  snap_shot_trigger(unsigned long long interval)
      : m_snap_shot_interval(interval) {}
  virtual ~snap_shot_trigger() {}

  void try_snap_shot(unsigned long long current_cycle) {
    if ((current_cycle % m_snap_shot_interval == 0) && current_cycle != 0) {
      snap_shot(current_cycle);
    }
  }

  virtual void snap_shot(unsigned long long current_cycle) = 0;

  const unsigned long long &get_interval() const {
    return m_snap_shot_interval;
  }

 protected:
  unsigned long long m_snap_shot_interval;
};
