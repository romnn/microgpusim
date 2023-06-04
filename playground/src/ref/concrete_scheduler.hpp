#pragma once

// Each of these corresponds to a string value in the gpgpsim.config file
// For example - to specify the LRR scheudler the config must contain lrr
enum concrete_scheduler {
  CONCRETE_SCHEDULER_LRR = 0,
  CONCRETE_SCHEDULER_GTO,
  CONCRETE_SCHEDULER_TWO_LEVEL_ACTIVE,
  CONCRETE_SCHEDULER_RRR,
  CONCRETE_SCHEDULER_WARP_LIMITING,
  CONCRETE_SCHEDULER_OLDEST_FIRST,
  NUM_CONCRETE_SCHEDULERS
};
