#pragma once

enum scheduler_prioritization_type {
  SCHEDULER_PRIORITIZATION_LRR = 0,  // Loose Round Robin
  SCHEDULER_PRIORITIZATION_SRR,      // Strict Round Robin
  SCHEDULER_PRIORITIZATION_GTO,      // Greedy Then Oldest
  SCHEDULER_PRIORITIZATION_GTLRR,    // Greedy Then Loose Round Robin
  SCHEDULER_PRIORITIZATION_GTY,      // Greedy Then Youngest
  SCHEDULER_PRIORITIZATION_OLDEST,   // Oldest First
  SCHEDULER_PRIORITIZATION_YOUNGEST, // Youngest First
};
