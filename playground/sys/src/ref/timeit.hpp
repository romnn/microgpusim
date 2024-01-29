#include <chrono>
#include <utility>
#include <iostream>
#include <iomanip>
#include <unordered_map>

typedef std::chrono::high_resolution_clock::time_point Instant;

#define duration(a) std::chrono::duration_cast<std::chrono::nanoseconds>(a)
// std::chrono::duration_cast<std::chrono::nanoseconds>(a).count()

#define now() std::chrono::high_resolution_clock::now()

template <typename F, typename... Args>
// double timeit(F func, Args&&... args) {
std::chrono::nanoseconds timeit(F func, Args&&... args) {
  Instant t1 = now();
  func(std::forward<Args>(args)...);
  return duration(now() - t1);
}

void increment_timing(
    std::unordered_map<std::string, std::chrono::nanoseconds>& timings,
    std::string key, std::chrono::nanoseconds value);

std::ostream& human_time(std::ostream& os, std::chrono::nanoseconds ns);
