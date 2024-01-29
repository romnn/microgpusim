#include "timeit.hpp"

// void increment_timing(
//     std::unordered_map<std::string, std::chrono::nanoseconds>& timings,
//     std::string key, std::chrono::nanoseconds value) {
//   auto result = timings.insert({key, value});
//   if (!result.second) {
//     // element was already present: increment the value at the key
//     result.first->second = result.first->second + value;
//   }
// }

std::ostream& human_time(std::ostream& os, std::chrono::nanoseconds ns) {
  using namespace std;
  using namespace std::chrono;
  typedef duration<int, ratio<86400>> days;
  // char fill = os.fill();
  // os.fill('0');
  auto d = duration_cast<days>(ns);
  ns -= d;
  auto h = duration_cast<hours>(ns);
  ns -= h;
  auto m = duration_cast<minutes>(ns);
  ns -= m;
  auto s = duration_cast<seconds>(ns);
  ns -= s;
  auto ms = duration_cast<milliseconds>(ns);
  ns -= ms;
  auto qs = duration_cast<microseconds>(ns);
  ns -= qs;
  if (d.count() > 0) {
    os << d.count() << "d:";
    // os << setw(2) << d.count() << "d:";
  }
  if (h.count() > 0) {
    os << h.count() << "h:";
    // os << setw(2) << h.count() << "h:";
  }
  if (m.count() > 0) {
    os << m.count() << "m:";
    // os << setw(2) << m.count() << "m:";
  }
  if (s.count() > 0) {
    os << s.count() << 's';
    // os << setw(2) << s.count() << 's';
  }
  if (ms.count() > 0) {
    os << ms.count() << "ms";
    // os << setw(2) << ms.count() << "ms";
  }
  if (qs.count() > 0) {
    os << qs.count() << "qs";
    // os << setw(2) << ms.count() << "qs";
  }

  // os.fill(fill);
  return os;
};
