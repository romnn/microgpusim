#pragma once

#include <iostream>
#include <queue>
#include <list>
#include <map>
#include <bitset>
#include <set>

template <typename T>
std::vector<T> queue_to_vector(std::queue<T> q) {
  std::vector<T> v;
  while (!q.empty()) {
    v.push_back(q.front());
    q.pop();
  }
  return v;
}

// must take queue by-value (using copy constructor) for pop and print
template <typename T>
std::ostream &operator<<(std::ostream &os, std::queue<T> q) {
  os << queue_to_vector(q);
  return os;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::list<T> &l) {
  os << "[";
  for (typename std::list<T>::const_iterator it = l.begin(); it != l.end();
       ++it) {
    os << *it << ",";
  }
  os << "]";
  return os;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, std::vector<T> v) {
  os << "[";
  for (typename std::vector<T>::const_iterator it = v.begin(); it != v.end();
       ++it) {
    os << *it << ",";
  }
  os << "]";
  return os;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::set<T> &s) {
  os << "[";
  for (typename std::set<T>::const_iterator it = s.begin(); it != s.end();
       ++it) {
    os << *it << ",";
  }
  os << "]";
  return os;
}

template <typename K, typename V>
std::ostream &operator<<(std::ostream &os, const std::map<K, V> &m) {
  os << "[";
  for (typename std::map<K, V>::const_iterator it = m.begin(); it != m.end();
       ++it) {
    os << it->first << ":" << it->second << ",";
  }
  os << "]";
  return os;
}

template <size_t N>
std::string mask_to_string(const std::bitset<N> &mask) {
  std::string out;
  for (int i = mask.size() - 1; i >= 0; i--)
    out.append(((mask[i]) ? "1" : "0"));
  return out;
}
