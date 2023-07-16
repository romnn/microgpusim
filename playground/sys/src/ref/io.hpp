#pragma once

#include <iostream>
#include <queue>
#include <list>
#include <set>

// must take queue by-value (using copy constructor) for pop and print
template <typename T>
std::ostream &operator<<(std::ostream &os, std::queue<T> q) {
  os << "[";
  while (!q.empty()) {
    os << q.front() << ",";
    q.pop();
  }
  os << "]";
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
