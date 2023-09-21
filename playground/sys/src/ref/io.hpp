#pragma once

#include <iostream>
#include <deque>
#include <queue>
#include <list>
#include <map>
#include <bitset>
#include <set>

template <typename T>
std::vector<T> deque_to_vector(std::deque<T> q) {
  std::vector<T> v;
  typename std::deque<T>::const_iterator iter;
  for (iter = q.begin(); iter != q.end(); ++iter) {
    v.push_back(*iter);
  }
  return v;
}

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
  typename std::list<T>::const_iterator iter;
  for (iter = l.begin(); iter != l.end(); ++iter) {
    os << *iter << ",";
  }
  os << "]";
  return os;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, std::vector<T> v) {
  os << "[";
  typename std::vector<T>::const_iterator iter;
  for (iter = v.begin(); iter != v.end(); ++iter) {
    os << *iter << ",";
  }
  os << "]";
  return os;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::set<T> &s) {
  os << "[";
  typename std::set<T>::const_iterator iter;
  for (iter = s.begin(); iter != s.end(); ++iter) {
    os << *iter << ",";
  }
  os << "]";
  return os;
}

template <typename K, typename V>
std::ostream &operator<<(std::ostream &os, const std::map<K, V> &m) {
  os << "[";
  typename std::map<K, V>::const_iterator iter;
  for (iter = m.begin(); iter != m.end(); ++iter) {
    os << iter->first << ":" << iter->second << ",";
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
