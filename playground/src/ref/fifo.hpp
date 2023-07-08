#pragma once

#include <assert.h>
#include <cstdio>
#include <sstream>

template <class T>
struct fifo_data {
  T *m_data;
  fifo_data *m_next;
};

template <class T>
class fifo_pipeline {
 public:
  fifo_pipeline(const char *nm, unsigned int minlen, unsigned int maxlen) {
    assert(maxlen);
    m_name = nm;
    m_min_len = minlen;
    m_max_len = maxlen;
    m_length = 0;
    m_n_element = 0;
    m_head = NULL;
    m_tail = NULL;
    for (unsigned i = 0; i < m_min_len; i++) push(NULL);
  }

  ~fifo_pipeline() {
    while (m_head) {
      m_tail = m_head;
      m_head = m_head->m_next;
      delete m_tail;
    }
  }

  void push(T *data) {
    assert(m_length < m_max_len);
    if (m_head) {
      if (m_tail->m_data || m_length < m_min_len) {
        m_tail->m_next = new fifo_data<T>();
        m_tail = m_tail->m_next;
        m_length++;
        m_n_element++;
      }
    } else {
      m_head = m_tail = new fifo_data<T>();
      m_length++;
      m_n_element++;
    }
    m_tail->m_next = NULL;
    m_tail->m_data = data;
  }

  T *pop() {
    fifo_data<T> *next;
    T *data;
    if (m_head) {
      next = m_head->m_next;
      data = m_head->m_data;
      if (m_head == m_tail) {
        assert(next == NULL);
        m_tail = NULL;
      }
      delete m_head;
      m_head = next;
      m_length--;
      if (m_length == 0) {
        assert(m_head == NULL);
        m_tail = m_head;
      }
      m_n_element--;
      if (m_min_len && m_length < m_min_len) {
        push(NULL);
        m_n_element--;  // uncount NULL elements inserted to create delays
      }
    } else {
      data = NULL;
    }
    return data;
  }

  T *top() const {
    if (m_head) {
      return m_head->m_data;
    } else {
      return NULL;
    }
  }

  void set_min_length(unsigned int new_min_len) {
    if (new_min_len == m_min_len) return;

    if (new_min_len > m_min_len) {
      m_min_len = new_min_len;
      while (m_length < m_min_len) {
        push(NULL);
        m_n_element--;  // uncount NULL elements inserted to create delays
      }
    } else {
      // in this branch imply that the original min_len is larger then 0
      // ie. head != 0
      assert(m_head);
      m_min_len = new_min_len;
      while ((m_length > m_min_len) && (m_tail->m_data == 0)) {
        fifo_data<T> *iter;
        iter = m_head;
        while (iter && (iter->m_next != m_tail)) iter = iter->m_next;
        if (!iter) {
          // there is only one node, and that node is empty
          assert(m_head->m_data == 0);
          pop();
        } else {
          // there are more than one node, and tail node is empty
          assert(iter->m_next == m_tail);
          delete m_tail;
          m_tail = iter;
          m_tail->m_next = 0;
          m_length--;
        }
      }
    }
  }

  bool full() const { return (m_max_len && m_length >= m_max_len); }
  bool is_available_size(unsigned size) const {
    return (m_max_len && m_length + size - 1 >= m_max_len);
  }
  bool empty() const { return m_head == NULL; }
  unsigned get_n_element() const { return m_n_element; }
  unsigned get_length() const { return m_length; }
  unsigned get_max_len() const { return m_max_len; }

  // std::string to_string() const {
  //   std::stringstream buffer;
  //   fifo_data<T> *ddp = m_head;
  //   buffer << m_name << "{" << m_length << "}";
  //   buffer << "[";
  //   while (ddp) {
  //     buffer << ddp->m_data << ",";
  //     ddp = ddp->m_next;
  //   }
  //   buffer << "]";
  //   return buffer.str();
  // }
  //
  // void print() const {
  //   printf("%d", get_n_element());
  //
  //   // fifo_data<T> *ddp = m_head;
  //   //   printf("%s(%d): ", m_name, m_length);
  //   //   while (ddp) {
  //   //     printf("%p ", ddp->m_data);
  //   //     ddp = ddp->m_next;
  //   //   }
  //   //   printf("\n");
  // }

  const char *m_name;

  fifo_data<T> *m_tail;
  fifo_data<T> *m_head;

 private:
  unsigned int m_min_len;
  unsigned int m_max_len;
  unsigned int m_length;
  unsigned int m_n_element;
};

template <class T>
std::ostream &operator<<(std::ostream &os, const fifo_pipeline<T> *pipe) {
  if (pipe == NULL) {
    os << "NULL";
  } else {
    os << "[ ";
    fifo_data<T> *ddp = pipe->m_head;
    while (ddp) {
      os << ddp->m_data << ",";
      ddp = ddp->m_next;
    }
    os << "]";
  }
  return os;
}
