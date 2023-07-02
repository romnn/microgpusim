#pragma once

#include <assert.h>
#include <cstddef>
#include <cstring>
#include <memory>

template <unsigned BSIZE> class mem_storage {
public:
  mem_storage(const mem_storage &another) {
    m_data = (unsigned char *)calloc(1, BSIZE);
    memcpy(m_data, another.m_data, BSIZE);
  }
  mem_storage() { m_data = (unsigned char *)calloc(1, BSIZE); }
  ~mem_storage() { free(m_data); }

  void write(unsigned offset, size_t length, const unsigned char *data) {
    assert(offset + length <= BSIZE);
    memcpy(m_data + offset, data, length);
  }

  void read(unsigned offset, size_t length, unsigned char *data) const {
    assert(offset + length <= BSIZE);
    memcpy(data, m_data + offset, length);
  }

  void print(const char *format, FILE *fout) const {
    unsigned int *i_data = (unsigned int *)m_data;
    for (int d = 0; d < (BSIZE / sizeof(unsigned int)); d++) {
      if (d % 1 == 0) {
        fprintf(fout, "\n");
      }
      fprintf(fout, format, i_data[d]);
      fprintf(fout, " ");
    }
    fprintf(fout, "\n");
    fflush(fout);
  }

private:
  unsigned m_nbytes;
  unsigned char *m_data;
};
