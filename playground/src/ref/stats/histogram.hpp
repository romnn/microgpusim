#pragma once

#include <cstdio>
#include <string>

class binned_histogram {
public:
  // creators
  binned_histogram(std::string name = "", int nbins = 32, int *bins = NULL);
  binned_histogram(const binned_histogram &other);
  virtual ~binned_histogram();

  // modifiers:
  void reset_bins();
  void add2bin(int sample);

  // accessors:
  void fprint(FILE *fout) const;

protected:
  std::string m_name;
  int m_nbins;
  int *m_bins;                // bin boundaries
  int *m_bin_cnts;            // counters
  int m_maximum;              // the maximum sample
  signed long long int m_sum; // for calculating the average
};

class pow2_histogram : public binned_histogram {
public:
  pow2_histogram(std::string name = "", int nbins = 32, int *bins = NULL);
  ~pow2_histogram() {}

  void add2bin(int sample);
};

class linear_histogram : public binned_histogram {
public:
  linear_histogram(int stride = 1, const char *name = NULL, int nbins = 32,
                   int *bins = NULL);
  ~linear_histogram() {}

  void add2bin(int sample);

private:
  int m_stride;
};
