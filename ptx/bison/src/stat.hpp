#pragma once

#include <iostream>
#include <ostream>
#include <string>
#include <vector>

class Stats {
  std::string _name;

  int _num_samples;
  double _sample_sum;
  double _sample_squared_sum;

  // bool _reset;
  double _min;
  double _max;

  int _num_bins;
  double _bin_size;

  std::vector<int> _hist;

public:
  Stats(const std::string &name, double bin_size = 1.0, int num_bins = 10);

  std::string Name() { return _name; }

  void Clear();

  double Average() const;
  double Variance() const;
  double Max() const;
  double Min() const;
  double Sum() const;
  double SquaredSum() const;
  int NumSamples() const;

  void AddSample(double val);
  inline void AddSample(int val) { AddSample((double)val); }
  inline void AddSample(unsigned long long val) { AddSample((double)val); }

  int GetBin(int b) { return _hist[b]; }

  void Display(std::ostream &os = std::cout) const;

  friend std::ostream &operator<<(std::ostream &os, const Stats &s);
};

std::ostream &operator<<(std::ostream &os, const Stats &s);

class Stats *StatCreate(const char *name, double bin_size, int num_bins);
void StatClear(void *st);
void StatAddSample(void *st, int val);
double StatAverage(void *st);
double StatMax(void *st);
double StatMin(void *st);
void StatDisp(void *st);
