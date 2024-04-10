#include "stat.hpp"

#include <cmath>
#include <limits>

Stats::Stats(const std::string &name, double bin_size, int num_bins)
    : _name(name), _num_bins(num_bins), _bin_size(bin_size) {
  Clear();
}

void Stats::Clear() {
  _num_samples = 0;
  _sample_sum = 0.0;
  _sample_squared_sum = 0.0;

  _hist.assign(_num_bins, 0);

  _min = std::numeric_limits<double>::quiet_NaN();
  _max = -std::numeric_limits<double>::quiet_NaN();

  //  _reset = true;
}

double Stats::Average() const { return _sample_sum / (double)_num_samples; }

double Stats::Variance() const {
  return (_sample_squared_sum * (double)_num_samples -
          _sample_sum * _sample_sum) /
         ((double)_num_samples * (double)_num_samples);
}

double Stats::Min() const { return _min; }

double Stats::Max() const { return _max; }

double Stats::Sum() const { return _sample_sum; }

double Stats::SquaredSum() const { return _sample_squared_sum; }

int Stats::NumSamples() const { return _num_samples; }

void Stats::AddSample(double val) {
  ++_num_samples;
  _sample_sum += val;

  // NOTE: the negation ensures that NaN values are handled correctly!
  _max = !(val <= _max) ? val : _max;
  _min = !(val >= _min) ? val : _min;

  // double clamp between 0 and num_bins-1
  int b = (int)fmax(floor(val / _bin_size), 0.0);
  b = (b >= _num_bins) ? (_num_bins - 1) : b;

  _hist[b]++;
}

void Stats::Display(std::ostream &os) const { os << *this << std::endl; }

std::ostream &operator<<(std::ostream &os, const Stats &s) {
  std::vector<int> const &v = s._hist;
  os << "[ ";
  for (size_t i = 0; i < v.size(); ++i) {
    os << v[i] << " ";
  }
  os << "]";
  return os;
}

Stats *StatCreate(const char *name, double bin_size, int num_bins) {
  Stats *newstat = new Stats(name, bin_size, num_bins);
  newstat->Clear();
  return newstat;
}

void StatClear(void *st) { ((Stats *)st)->Clear(); }

void StatAddSample(void *st, int val) { ((Stats *)st)->AddSample(val); }

double StatAverage(void *st) { return ((Stats *)st)->Average(); }

double StatMax(void *st) { return ((Stats *)st)->Max(); }

double StatMin(void *st) { return ((Stats *)st)->Min(); }

void StatDisp(void *st) {
  Stats *stat = (Stats *)st;
  printf("Stats for %s", stat->Name().c_str());
  // ()->DisplayHierarchy()/*  */;
  //   if (((Stats *)st)->NeverUsed()) {
  //      printf (" was never updated!\n");
  //   } else {
  printf("Min %f Max %f Average %f \n", ((Stats *)st)->Min(),
         ((Stats *)st)->Max(), StatAverage(st));
  ((Stats *)st)->Display();
  //   }
}
