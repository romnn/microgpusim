#pragma once

struct PowerscalingCoefficients {
  double int_coeff;
  double int_mul_coeff;
  double int_mul24_coeff;
  double int_mul32_coeff;
  double int_div_coeff;
  double fp_coeff;
  double dp_coeff;
  double fp_mul_coeff;
  double fp_div_coeff;
  double dp_mul_coeff;
  double dp_div_coeff;
  double sqrt_coeff;
  double log_coeff;
  double sin_coeff;
  double exp_coeff;
  double tensor_coeff;
  double tex_coeff;
};
