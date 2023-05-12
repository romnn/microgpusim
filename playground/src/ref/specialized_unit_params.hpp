#pragma once

struct specialized_unit_params {
  unsigned latency;
  unsigned num_units;
  unsigned id_oc_spec_reg_width;
  unsigned oc_ex_spec_reg_width;
  char name[20];
  unsigned ID_OC_SPEC_ID;
  unsigned OC_EX_SPEC_ID;
};
