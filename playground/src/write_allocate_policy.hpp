#pragma once

enum write_allocate_policy_t {
  NO_WRITE_ALLOCATE,
  WRITE_ALLOCATE,
  FETCH_ON_WRITE,
  LAZY_FETCH_ON_READ
};
