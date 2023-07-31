#pragma once

static const char *cache_event_type_str[] = {
    "WRITE_BACK_REQUEST_SENT",
    "READ_REQUEST_SENT",
    "WRITE_REQUEST_SENT",
    "WRITE_ALLOCATE_SENT",
};

enum cache_event_type {
  WRITE_BACK_REQUEST_SENT,
  READ_REQUEST_SENT,
  WRITE_REQUEST_SENT,
  WRITE_ALLOCATE_SENT
};
