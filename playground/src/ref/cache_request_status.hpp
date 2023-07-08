#pragma once

static const char *cache_request_status_str[] = {
    "HIT",         "HIT_RESERVED", "MISS", "RESERVATION_FAIL",
    "SECTOR_MISS", "MSHR_HIT"};

enum cache_request_status {
  HIT = 0,
  HIT_RESERVED,
  MISS,
  RESERVATION_FAIL,
  SECTOR_MISS,
  MSHR_HIT,
  NUM_CACHE_REQUEST_STATUS
};

const char *get_cache_request_status_str(enum cache_request_status status);
