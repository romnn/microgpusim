#pragma once

enum cache_request_status {
    HIT = 0,
    HIT_RESERVED,
    MISS,
    RESERVATION_FAIL,
    SECTOR_MISS,
    MSHR_HIT,
    NUM_CACHE_REQUEST_STATUS
};
