#pragma once

enum cache_operator_type {
    CACHE_UNDEFINED,

    // loads
    CACHE_ALL, // .ca
    CACHE_LAST_USE, // .lu
    CACHE_VOLATILE, // .cv
    CACHE_L1, // .nc

    // loads and stores
    CACHE_STREAMING, // .cs
    CACHE_GLOBAL, // .cg

    // stores
    CACHE_WRITE_BACK, // .wb
    CACHE_WRITE_THROUGH // .wt
};
