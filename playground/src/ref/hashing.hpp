#pragma once

#include "hal.hpp"

unsigned ipoly_hash_function(new_addr_type higher_bits, unsigned index,
    unsigned bank_set_num);

unsigned bitwise_hash_function(new_addr_type higher_bits, unsigned index,
    unsigned bank_set_num);

unsigned PAE_hash_function(new_addr_type higher_bits, unsigned index,
    unsigned bank_set_num);
