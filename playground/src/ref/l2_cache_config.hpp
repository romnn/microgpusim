#pragma once

#include "addrdec.hpp"
#include "cache_config.hpp"

class l2_cache_config : public cache_config {
public:
    l2_cache_config()
        : cache_config()
    {
    }
    void init(linear_to_raw_address_translation* address_mapping);
    virtual unsigned set_index(new_addr_type addr) const;

private:
    linear_to_raw_address_translation* m_address_mapping;
};
