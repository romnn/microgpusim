#pragma once

static const char* allocation_policy_t_str[] = {"ON_MISS", "ON_FILL",
                                                "STREAMING"};

enum allocation_policy_t { ON_MISS, ON_FILL, STREAMING };
