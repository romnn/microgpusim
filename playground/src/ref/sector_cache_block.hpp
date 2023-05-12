#pragma once

#include <assert.h>

#include "cache_block.hpp"
#include "mem_fetch.hpp"

struct sector_cache_block : public cache_block_t {
    sector_cache_block() { init(); }

    void init()
    {
        for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; ++i) {
            m_sector_alloc_time[i] = 0;
            m_sector_fill_time[i] = 0;
            m_last_sector_access_time[i] = 0;
            m_status[i] = INVALID;
            m_ignore_on_fill_status[i] = false;
            m_set_modified_on_fill[i] = false;
            m_set_readable_on_fill[i] = false;
            m_readable[i] = true;
        }
        m_line_alloc_time = 0;
        m_line_last_access_time = 0;
        m_line_fill_time = 0;
        m_dirty_byte_mask.reset();
    }

    virtual void allocate(new_addr_type tag, new_addr_type block_addr,
        unsigned time, mem_access_sector_mask_t sector_mask)
    {
        allocate_line(tag, block_addr, time, sector_mask);
    }

    void allocate_line(new_addr_type tag, new_addr_type block_addr, unsigned time,
        mem_access_sector_mask_t sector_mask)
    {
        // allocate a new line
        // assert(m_block_addr != 0 && m_block_addr != block_addr);
        init();
        m_tag = tag;
        m_block_addr = block_addr;

        unsigned sidx = get_sector_index(sector_mask);

        // set sector stats
        m_sector_alloc_time[sidx] = time;
        m_last_sector_access_time[sidx] = time;
        m_sector_fill_time[sidx] = 0;
        m_status[sidx] = RESERVED;
        m_ignore_on_fill_status[sidx] = false;
        m_set_modified_on_fill[sidx] = false;
        m_set_readable_on_fill[sidx] = false;
        m_set_byte_mask_on_fill = false;

        // set line stats
        m_line_alloc_time = time; // only set this for the first allocated sector
        m_line_last_access_time = time;
        m_line_fill_time = 0;
    }

    void allocate_sector(unsigned time, mem_access_sector_mask_t sector_mask)
    {
        // allocate invalid sector of this allocated valid line
        assert(is_valid_line());
        unsigned sidx = get_sector_index(sector_mask);

        // set sector stats
        m_sector_alloc_time[sidx] = time;
        m_last_sector_access_time[sidx] = time;
        m_sector_fill_time[sidx] = 0;
        if (m_status[sidx] == MODIFIED) // this should be the case only for
            // fetch-on-write policy //TO DO
            m_set_modified_on_fill[sidx] = true;
        else
            m_set_modified_on_fill[sidx] = false;

        m_set_readable_on_fill[sidx] = false;

        m_status[sidx] = RESERVED;
        m_ignore_on_fill_status[sidx] = false;
        // m_set_modified_on_fill[sidx] = false;
        m_readable[sidx] = true;

        // set line stats
        m_line_last_access_time = time;
        m_line_fill_time = 0;
    }

    virtual void fill(unsigned time, mem_access_sector_mask_t sector_mask,
        mem_access_byte_mask_t byte_mask)
    {
        unsigned sidx = get_sector_index(sector_mask);

        //	if(!m_ignore_on_fill_status[sidx])
        //	         assert( m_status[sidx] == RESERVED );
        m_status[sidx] = m_set_modified_on_fill[sidx] ? MODIFIED : VALID;

        if (m_set_readable_on_fill[sidx]) {
            m_readable[sidx] = true;
            m_set_readable_on_fill[sidx] = false;
        }
        if (m_set_byte_mask_on_fill)
            set_byte_mask(byte_mask);

        m_sector_fill_time[sidx] = time;
        m_line_fill_time = time;
    }
    virtual bool is_invalid_line()
    {
        // all the sectors should be invalid
        for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; ++i) {
            if (m_status[i] != INVALID)
                return false;
        }
        return true;
    }
    virtual bool is_valid_line() { return !(is_invalid_line()); }
    virtual bool is_reserved_line()
    {
        // if any of the sector is reserved, then the line is reserved
        for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; ++i) {
            if (m_status[i] == RESERVED)
                return true;
        }
        return false;
    }
    virtual bool is_modified_line()
    {
        // if any of the sector is modified, then the line is modified
        for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; ++i) {
            if (m_status[i] == MODIFIED)
                return true;
        }
        return false;
    }

    virtual enum cache_block_state
    get_status(mem_access_sector_mask_t sector_mask)
    {
        unsigned sidx = get_sector_index(sector_mask);

        return m_status[sidx];
    }

    virtual void set_status(enum cache_block_state status,
        mem_access_sector_mask_t sector_mask)
    {
        unsigned sidx = get_sector_index(sector_mask);
        m_status[sidx] = status;
    }

    virtual void set_byte_mask(mem_fetch* mf)
    {
        m_dirty_byte_mask = m_dirty_byte_mask | mf->get_access_byte_mask();
    }
    virtual void set_byte_mask(mem_access_byte_mask_t byte_mask)
    {
        m_dirty_byte_mask = m_dirty_byte_mask | byte_mask;
    }
    virtual mem_access_byte_mask_t get_dirty_byte_mask()
    {
        return m_dirty_byte_mask;
    }
    virtual mem_access_sector_mask_t get_dirty_sector_mask()
    {
        mem_access_sector_mask_t sector_mask;
        for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; i++) {
            if (m_status[i] == MODIFIED)
                sector_mask.set(i);
        }
        return sector_mask;
    }
    virtual unsigned long long get_last_access_time()
    {
        return m_line_last_access_time;
    }

    virtual void set_last_access_time(unsigned long long time,
        mem_access_sector_mask_t sector_mask)
    {
        unsigned sidx = get_sector_index(sector_mask);

        m_last_sector_access_time[sidx] = time;
        m_line_last_access_time = time;
    }

    virtual unsigned long long get_alloc_time() { return m_line_alloc_time; }

    virtual void set_ignore_on_fill(bool m_ignore,
        mem_access_sector_mask_t sector_mask)
    {
        unsigned sidx = get_sector_index(sector_mask);
        m_ignore_on_fill_status[sidx] = m_ignore;
    }

    virtual void set_modified_on_fill(bool m_modified,
        mem_access_sector_mask_t sector_mask)
    {
        unsigned sidx = get_sector_index(sector_mask);
        m_set_modified_on_fill[sidx] = m_modified;
    }
    virtual void set_byte_mask_on_fill(bool m_modified)
    {
        m_set_byte_mask_on_fill = m_modified;
    }

    virtual void set_readable_on_fill(bool readable,
        mem_access_sector_mask_t sector_mask)
    {
        unsigned sidx = get_sector_index(sector_mask);
        m_set_readable_on_fill[sidx] = readable;
    }
    virtual void set_m_readable(bool readable,
        mem_access_sector_mask_t sector_mask)
    {
        unsigned sidx = get_sector_index(sector_mask);
        m_readable[sidx] = readable;
    }

    virtual bool is_readable(mem_access_sector_mask_t sector_mask)
    {
        unsigned sidx = get_sector_index(sector_mask);
        return m_readable[sidx];
    }

    virtual unsigned get_modified_size()
    {
        unsigned modified = 0;
        for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; ++i) {
            if (m_status[i] == MODIFIED)
                modified++;
        }
        return modified * SECTOR_SIZE;
    }

    virtual void print_status()
    {
        printf("m_block_addr is %lu, status = %u %u %u %u\n", m_block_addr,
            m_status[0], m_status[1], m_status[2], m_status[3]);
    }

private:
    unsigned m_sector_alloc_time[SECTOR_CHUNCK_SIZE];
    unsigned m_last_sector_access_time[SECTOR_CHUNCK_SIZE];
    unsigned m_sector_fill_time[SECTOR_CHUNCK_SIZE];
    unsigned m_line_alloc_time;
    unsigned m_line_last_access_time;
    unsigned m_line_fill_time;
    cache_block_state m_status[SECTOR_CHUNCK_SIZE];
    bool m_ignore_on_fill_status[SECTOR_CHUNCK_SIZE];
    bool m_set_modified_on_fill[SECTOR_CHUNCK_SIZE];
    bool m_set_readable_on_fill[SECTOR_CHUNCK_SIZE];
    bool m_set_byte_mask_on_fill;
    bool m_readable[SECTOR_CHUNCK_SIZE];
    mem_access_byte_mask_t m_dirty_byte_mask;

    unsigned get_sector_index(mem_access_sector_mask_t sector_mask)
    {
        assert(sector_mask.count() == 1);
        for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; ++i) {
            if (sector_mask.to_ulong() & (1 << i))
                return i;
        }
        return (unsigned)-1; // ROMAN FIX
    }
};
