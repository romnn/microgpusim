#pragma once

#include <algorithm>
#include <assert.h>
#include <cstring>
#include <memory>

#include "allocate_policy.hpp"
#include "cache_type.hpp"
#include "func_cache.hpp"
#include "hal.hpp"
#include "mshr_config.hpp"
#include "replacement_policy.hpp"
#include "set_index_function.hpp"
#include "utils.hpp"
#include "write_allocate_policy.hpp"
#include "write_policy.hpp"

struct cache_config_params {
  bool valid;
  bool disabled;
  unsigned line_sz;
  unsigned n_set;
  unsigned assoc;
  unsigned atom_sz;
  enum replacement_policy_t replacement_policy;
  enum write_policy_t write_policy;
  enum allocation_policy_t alloc_policy;
  enum mshr_config_t mshr_type;
  enum cache_type cache_type;
  write_allocate_policy_t write_alloc_policy;
  set_index_function index_function;
  bool is_streaming;
  FuncCache status;
  unsigned wr_percent;
};

class cache_config {
public:
  cache_config() {
    m_valid = false;
    m_disabled = false;
    m_config_string = NULL; // set by option parser
    m_config_stringPrefL1 = NULL;
    m_config_stringPrefShared = NULL;
    m_data_port_width = 0;
    m_set_index_function = LINEAR_SET_FUNCTION;
    m_is_streaming = false;
    m_wr_percent = 0;
  }

  cache_config(cache_config_params params) : cache_config(){};

  cache_config(bool valid, bool disabled, unsigned line_sz, unsigned n_set,
               unsigned assoc, unsigned atom_sz,
               enum replacement_policy_t replacement_policy,
               enum write_policy_t write_policy,
               enum allocation_policy_t alloc_policy,
               enum mshr_config_t mshr_type, enum cache_type cache_type,
               write_allocate_policy_t write_alloc_policy,
               set_index_function index_function, bool is_streaming,
               FuncCache status, unsigned wr_percent) {
    m_valid = valid;
    m_disabled = disabled;
    m_line_sz = line_sz;
    m_line_sz_log2 = line_sz;
    m_nset = n_set;
    m_nset_log2 = n_set;
    m_assoc = assoc;
    m_atom_sz = atom_sz;
    m_sector_sz_log2 = atom_sz; // todo
    m_replacement_policy = replacement_policy;
    m_write_policy = write_policy;
    m_alloc_policy = alloc_policy;
    m_mshr_type = mshr_type;
    m_cache_type = cache_type;
    m_write_alloc_policy = write_alloc_policy;
    m_set_index_function = index_function;
    m_is_streaming = is_streaming;
    m_wr_percent = wr_percent;
    m_cache_status = status;
  }

  void init(char *config, FuncCache status) {
    m_cache_status = status;
    assert(config);
    char ct, rp, wp, ap, mshr_type, wap, sif;

    int ntok =
        sscanf(config, "%c:%u:%u:%u,%c:%c:%c:%c:%c,%c:%u:%u,%u:%u,%u", &ct,
               &m_nset, &m_line_sz, &m_assoc, &rp, &wp, &ap, &wap, &sif,
               &mshr_type, &m_mshr_entries, &m_mshr_max_merge,
               &m_miss_queue_size, &m_result_fifo_entries, &m_data_port_width);

    if (ntok < 12) {
      if (!std::strcmp(config, "none")) {
        m_disabled = true;
        return;
      }
      exit_parse_error();
    }

    switch (ct) {
    case 'N':
      m_cache_type = NORMAL;
      break;
    case 'S':
      m_cache_type = SECTOR;
      break;
    default:
      exit_parse_error();
    }
    switch (rp) {
    case 'L':
      m_replacement_policy = LRU;
      break;
    case 'F':
      m_replacement_policy = FIFO;
      break;
    default:
      exit_parse_error();
    }
    switch (wp) {
    case 'R':
      m_write_policy = READ_ONLY;
      break;
    case 'B':
      m_write_policy = WRITE_BACK;
      break;
    case 'T':
      m_write_policy = WRITE_THROUGH;
      break;
    case 'E':
      m_write_policy = WRITE_EVICT;
      break;
    case 'L':
      m_write_policy = LOCAL_WB_GLOBAL_WT;
      break;
    default:
      exit_parse_error();
    }
    switch (ap) {
    case 'm':
      m_alloc_policy = ON_MISS;
      break;
    case 'f':
      m_alloc_policy = ON_FILL;
      break;
    case 's':
      m_alloc_policy = STREAMING;
      break;
    default:
      exit_parse_error();
    }
    if (m_alloc_policy == STREAMING) {
      /*
For streaming cache:
(1) we set the alloc policy to be on-fill to remove all line_alloc_fail
stalls. if the whole memory is allocated to the L1 cache, then make the
allocation to be on_MISS otherwise, make it ON_FILL to eliminate line
allocation fails. i.e. MSHR throughput is the same, independent on the L1
cache size/associativity So, we set the allocation policy per kernel
basis, see shader.cc, max_cta() function

(2) We also set the MSHRs to be equal to max
allocated cache lines. This is possible by moving TAG to be shared
between cache line and MSHR enrty (i.e. for each cache line, there is
an MSHR rntey associated with it). This is the easiest think we can
think of to model (mimic) L1 streaming cache in Pascal and Volta

For more information about streaming cache, see:
http://on-demand.gputechconf.com/gtc/2017/presentation/s7798-luke-durant-inside-volta.pdf
https://ieeexplore.ieee.org/document/8344474/
*/
      m_is_streaming = true;
      m_alloc_policy = ON_FILL;
    }
    switch (mshr_type) {
    case 'F':
      m_mshr_type = TEX_FIFO;
      assert(ntok == 14);
      break;
    case 'T':
      m_mshr_type = SECTOR_TEX_FIFO;
      assert(ntok == 14);
      break;
    case 'A':
      m_mshr_type = ASSOC;
      break;
    case 'S':
      m_mshr_type = SECTOR_ASSOC;
      break;
    default:
      exit_parse_error();
    }
    m_line_sz_log2 = LOGB2(m_line_sz);
    m_nset_log2 = LOGB2(m_nset);
    m_valid = true;
    m_atom_sz = (m_cache_type == SECTOR) ? SECTOR_SIZE : m_line_sz;
    m_sector_sz_log2 = LOGB2(SECTOR_SIZE);
    original_m_assoc = m_assoc;

    // For more details about difference between FETCH_ON_WRITE and WRITE
    // VALIDAE policies Read: Jouppi, Norman P. "Cache write policies and
    // performance". ISCA 93. WRITE_ALLOCATE is the old write policy in
    // GPGPU-sim 3.x, that send WRITE and READ for every write request
    switch (wap) {
    case 'N':
      m_write_alloc_policy = NO_WRITE_ALLOCATE;
      break;
    case 'W':
      m_write_alloc_policy = WRITE_ALLOCATE;
      break;
    case 'F':
      m_write_alloc_policy = FETCH_ON_WRITE;
      break;
    case 'L':
      m_write_alloc_policy = LAZY_FETCH_ON_READ;
      break;
    default:
      exit_parse_error();
    }

    // detect invalid configuration
    if ((m_alloc_policy == ON_FILL || m_alloc_policy == STREAMING) and
        m_write_policy == WRITE_BACK) {
      // A writeback cache with allocate-on-fill policy will inevitably lead to
      // deadlock: The deadlock happens when an incoming cache-fill evicts a
      // dirty line, generating a writeback request.  If the memory subsystem is
      // congested, the interconnection network may not have sufficient buffer
      // for the writeback request.  This stalls the incoming cache-fill.  The
      // stall may propagate through the memory subsystem back to the output
      // port of the same core, creating a deadlock where the wrtieback request
      // and the incoming cache-fill are stalling each other.
      assert(0 &&
             "Invalid cache configuration: Writeback cache cannot allocate new "
             "line on fill. ");
    }

    if ((m_write_alloc_policy == FETCH_ON_WRITE ||
         m_write_alloc_policy == LAZY_FETCH_ON_READ) &&
        m_alloc_policy == ON_FILL) {
      assert(
          0 &&
          "Invalid cache configuration: FETCH_ON_WRITE and LAZY_FETCH_ON_READ "
          "cannot work properly with ON_FILL policy. Cache must be ON_MISS. ");
    }
    if (m_cache_type == SECTOR) {
      assert(m_line_sz / SECTOR_SIZE == SECTOR_CHUNCK_SIZE &&
             m_line_sz % SECTOR_SIZE == 0);
    }

    // default: port to data array width and granularity = line size
    if (m_data_port_width == 0) {
      m_data_port_width = m_line_sz;
    }
    assert(m_line_sz % m_data_port_width == 0);

    switch (sif) {
    case 'H':
      m_set_index_function = FERMI_HASH_SET_FUNCTION;
      break;
    case 'P':
      m_set_index_function = HASH_IPOLY_FUNCTION;
      break;
    case 'C':
      m_set_index_function = CUSTOM_SET_FUNCTION;
      break;
    case 'L':
      m_set_index_function = LINEAR_SET_FUNCTION;
      break;
    case 'X':
      m_set_index_function = BITWISE_XORING_FUNCTION;
      break;
    default:
      exit_parse_error();
    }
  }
  bool disabled() const { return m_disabled; }
  unsigned get_line_sz() const {
    assert(m_valid);
    return m_line_sz;
  }
  unsigned get_atom_sz() const {
    assert(m_valid);
    return m_atom_sz;
  }
  unsigned get_num_lines() const {
    assert(m_valid);
    return m_nset * m_assoc;
  }
  unsigned get_max_num_lines() const {
    assert(m_valid);
    return get_max_cache_multiplier() * m_nset * original_m_assoc;
  }
  unsigned get_max_assoc() const {
    assert(m_valid);
    return get_max_cache_multiplier() * original_m_assoc;
  }
  void print(FILE *fp) const {
    fprintf(fp, "Size = %d B (%d Set x %d-way x %d byte line)\n",
            m_line_sz * m_nset * m_assoc, m_nset, m_assoc, m_line_sz);
  }

  virtual unsigned set_index(new_addr_type addr) const;

  virtual unsigned get_max_cache_multiplier() const {
    return MAX_DEFAULT_CACHE_SIZE_MULTIBLIER;
  }

  unsigned hash_function(new_addr_type addr, unsigned m_nset,
                         unsigned m_line_sz_log2, unsigned m_nset_log2,
                         unsigned m_index_function) const;

  new_addr_type tag(new_addr_type addr) const {
    // For generality, the tag includes both index and tag. This allows for more
    // complex set index calculations that can result in different indexes
    // mapping to the same set, thus the full tag + index is required to check
    // for hit/miss. Tag is now identical to the block address.

    // return addr >> (m_line_sz_log2+m_nset_log2);
    return addr & ~(new_addr_type)(m_line_sz - 1);
  }
  new_addr_type block_addr(new_addr_type addr) const {
    return addr & ~(new_addr_type)(m_line_sz - 1);
  }
  new_addr_type mshr_addr(new_addr_type addr) const {
    return addr & ~(new_addr_type)(m_atom_sz - 1);
  }
  enum mshr_config_t get_mshr_type() const { return m_mshr_type; }
  void set_assoc(unsigned n) {
    // set new assoc. L1 cache dynamically resized in Volta
    m_assoc = n;
  }
  unsigned get_nset() const {
    assert(m_valid);
    return m_nset;
  }
  unsigned get_total_size_inKB() const {
    assert(m_valid);
    return (m_assoc * m_nset * m_line_sz) / 1024;
  }
  bool is_streaming() const { return m_is_streaming; }
  FuncCache get_cache_status() const { return m_cache_status; }
  void set_allocation_policy(enum allocation_policy_t alloc) {
    m_alloc_policy = alloc;
  }

  write_allocate_policy_t get_write_allocate_policy() const {
    return m_write_alloc_policy;
  }
  write_policy_t get_write_policy() const { return m_write_policy; }

  char *m_config_string;
  char *m_config_stringPrefL1;
  char *m_config_stringPrefShared;
  FuncCache m_cache_status;
  unsigned m_wr_percent;

protected:
  void exit_parse_error() {
    printf("GPGPU-Sim uArch: cache configuration parsing error (%s)\n",
           m_config_string);
    abort();
  }

  bool m_valid;
  bool m_disabled;
  unsigned m_line_sz;
  unsigned m_line_sz_log2;
  unsigned m_nset;
  unsigned m_nset_log2;
  unsigned m_assoc;
  unsigned m_atom_sz;
  unsigned m_sector_sz_log2;
  unsigned original_m_assoc;
  bool m_is_streaming;

  enum replacement_policy_t m_replacement_policy; // 'L' = LRU, 'F' = FIFO
  enum write_policy_t
      m_write_policy; // 'T' = write through, 'B' = write back, 'R' = read only
  enum allocation_policy_t
      m_alloc_policy; // 'm' = allocate on miss, 'f' = allocate on fill
  enum mshr_config_t m_mshr_type;
  enum cache_type m_cache_type;

  write_allocate_policy_t
      m_write_alloc_policy; // 'W' = Write allocate, 'N' = No write allocate

  union {
    unsigned m_mshr_entries;
    unsigned m_fragment_fifo_entries;
  };
  union {
    unsigned m_mshr_max_merge;
    unsigned m_request_fifo_entries;
  };
  union {
    unsigned m_miss_queue_size;
    unsigned m_rob_entries;
  };
  unsigned m_result_fifo_entries;
  unsigned m_data_port_width; //< number of byte the cache can access per cycle
  enum set_index_function
      m_set_index_function; // Hash, linear, or custom set index function

  friend class tag_array;
  friend class baseline_cache;
  friend class read_only_cache;
  friend class tex_cache;
  friend class data_cache;
  friend class l1_cache;
  friend class l2_cache;
  friend class memory_sub_partition;
};

// std::unique_ptr<cache_config> new_cache_config();
std::unique_ptr<cache_config> new_cache_config(cache_config_params);
