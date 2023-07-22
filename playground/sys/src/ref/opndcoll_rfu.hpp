#pragma once

#include <bitset>
#include <cstdio>
#include <list>
#include <memory>
#include <map>
#include <vector>

#include "io.hpp"
#include "hal.hpp"
#include "warp_instr.hpp"

class core_config;
class trace_shader_core_ctx;
class register_set;
class warp_inst_t;

int register_bank(int regnum, int wid, unsigned num_banks,
                  unsigned bank_warp_shift, bool sub_core_model,
                  unsigned banks_per_sched, unsigned sched_id);

typedef std::vector<register_set *> port_vector_t;
typedef std::vector<unsigned int> uint_vector_t;

enum operand_collector_unit_kind {
  SP_CUS,
  DP_CUS,
  SFU_CUS,
  TENSOR_CORE_CUS,
  INT_CUS,
  MEM_CUS,
  GEN_CUS
};

static const char *operand_collector_unit_kind_str[] = {
    "SP_CUS",  "DP_CUS",  "SFU_CUS", "TENSOR_CORE_CUS",
    "INT_CUS", "MEM_CUS", "GEN_CUS"};

class op_t;
class opndcoll_rfu_t;

class collector_unit_t {
 public:
  collector_unit_t(unsigned set_id, std::shared_ptr<spdlog::logger> logger);

  // accessors
  bool ready() const;
  const op_t *get_operands() const { return m_src_op; }
  void dump(FILE *fp, const trace_shader_core_ctx *shader) const;

  warp_inst_t *get_warp_instruction() const { return m_warp; }
  unsigned get_warp_id() const { return m_warp_id; }
  unsigned get_active_count() const { return m_warp->active_count(); }
  const active_mask_t &get_active_mask() const {
    return m_warp->get_active_mask();
  }
  std::unique_ptr<std::string> get_not_ready_mask() const {
    return std::make_unique<std::string>(mask_to_string(m_not_ready));
  }
  unsigned get_sp_op() const { return m_warp->sp_op; }
  unsigned get_id() const { return m_cuid; }  // returns CU hw id
  unsigned get_reg_id() const { return m_reg_id; }

  // modifiers
  void init(unsigned n, unsigned num_banks, unsigned log2_warp_size,
            const core_config *config, opndcoll_rfu_t *rfu,
            bool m_sub_core_model, unsigned reg_id,
            unsigned num_banks_per_sched);
  bool allocate(register_set *pipeline_reg, register_set *output_reg);

  void collect_operand(unsigned op) {
    logger->debug("collector unit [{}] {} collecting operand for {}", m_cuid,
                  warp_instr_ptr(m_warp), op);
    m_not_ready.reset(op);
  }
  unsigned get_num_operands() const { return m_warp->get_num_operands(); }
  unsigned get_num_regs() const { return m_warp->get_num_regs(); }
  void dispatch();
  bool is_free() const { return m_free; }

  register_set *get_output_register() const { return m_output_register; }

  std::shared_ptr<spdlog::logger> logger;

 private:
  bool m_free;
  unsigned m_set_id;
  unsigned m_cuid;  // collector unit hw id
  unsigned m_warp_id;
  warp_inst_t *m_warp;
  register_set *m_output_register;  // pipeline register to issue to when ready
  op_t *m_src_op;
  std::bitset<MAX_REG_OPERANDS * 2> m_not_ready;
  unsigned m_num_banks;
  unsigned m_bank_warp_shift;
  opndcoll_rfu_t *m_rfu;

  unsigned m_num_banks_per_sched;
  bool m_sub_core_model;
  unsigned m_reg_id;  // if sub_core_model enabled, limit regs this cu can r/w
};

// needs collector unit
class op_t {
 public:
  op_t() {
    m_valid = false;
    m_warp = NULL;
    m_cu = NULL;
    m_operand = (unsigned)-1;
    m_register = (unsigned)-1;
    m_shced_id = (unsigned)-1;
    m_bank = (unsigned)-1;
  }
  op_t(collector_unit_t *cu, unsigned op, unsigned reg, unsigned num_banks,
       unsigned bank_warp_shift, bool sub_core_model, unsigned banks_per_sched,
       unsigned sched_id) {
    m_valid = true;
    m_warp = NULL;
    m_cu = cu;
    m_operand = op;
    m_register = reg;
    m_shced_id = sched_id;
    m_bank = register_bank(reg, cu->get_warp_id(), num_banks, bank_warp_shift,
                           sub_core_model, banks_per_sched, sched_id);
  }
  op_t(const warp_inst_t *warp, unsigned reg, unsigned num_banks,
       unsigned bank_warp_shift, bool sub_core_model, unsigned banks_per_sched,
       unsigned sched_id) {
    m_valid = true;
    m_warp = warp;
    m_register = reg;
    m_cu = NULL;
    m_operand = -1;
    m_shced_id = sched_id;
    m_bank = register_bank(reg, warp->warp_id(), num_banks, bank_warp_shift,
                           sub_core_model, banks_per_sched, sched_id);
  }

  // accessors
  bool valid() const { return m_valid; }
  unsigned get_reg() const {
    assert(m_valid);
    return m_register;
  }
  unsigned get_wid() const {
    if (m_warp)
      return m_warp->warp_id();
    else if (m_cu)
      return m_cu->get_warp_id();
    else
      abort();
  }
  unsigned get_sid() const { return m_shced_id; }
  unsigned get_active_count() const {
    if (m_warp)
      return m_warp->active_count();
    else if (m_cu)
      return m_cu->get_active_count();
    else
      abort();
  }
  const active_mask_t &get_active_mask() {
    if (m_warp)
      return m_warp->get_active_mask();
    else if (m_cu)
      return m_cu->get_active_mask();
    else
      abort();
  }
  unsigned get_sp_op() const {
    if (m_warp)
      return m_warp->sp_op;
    else if (m_cu)
      return m_cu->get_sp_op();
    else
      abort();
  }
  unsigned get_oc_id() const { return m_cu->get_id(); }
  unsigned get_bank() const { return m_bank; }
  unsigned get_operand() const { return m_operand; }

  // void dump(FILE *fp) const {
  //   if (m_cu)
  //     fprintf(fp, " <R%u, CU:%u, w:%02u> ", m_register, m_cu->get_id(),
  //             m_cu->get_warp_id());
  //   else if (!m_warp->empty())
  //     fprintf(fp, " <R%u, wid:%02u> ", m_register, m_warp->warp_id());
  // }

  std::string get_reg_string() const {
    char buffer[64];
    snprintf(buffer, 64, "R%u", m_register);
    return std::string(buffer);
  }

  // modifiers
  void reset() { m_valid = false; }

  friend struct fmt::formatter<op_t>;
  friend std::ostream &operator<<(std::ostream &os, const op_t &op);

 private:
  bool m_valid;
  collector_unit_t *m_cu;
  const warp_inst_t *m_warp;
  unsigned m_operand;  // operand offset in instruction. e.g., add r1,r2,r3;
                       // r2 is oprd 0, r3 is 1 (r1 is dst)
  unsigned m_register;
  unsigned m_bank;
  unsigned m_shced_id;  // scheduler id that has issued this inst
};

template <>
struct fmt::formatter<op_t> {
  constexpr auto parse(format_parse_context &ctx)
      -> format_parse_context::iterator {
    return ctx.end();
  }

  auto format(const op_t &op, format_context &ctx) const
      -> format_context::iterator {
    return fmt::format_to(
        ctx.out(), "Op(operand={}, reg={}, bank={}, sched={})", op.m_operand,
        op.m_register, op.m_bank, op.m_shced_id);
  }
};

// needs nothing
enum alloc_t {
  NO_ALLOC,
  READ_ALLOC,
  WRITE_ALLOC,
};

// needs op and alloc_t
class allocation_t {
 public:
  allocation_t() { m_allocation = NO_ALLOC; }
  bool is_read() const { return m_allocation == READ_ALLOC; }
  bool is_write() const { return m_allocation == WRITE_ALLOC; }
  bool is_free() const { return m_allocation == NO_ALLOC; }

  // void dump(FILE *fp) const {
  //   if (m_allocation == NO_ALLOC) {
  //     fprintf(fp, "<free>");
  //   } else if (m_allocation == READ_ALLOC) {
  //     fprintf(fp, "rd: ");
  //     m_op.dump(fp);
  //   } else if (m_allocation == WRITE_ALLOC) {
  //     fprintf(fp, "wr: ");
  //     m_op.dump(fp);
  //   }
  //   fprintf(fp, "\n");
  // }

  void alloc_read(const op_t &op) {
    assert(is_free());
    m_allocation = READ_ALLOC;
    m_op = op;
  }
  void alloc_write(const op_t &op) {
    assert(is_free());
    m_allocation = WRITE_ALLOC;
    m_op = op;
  }
  void reset() { m_allocation = NO_ALLOC; }

 private:
  enum alloc_t m_allocation;
  op_t m_op;
};

class arbiter_t {
 public:
  // constructors
  arbiter_t(std::shared_ptr<spdlog::logger> logger) : logger(logger) {
    m_queue = NULL;
    m_allocated_bank = NULL;
    m_allocator_rr_head = NULL;
    _inmatch = NULL;
    _outmatch = NULL;
    _request = NULL;
    m_last_cu = 0;
  }

  void init(unsigned num_cu, unsigned num_banks) {
    assert(num_cu > 0);
    assert(num_banks > 0);
    m_num_collectors = num_cu;
    m_num_banks = num_banks;
    _inmatch = new int[m_num_banks];
    _outmatch = new int[m_num_collectors];
    _request = new int *[m_num_banks];
    for (unsigned i = 0; i < m_num_banks; i++)
      _request[i] = new int[m_num_collectors];
    m_queue = new std::list<op_t>[num_banks];
    m_allocated_bank = new allocation_t[num_banks];
    m_allocator_rr_head = new unsigned[num_cu];
    for (unsigned n = 0; n < num_cu; n++)
      m_allocator_rr_head[n] = n % num_banks;
    reset_alloction();
  }

  // accessors
  // void dump(FILE *fp) const {
  //   fprintf(fp, "\n");
  //   fprintf(fp, "  Arbiter State:\n");
  //   fprintf(fp, "  requests:\n");
  //   for (unsigned b = 0; b < m_num_banks; b++) {
  //     fprintf(fp, "    bank %u : ", b);
  //     std::list<op_t>::const_iterator o = m_queue[b].begin();
  //     for (; o != m_queue[b].end(); o++) {
  //       o->dump(fp);
  //     }
  //     fprintf(fp, "\n");
  //   }
  //   fprintf(fp, "  grants:\n");
  //   for (unsigned b = 0; b < m_num_banks; b++) {
  //     fprintf(fp, "    bank %u : ", b);
  //     m_allocated_bank[b].dump(fp);
  //   }
  //   fprintf(fp, "\n");
  // }

  // modifiers
  std::list<op_t> allocate_reads();

  void add_read_requests(collector_unit_t *cu) {
    const op_t *src = cu->get_operands();
    for (unsigned i = 0; i < MAX_REG_OPERANDS * 2; i++) {
      const op_t &op = src[i];
      if (op.valid()) {
        unsigned bank = op.get_bank();
        m_queue[bank].push_back(op);
      }
    }
  }
  bool bank_idle(unsigned bank) const {
    return m_allocated_bank[bank].is_free();
  }
  void allocate_bank_for_write(unsigned bank, const op_t &op) {
    assert(bank < m_num_banks);
    m_allocated_bank[bank].alloc_write(op);
  }
  void allocate_for_read(unsigned bank, const op_t &op) {
    assert(bank < m_num_banks);
    m_allocated_bank[bank].alloc_read(op);
  }
  void reset_alloction() {
    for (unsigned b = 0; b < m_num_banks; b++) m_allocated_bank[b].reset();
  }
  std::shared_ptr<spdlog::logger> logger;

 private:
  unsigned m_num_banks;
  unsigned m_num_collectors;

  allocation_t *m_allocated_bank;  // bank # -> register that wins
  std::list<op_t> *m_queue;

  unsigned
      *m_allocator_rr_head;  // cu # -> next bank to check for request (rr-arb)
  unsigned m_last_cu;        // first cu to check while arb-ing banks (rr)

  int *_inmatch;
  int *_outmatch;
  int **_request;
};

class input_port_t {
 public:
  input_port_t(port_vector_t &input, port_vector_t &output,
               uint_vector_t cu_sets)
      : m_in(input), m_out(output), m_cu_sets(cu_sets) {
    assert(input.size() == output.size());
    assert(not m_cu_sets.empty());
  }
  // private:
  port_vector_t m_in, m_out;
  uint_vector_t m_cu_sets;
};

class dispatch_unit_t {
 public:
  dispatch_unit_t(unsigned set_id, std::vector<collector_unit_t> *cus,
                  std::shared_ptr<spdlog::logger> logger)
      : logger(logger) {
    m_last_cu = 0;
    m_collector_units = cus;
    m_num_collectors = (*cus).size();
    m_next_cu = 0;
    m_set_id = set_id;

    m_sub_core_model = false;
    m_num_warp_scheds = (unsigned)-1;
  }
  void init(bool sub_core_model, unsigned num_warp_scheds) {
    m_sub_core_model = sub_core_model;
    m_num_warp_scheds = num_warp_scheds;
  }

  unsigned get_last_cu() const { return m_last_cu; }
  unsigned get_next_cu() const { return m_next_cu; }
  unsigned get_set_id() const { return m_set_id; }
  // unsigned get_kind() const { return m_last_cu; }

  collector_unit_t *find_ready() {
    // With sub-core enabled round robin starts with the next cu assigned to a
    // different sub-core than the one that dispatched last
    unsigned cusPerSched = m_num_collectors / m_num_warp_scheds;
    unsigned rr_increment =
        m_sub_core_model ? cusPerSched - (m_last_cu % cusPerSched) : 1;

    assert(m_set_id < 7);
    const char *kind = operand_collector_unit_kind_str[m_set_id];
    logger->debug(
        "dispatch unit {}: find ready: rr_inc = {}, last cu = {}, num "
        "collectors = {}, num warp schedulers = {}, cusPerSched = {}",
        kind, rr_increment, m_last_cu, m_num_collectors, m_num_warp_scheds,
        cusPerSched);

    for (unsigned n = 0; n < m_num_collectors; n++) {
      unsigned c = (m_last_cu + n + rr_increment) % m_num_collectors;
      // logger->debug("dispatch unit {}: checking collector unit {}", kind, c);

      if ((*m_collector_units)[c].ready()) {
        logger->debug(
            "dispatch unit {}: FOUND ready: chose collector unit {} (?)", kind,
            c);
        m_last_cu = c;
        return &((*m_collector_units)[c]);
      }
    }
    logger->debug("dispatch unit {}: did NOT find ready", kind);
    return NULL;
  }

  std::shared_ptr<spdlog::logger> logger;

 private:
  unsigned m_num_collectors;
  std::vector<collector_unit_t> *m_collector_units;
  unsigned m_last_cu;  // dispatch ready cu's rr
  // next cu is never used
  unsigned m_next_cu;  // for initialization
  bool m_sub_core_model;
  unsigned m_num_warp_scheds;
  unsigned m_set_id;
};

// operand collector based register file unit
class opndcoll_rfu_t {
 public:
  // constructors
  opndcoll_rfu_t(std::shared_ptr<spdlog::logger> logger)
      : logger(logger), m_arbiter(logger) {
    m_num_banks = 0;
    m_shader = NULL;
    m_initialized = false;
  }
  // std::shared_ptr<spdlog::logger> logger
  void add_cu_set(unsigned cu_set, unsigned num_cu, unsigned num_dispatch);
  void add_port(port_vector_t &input, port_vector_t &ouput,
                uint_vector_t cu_sets);

  void init(unsigned num_banks, trace_shader_core_ctx *shader);

  // modifiers
  bool writeback(warp_inst_t &warp);

  void step() {
    logger->debug("operand collector::step()");
    dispatch_ready_cu();
    allocate_reads();
    for (unsigned p = 0; p < m_in_ports.size(); p++) allocate_cu(p);
    process_banks();
  }

  // void dump(FILE *fp) const {
  //   fprintf(fp, "\n");
  //   fprintf(fp, "Operand Collector State:\n");
  //   for (unsigned n = 0; n < m_cu.size(); n++) {
  //     fprintf(fp, "   CU-%2u: ", n);
  //     m_cu[n]->dump(fp, m_shader);
  //   }
  //   m_arbiter.dump(fp);
  // }

  trace_shader_core_ctx *shader_core() { return m_shader; }

  std::shared_ptr<spdlog::logger> logger;

 private:
  void process_banks() { m_arbiter.reset_alloction(); }

  void dispatch_ready_cu();
  void allocate_cu(unsigned port);
  void allocate_reads();

  // types

  // opndcoll_rfu_t data members
  bool m_initialized;

  unsigned m_num_collector_sets;
  // unsigned m_num_collectors;
  unsigned m_num_banks;
  unsigned m_bank_warp_shift;
  unsigned m_warp_size;
  std::vector<collector_unit_t *> m_cu;
  arbiter_t m_arbiter;

  unsigned m_num_banks_per_sched;
  unsigned m_num_warp_scheds;
  bool sub_core_model;

  // unsigned m_num_ports;
  // std::vector<warp_inst_t**> m_input;
  // std::vector<warp_inst_t**> m_output;
  // std::vector<unsigned> m_num_collector_units;
  // warp_inst_t **m_alu_port;

  std::vector<input_port_t> m_in_ports;

  // map is ordered
  typedef std::map<unsigned /* collector set */,
                   std::vector<collector_unit_t> /*collector sets*/>
      cu_sets_t;
  cu_sets_t m_cus;
  std::vector<dispatch_unit_t> m_dispatch_units;

  // typedef std::map<warp_inst_t**/*port*/,dispatch_unit_t> port_to_du_t;
  // port_to_du_t                     m_dispatch_units;
  // std::map<warp_inst_t**,std::list<collector_unit_t*> > m_free_cu;
  trace_shader_core_ctx *m_shader;

  friend class operand_collector_bridge;
};
