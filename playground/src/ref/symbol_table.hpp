#pragma once

#include <list>
#include <map>

// #include "gpgpu_context.hpp"
#include "ptx_version.hpp"
#include "symbol.hpp"
#include "type_info.hpp"

class gpgpu_context;

class symbol_table {
public:
  symbol_table();
  symbol_table(const char *scope_name, unsigned entry_point,
               symbol_table *parent, gpgpu_context *ctx);
  void set_name(const char *name);
  const ptx_version &get_ptx_version() const;
  unsigned get_sm_target() const;
  void set_ptx_version(float ver, unsigned ext);
  void set_sm_target(const char *target, const char *ext, const char *ext2);
  symbol *lookup(const char *identifier);
  std::string get_scope_name() const { return m_scope_name; }
  symbol *add_variable(const char *identifier, const type_info *type,
                       unsigned size, const char *filename, unsigned line);
  void add_function(function_info *func, const char *filename,
                    unsigned linenumber);
  bool add_function_decl(const char *name, int entry_point,
                         function_info **func_info,
                         symbol_table **symbol_table);
  function_info *lookup_function(std::string name);
  type_info *add_type(memory_space_t space_spec, int scalar_type_spec,
                      int vector_spec, int alignment_spec, int extern_spec);
  type_info *add_type(function_info *func);
  type_info *get_array_type(type_info *base_type, unsigned array_dim);
  void set_label_address(const symbol *label, unsigned addr);
  unsigned next_reg_num() { return ++m_reg_allocator; }
  addr_t get_shared_next() { return m_shared_next; }
  addr_t get_sstarr_next() { return m_sstarr_next; }
  addr_t get_global_next() { return m_global_next; }
  addr_t get_local_next() { return m_local_next; }
  addr_t get_tex_next() { return m_tex_next; }
  void alloc_shared(unsigned num_bytes) { m_shared_next += num_bytes; }
  void alloc_sstarr(unsigned num_bytes) { m_sstarr_next += num_bytes; }
  void alloc_global(unsigned num_bytes) { m_global_next += num_bytes; }
  void alloc_local(unsigned num_bytes) { m_local_next += num_bytes; }
  void alloc_tex(unsigned num_bytes) { m_tex_next += num_bytes; }

  typedef std::list<symbol *>::iterator iterator;

  iterator global_iterator_begin() { return m_globals.begin(); }
  iterator global_iterator_end() { return m_globals.end(); }

  iterator const_iterator_begin() { return m_consts.begin(); }
  iterator const_iterator_end() { return m_consts.end(); }

  void dump();

  // Jin: handle instruction group for cdp
  symbol_table *start_inst_group();
  symbol_table *end_inst_group();

  // backward pointer
  class gpgpu_context *gpgpu_ctx;

private:
  unsigned m_reg_allocator;
  unsigned m_shared_next;
  unsigned m_sstarr_next;
  unsigned m_const_next;
  unsigned m_global_next;
  unsigned m_local_next;
  unsigned m_tex_next;

  symbol_table *m_parent;
  ptx_version m_ptx_version;
  std::string m_scope_name;
  std::map<std::string, symbol *>
      m_symbols; // map from name of register to pointers to the registers
  std::map<type_info_key, type_info *, type_info_key_compare> m_types;
  std::list<symbol *> m_globals;
  std::list<symbol *> m_consts;
  std::map<std::string, function_info *> m_function_info_lookup;
  std::map<std::string, symbol_table *> m_function_symtab_lookup;

  // Jin: handle instruction group for cdp
  unsigned m_inst_group_id;
  std::map<std::string, symbol_table *> m_inst_group_symtab;
};
