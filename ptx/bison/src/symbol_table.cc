#include "symbol_table.hpp"

#include "function_info.hpp"
#include "symbol.hpp"

symbol_table::symbol_table() { assert(0); }

symbol_table::symbol_table(const char *scope_name, unsigned entry_point,
                           symbol_table *parent, gpgpu_context *ctx) {
  gpgpu_ctx = ctx;
  m_scope_name = std::string(scope_name);
  m_reg_allocator = 0;
  m_shared_next = 0;
  m_const_next = 0;
  m_global_next = 0x100;
  m_local_next = 0;
  m_tex_next = 0;

  // Jin: handle instruction group for cdp
  m_inst_group_id = 0;

  m_parent = parent;
  if (m_parent) {
    m_shared_next = m_parent->m_shared_next;
    m_global_next = m_parent->m_global_next;
  }
}

void symbol_table::set_name(const char *name) {
  m_scope_name = std::string(name);
}

const ptx_version &symbol_table::get_ptx_version() const {
  if (m_parent == NULL)
    return m_ptx_version;
  else
    return m_parent->get_ptx_version();
}

unsigned symbol_table::get_sm_target() const {
  if (m_parent == NULL)
    return m_ptx_version.target();
  else
    return m_parent->get_sm_target();
}

void symbol_table::set_ptx_version(float ver, unsigned ext) {
  m_ptx_version = ptx_version(ver, ext);
}

void symbol_table::set_sm_target(const char *target, const char *ext,
                                 const char *ext2) {
  m_ptx_version.set_target(target, ext, ext2);
}

symbol *symbol_table::lookup(const char *identifier) {
  std::string key(identifier);
  std::map<std::string, symbol *>::iterator i = m_symbols.find(key);
  if (i != m_symbols.end()) {
    return i->second;
  }
  if (m_parent) {
    return m_parent->lookup(identifier);
  }
  return NULL;
}

symbol *symbol_table::add_variable(const char *identifier,
                                   const type_info *type, unsigned size,
                                   const char *filename, unsigned line) {
  char buf[1024];
  std::string key(identifier);
  assert(m_symbols.find(key) == m_symbols.end());
  snprintf(buf, 1024, "%s:%u", filename, line);
  symbol *s = new symbol(identifier, type, buf, size, gpgpu_ctx);
  m_symbols[key] = s;

  if (type != NULL && type->get_key().is_global()) {
    m_globals.push_back(s);
  }
  if (type != NULL && type->get_key().is_const()) {
    m_consts.push_back(s);
  }

  return s;
}

void symbol_table::add_function(function_info *func, const char *filename,
                                unsigned linenumber) {
  std::map<std::string, symbol *>::iterator i =
      m_symbols.find(func->get_name());
  if (i != m_symbols.end())
    return;
  char buf[1024];
  snprintf(buf, 1024, "%s:%u", filename, linenumber);
  type_info *type = add_type(func);
  symbol *s = new symbol(func->get_name().c_str(), type, buf, 0, gpgpu_ctx);
  s->set_function(func);
  m_symbols[func->get_name()] = s;
}

// Jin: handle instruction group for cdp
symbol_table *symbol_table::start_inst_group() {
  char inst_group_name[4096];
  snprintf(inst_group_name, 4096, "%s_inst_group_%u", m_scope_name.c_str(),
           m_inst_group_id);

  // previous added
  assert(m_inst_group_symtab.find(std::string(inst_group_name)) ==
         m_inst_group_symtab.end());
  symbol_table *sym_table =
      new symbol_table(inst_group_name, 3 /*inst group*/, this, gpgpu_ctx);

  sym_table->m_global_next = m_global_next;
  sym_table->m_shared_next = m_shared_next;
  sym_table->m_local_next = m_local_next;
  sym_table->m_reg_allocator = m_reg_allocator;
  sym_table->m_tex_next = m_tex_next;
  sym_table->m_const_next = m_const_next;

  m_inst_group_symtab[std::string(inst_group_name)] = sym_table;

  return sym_table;
}

symbol_table *symbol_table::end_inst_group() {
  symbol_table *sym_table = m_parent;

  sym_table->m_global_next = m_global_next;
  sym_table->m_shared_next = m_shared_next;
  sym_table->m_local_next = m_local_next;
  sym_table->m_reg_allocator = m_reg_allocator;
  sym_table->m_tex_next = m_tex_next;
  sym_table->m_const_next = m_const_next;
  sym_table->m_inst_group_id++;

  return sym_table;
}

// either libcuda or libopencl
// void register_ptx_function(const char *name, function_info *impl);

bool symbol_table::add_function_decl(const char *name, int entry_point,
                                     function_info **func_info,
                                     symbol_table **sym_table) {
  std::string key = std::string(name);
  bool prior_decl = false;
  if (m_function_info_lookup.find(key) != m_function_info_lookup.end()) {
    *func_info = m_function_info_lookup[key];
    prior_decl = true;
  } else {
    *func_info = new function_info(entry_point, gpgpu_ctx);
    (*func_info)->set_name(name);
    (*func_info)->set_maxnt_id(0);
    m_function_info_lookup[key] = *func_info;
  }

  if (m_function_symtab_lookup.find(key) != m_function_symtab_lookup.end()) {
    assert(prior_decl);
    *sym_table = m_function_symtab_lookup[key];
  } else {
    assert(!prior_decl);
    *sym_table = new symbol_table("", entry_point, this, gpgpu_ctx);

    // Initial setup code to support a register represented as "_".
    // This register is used when an instruction operand is
    // not read or written.  However, the parser must recognize it
    // as a legitimate register but we do not want to pass
    // it to the micro-architectural register to the performance simulator.
    // For this purpose we add a symbol to the symbol table but
    // mark it as a non_arch_reg so it does not effect the performance sim.
    type_info_key null_key(reg_space, 0, 0, 0, 0, 0);
    null_key.set_is_non_arch_reg();
    // First param is null - which is bad.
    // However, the first parameter is actually unread in the constructor...
    // TODO - remove the symbol_table* from type_info
    type_info *null_type_info = new type_info(NULL, null_key);
    symbol *null_reg =
        (*sym_table)->add_variable("_", null_type_info, 0, "", 0);
    null_reg->set_regno(0, 0);

    (*sym_table)->set_name(name);
    (*func_info)->set_symtab(*sym_table);
    m_function_symtab_lookup[key] = *sym_table;
    assert((*func_info)->get_symtab() == *sym_table);
    // register_ptx_function(name, *func_info);
  }
  return prior_decl;
}

function_info *symbol_table::lookup_function(std::string name) {
  std::string key = std::string(name);
  std::map<std::string, function_info *>::iterator it =
      m_function_info_lookup.find(key);
  assert(it != m_function_info_lookup.end());
  return it->second;
}

type_info *symbol_table::add_type(memory_space_t space_spec,
                                  int scalar_type_spec, int vector_spec,
                                  int alignment_spec, int extern_spec) {
  if (space_spec == param_space_unclassified)
    space_spec = param_space_local;
  type_info_key t(space_spec, scalar_type_spec, vector_spec, alignment_spec,
                  extern_spec, 0);
  type_info *pt;
  pt = new type_info(this, t);
  return pt;
}

type_info *symbol_table::add_type(function_info *func) {
  type_info_key t;
  type_info *pt;
  t.set_is_func();
  pt = new type_info(this, t);
  return pt;
}

type_info *symbol_table::get_array_type(type_info *base_type,
                                        unsigned array_dim) {
  type_info_key t = base_type->get_key();
  t.set_array_dim(array_dim);
  type_info *pt = new type_info(this, t);
  // Where else is m_types being used? As of now, I dont find any use of it and
  // causing seg fault. So disabling m_types.
  // TODO: find where m_types can be used in future and solve the seg fault.
  // pt = m_types[t] = new type_info(this,t);
  return pt;
}

void symbol_table::set_label_address(const symbol *label, unsigned addr) {
  std::map<std::string, symbol *>::iterator i = m_symbols.find(label->name());
  assert(i != m_symbols.end());
  symbol *s = i->second;
  s->set_label_address(addr);
}

void symbol_table::dump() {
  printf("\n\n");
  printf("Symbol table for \"%s\":\n", m_scope_name.c_str());
  std::map<std::string, symbol *>::iterator i;
  for (i = m_symbols.begin(); i != m_symbols.end(); i++) {
    printf("%30s : ", i->first.c_str());
    if (i->second)
      i->second->print_info(stdout);
    else
      printf(" <no symbol object> ");
    printf("\n");
  }
  printf("\n");
}
