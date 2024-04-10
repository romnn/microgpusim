#include "symbol.hpp"

#include "gpgpu_context.hpp"

unsigned symbol::get_uid() {
  unsigned result = (gpgpu_ctx->symbol_sm_next_uid)++;
  return result;
}

void symbol::add_initializer(const std::list<operand_info> &init) {
  m_initializer = init;
}

void symbol::print_info(FILE *fp) const {
  fprintf(fp, "uid:%u, decl:%s, type:%p, ", m_uid, m_decl_location.c_str(),
          m_type);
  if (m_address_valid)
    fprintf(fp, "<address valid>, ");
  if (m_is_label)
    fprintf(fp, " is_label ");
  if (m_is_shared)
    fprintf(fp, " is_shared ");
  if (m_is_const)
    fprintf(fp, " is_const ");
  if (m_is_global)
    fprintf(fp, " is_global ");
  if (m_is_local)
    fprintf(fp, " is_local ");
  if (m_is_tex)
    fprintf(fp, " is_tex ");
  if (m_is_func_addr)
    fprintf(fp, " is_func_addr ");
  if (m_function)
    fprintf(fp, " %p ", m_function);
}
