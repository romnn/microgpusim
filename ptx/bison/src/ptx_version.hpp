#pragma once

#include "assert.h"
#include <cstdio>
#include <cstring>
#include <string>

class ptx_version {
public:
  ptx_version() {
    m_valid = false;
    m_ptx_version = 0;
    m_ptx_extensions = 0;
    m_sm_version_valid = false;
    m_texmode_unified = true;
    m_map_f64_to_f32 = true;
  }
  ptx_version(float ver, unsigned extensions) {
    m_valid = true;
    m_ptx_version = ver;
    m_ptx_extensions = extensions;
    m_sm_version_valid = false;
    m_texmode_unified = true;
  }
  void set_target(const char *sm_ver, const char *ext, const char *ext2) {
    assert(m_valid);
    m_sm_version_str = sm_ver;
    check_target_extension(ext);
    check_target_extension(ext2);
    sscanf(sm_ver, "%u", &m_sm_version);
    m_sm_version_valid = true;
  }
  float ver() const {
    assert(m_valid);
    return m_ptx_version;
  }
  unsigned target() const {
    assert(m_valid && m_sm_version_valid);
    return m_sm_version;
  }
  unsigned extensions() const {
    assert(m_valid);
    return m_ptx_extensions;
  }

private:
  void check_target_extension(const char *ext) {
    if (ext) {
      if (!strcmp(ext, "texmode_independent"))
        m_texmode_unified = false;
      else if (!strcmp(ext, "texmode_unified"))
        m_texmode_unified = true;
      else if (!strcmp(ext, "map_f64_to_f32"))
        m_map_f64_to_f32 = true;
      else
        abort();
    }
  }

  bool m_valid;
  float m_ptx_version;
  unsigned m_sm_version_valid;
  std::string m_sm_version_str;
  bool m_texmode_unified;
  bool m_map_f64_to_f32;
  unsigned m_sm_version;
  unsigned m_ptx_extensions;
};
