#include "type_info.hpp"

#include "ptx_recognizer.hpp"

// requires yyscan_t defined in ptx_recognizer
#include "ptx.parser.tab.h"

unsigned type_info_key::type_decode(size_t &size, int &basic_type) const {
  int type = scalar_type();
  return type_decode(type, size, basic_type);
}

const char *decode_token(int type) { return g_ptx_token_decode[type].c_str(); }

unsigned type_info_key::type_decode(int type, size_t &size, int &basic_type) {
  switch (type) {
  case S8_TYPE:
    size = 8;
    basic_type = 1;
    return 0;
  case S16_TYPE:
    size = 16;
    basic_type = 1;
    return 1;
  case S32_TYPE:
    size = 32;
    basic_type = 1;
    return 2;
  case S64_TYPE:
    size = 64;
    basic_type = 1;
    return 3;
  case U8_TYPE:
    size = 8;
    basic_type = 0;
    return 4;
  case U16_TYPE:
    size = 16;
    basic_type = 0;
    return 5;
  case U32_TYPE:
    size = 32;
    basic_type = 0;
    return 6;
  case U64_TYPE:
    size = 64;
    basic_type = 0;
    return 7;
  case F16_TYPE:
    size = 16;
    basic_type = -1;
    return 8;
  case F32_TYPE:
    size = 32;
    basic_type = -1;
    return 9;
  case F64_TYPE:
    size = 64;
    basic_type = -1;
    return 10;
  case FF64_TYPE:
    size = 64;
    basic_type = -1;
    return 10;
  case PRED_TYPE:
    size = 1;
    basic_type = 2;
    return 11;
  case B8_TYPE:
    size = 8;
    basic_type = 0;
    return 12;
  case B16_TYPE:
    size = 16;
    basic_type = 0;
    return 13;
  case B32_TYPE:
    size = 32;
    basic_type = 0;
    return 14;
  case B64_TYPE:
    size = 64;
    basic_type = 0;
    return 15;
  case BB64_TYPE:
    size = 64;
    basic_type = 0;
    return 15;
  case BB128_TYPE:
    size = 128;
    basic_type = 0;
    return 16;
  case TEXREF_TYPE:
  case SAMPLERREF_TYPE:
  case SURFREF_TYPE:
    size = 32;
    basic_type = 3;
    return 16;
  default:
    printf("ERROR ** type_decode() does not know about \"%s\"\n",
           decode_token(type));
    assert(0);
    return 0xDEADBEEF;
  }
}
