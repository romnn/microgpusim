#pragma once

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

unsigned int LOGB2(unsigned int v);
unsigned int intLOGB2(unsigned int v);

#define gs_min2(a, b) (((a) < (b)) ? (a) : (b))
#define min3(x, y, z) (((x) < (y) && (x) < (z)) ? (x) : (gs_min2((y), (z))))
