#include "checkpoint.hpp"

#include <cassert>
#include <sstream>
#include <sys/stat.h>

#include "memory_space.hpp"

checkpoint::checkpoint() {
  struct stat st = {0};

  if (stat("checkpoint_files", &st) == -1) {
    mkdir("checkpoint_files", 0777);
  }
}

void checkpoint::load_global_mem(class memory_space *temp_mem, char *f1name) {
  FILE *fp2 = fopen(f1name, "r");
  assert(fp2 != NULL);
  char line[128]; /* or other suitable maximum line size */
  unsigned int offset = 0;
  while (fgets(line, sizeof line, fp2) != NULL) /* read a line */
  {
    unsigned int index;
    char *pch;
    pch = strtok(line, " ");
    if (pch[0] == 'g' || pch[0] == 's' || pch[0] == 'l') {
      pch = strtok(NULL, " ");

      std::stringstream ss;
      ss << std::hex << pch;
      ss >> index;

      offset = 0;
    } else {
      unsigned int data;
      std::stringstream ss;
      ss << std::hex << pch;
      ss >> data;
      temp_mem->write_only(offset, index, 4, &data);
      offset = offset + 4;
    }
    // fputs ( line, stdout ); /* write the line */
  }
  fclose(fp2);
}

void checkpoint::store_global_mem(class memory_space *mem, char *fname,
                                  char *format) {
  FILE *fp3 = fopen(fname, "w");
  assert(fp3 != NULL);
  mem->print(format, fp3);
  fclose(fp3);
}
