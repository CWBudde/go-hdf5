// load.c - loads SOFA files with libmysofa and prints one line per file:
//
//   <file>: load err <code>   (mysofa_load failed)
//   <file>: check <code>      (mysofa_load succeeded; result of mysofa_check)
//
// Exits 1 if any file failed to load. Build with build.sh; the resulting
// binary is what TestLibmysofaLoad expects in $LIBMYSOFA_LOAD.
#include <stdio.h>

#include "mysofa.h"

int main(int argc, char **argv) {
  int rc = 0;
  for (int i = 1; i < argc; i++) {
    int err = 0;
    struct MYSOFA_HRTF *h = mysofa_load(argv[i], &err);
    if (!h) {
      printf("%s: load err %d\n", argv[i], err);
      rc = 1;
      continue;
    }
    printf("%s: check %d\n", argv[i], mysofa_check(h));
    mysofa_free(h);
  }
  return rc;
}
