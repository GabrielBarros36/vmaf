/**
 * AFL++ persistent mode adapter.
 *
 * Only compiled for AFL++ builds. Wraps the LLVMFuzzerTestOneInput
 * interface to use AFL++'s persistent mode for efficient fuzzing.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

__AFL_FUZZ_INIT();

extern int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

int main(void) {
    __AFL_INIT();
    unsigned char *buf = __AFL_FUZZ_TESTCASE_BUF;

    while (__AFL_LOOP(10000)) {
        int len = __AFL_FUZZ_TESTCASE_LEN;
        LLVMFuzzerTestOneInput(buf, len);
    }
    return 0;
}
