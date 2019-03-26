#ifndef __TEST_COMMON_H
#define __TEST_COMMON_H

#include <random>
#include <ctime>
#include <string>

#include "src/blob.h"
#include "src/layer.h"

namespace fcnn {

static inline std::string int2string(int x) {
    char tmp[256];
    sprintf(tmp, "%d", x);
    return std::string(tmp);
}

static inline void fill_blob_random(Blob& blob) {
    float* data_ptr = blob.getPtr();
    for(int i = 0; i < blob.size; i++) {
        *(data_ptr++) = ((double)rand()-RAND_MAX/2)/RAND_MAX;
    }
}

static inline int rand_num(int min, int max) {
    return rand()%(max-min+1) + min;
}

}   // namespace fcnn


#endif  // __TEST_COMMON_H