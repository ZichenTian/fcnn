#ifndef __PADDING_H
#define __PADDING_H

#include "blob.h"
#include <fstream>

namespace fcnn {

static inline void padding_forward(Blob* bottom, Blob* padded, const int& left, const int& right, const int& top, const int& down) {
    const float* src_ptr = bottom->getPtr();
    float* dst_ptr = padded->getPtr();
    for(int c = 0; c < bottom->c; c++) {
        dst_ptr += padded->w * top;
        for(int h = 0; h < bottom->h; h++) {
            dst_ptr += left;
            for(int w = 0; w < bottom->w; w++) {
                *(dst_ptr++) = *(src_ptr++);
            }
            dst_ptr += right;
        }
        dst_ptr += padded->w * down;
    }
}

static inline void filter_padding_calculate(const int& input_h, const int& input_w, const int& kernel, const int& stride, 
                                            int& left, int& right, int& top, int& down, int& padded_h, int& padded_w) {
    padded_h = input_h + top + down;
    padded_w = input_w + left + right;
    int output_h = (padded_h-kernel)/stride+1;
    int output_w = (padded_w-kernel)/stride+1;
    if(output_h > 0) {
        // check and adjust pad_right and pad_down
        int remain = padded_h - ((output_h-1)*stride+kernel);
        if(remain > 0 && remain <= down) {
            down -= remain;
        } else if(remain > 0 && remain > down && remain < kernel) {
            down += remain;
        } else if(remain > 0 && remain > down && remain >= kernel) {
            ;
        }
        padded_h = input_h + top + down;
    } else {
        NOT_IMPLEMENT
    }
    if(output_w > 0) {
        int remain = padded_w - ((output_w-1)*stride+kernel);
        if(remain > 0 && remain <= right) {
            right -= remain;
        } else if(remain > 0 && remain > right && remain < kernel) {
            right += remain;
        } else if(remain > 0 && remain > right && remain >= kernel) {
            ;
        }
        padded_w = input_w + left + right;
    } else {
        NOT_IMPLEMENT
    }
}

}   // namespace fcnn


#endif  // __PADDING_H