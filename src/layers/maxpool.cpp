#include <float.h>
#include "maxpool.h"
#include "padding.h"

namespace fcnn {

#define MAXPOOL_V2 1

MaxPool::MaxPool(std::string name_) : Layer(name_), padded(NULL) { type = "MaxPool"; }

MaxPool::~MaxPool() {
    if(padded != NULL) {
        delete padded;
    }
}

int MaxPool::loadParam(std::vector<std::string> params, int offset) {
    kernel = string2int(params[offset]);
    stride = string2int(params[offset+1]);
    pad = string2int(params[offset+2]);
    if(kernel != 2 && kernel != 3) {
        INVALID_ERROR("kernel size should be 2 or 3!\n");
    }
    if(stride != 1 && stride != 2) {
        INVALID_ERROR("stride size should be 1 or 2!\n");
    }
    return 0;
}

int MaxPool::loadModel(std::ifstream& fp) {
    return 0;
}

std::vector<std::vector<int>> MaxPool::inferShape(std::vector<std::vector<int>> bottom_shapes) {
    std::vector<int> bottom_shape = bottom_shapes[0];
    if(bottom_shape.size() != 3) {
        INVALID_ERROR("shape should be (h,w,c)!\n");
    }
    // get correct pad
    pad_left = pad_right = pad_top = pad_down = pad;
    int padded_h, padded_w;
    filter_padding_calculate(bottom_shape[0], bottom_shape[1], kernel, stride, 
                             pad_left, pad_right, pad_top, pad_down, 
                             padded_h, padded_w);
    int channels = bottom_shape[2];
    // preparing padded blob
    if(pad_left || pad_right || pad_top || pad_down) {
        padded = new Blob(padded_h, padded_w, channels);
        padded->setValue(-FLT_MAX);
    }
    // get output shape
    output_h = (padded_h-kernel)/stride+1;
    output_w = (padded_w-kernel)/stride+1;
    std::vector<int> top_shape {output_h, output_w, channels};
    std::vector<std::vector<int>> top_shapes {top_shape};
    return top_shapes;
}

#ifdef MAXPOOL_V2

static inline void maxPoolKernel(float* src, float* dst, int output_h, int output_w, int ld_src, int ld_dst, 
                                 int kernel, int stride, int left, int right, int top, int down) {
    if(kernel==2 && stride==2) {
        int y = 0;
        if(top) {
            int x = 0;
            float* src_offset = src;
            if(left) {
                dst[x+ld_dst*y] = src_offset[0];
                src_offset += stride-1;
                x++;
            }
            for(; x < output_w-right; x++) {
                dst[x+ld_dst*y] = std::max(src_offset[0], src_offset[1]);
                src_offset += stride;
            }
            if(right) {
                dst[x+ld_dst*y] = src_offset[0];
            }
            y++;
        }

        for(; y < output_h-down; y++) {
            int x = 0;
            float* src_offset = src + (y*stride-top)*ld_src;
            if(left) {
                dst[x+ld_dst*y] = std::max(src_offset[0], src_offset[ld_src]);
                src_offset += stride-1;
                x++;
            }
            for(; x < output_w-right; x++) {
                dst[x+ld_dst*y] = std::max(std::max(src_offset[0], src_offset[1]), 
                                             std::max(src_offset[ld_src], src_offset[ld_src+1]));
                src_offset += stride;
            }
            if(right) {
                dst[x+ld_dst*y] = std::max(src_offset[0], src_offset[ld_src]);
            }
        }

        if(down) {
            int x = 0;
            float* src_offset = src + (y*stride-top)*ld_src;
            if(left) {
                dst[x+ld_dst*y] = src_offset[0];
                src_offset += stride-1;
                x++;
            }
            for(; x < output_w-right; x++) {
                dst[x+ld_dst*y] = std::max(src_offset[0], src_offset[1]);
                src_offset += stride;
            }
            if(right) {
                dst[x+ld_dst*y] = src_offset[0];
            }
        }
    } else if(kernel==3 && stride==2) {
        for(int y = 0; y < output_h; y++) {
            float* src_offset = src + y*ld_src*stride;
            for(int x = 0; x < output_w; x++) {
                dst[x + ld_dst*y] = std::max(
                    std::max(
                        std::max(std::max(src_offset[0], src_offset[1]), src_offset[2]), 
                        std::max(std::max(src_offset[ld_src], src_offset[ld_src+1]), src_offset[ld_src+2])
                    ), 
                    std::max(std::max(src_offset[2*ld_src], src_offset[2*ld_src+1]), src_offset[2*ld_src+2])
                );
                src_offset += 2;
            }
        }
    } else if(kernel==3 && stride==1) {
        for(int y = 0; y < output_h; y++) {
            float* src_offset = src + y*ld_src*stride;
            for(int x = 0; x < output_w; x++) {
                dst[x + ld_dst*y] = std::max(
                    std::max(
                        std::max(std::max(src_offset[0], src_offset[1]), src_offset[2]), 
                        std::max(std::max(src_offset[ld_src], src_offset[ld_src+1]), src_offset[ld_src+2])
                    ), 
                    std::max(std::max(src_offset[2*ld_src], src_offset[2*ld_src+1]), src_offset[2*ld_src+2])
                );
                src_offset += 1;
            }
        }
    } else {
        NOT_IMPLEMENT
    }
}

#else
static inline void maxPoolKernel(float* src, float* dst, int output_h, int output_w, int ld_src, int ld_dst, 
                                 int kernel, int stride) {
    if(kernel==2 && stride==2) {
        for(int y = 0; y < output_h; y++) {
            float* src_offset = src + y*ld_src*stride;
            for(int x = 0; x < output_w; x++) {
                dst[x + ld_dst*y] = std::max(std::max(src_offset[0], src_offset[1]), 
                                             std::max(src_offset[ld_src], src_offset[ld_src+1]));
                src_offset += 2;
            }
        }
    } else if(kernel==2 && stride==1) {
        for(int y = 0; y < output_h; y++) {
            float* src_offset = src + y*ld_src*stride;
            for(int x = 0; x < output_w; x++) {
                dst[x + ld_dst*y] = std::max(std::max(src_offset[0], src_offset[1]), 
                                             std::max(src_offset[ld_src], src_offset[ld_src+1]));
                src_offset += 1;
            }
        }
    } else if(kernel==3 && stride==2) {
        for(int y = 0; y < output_h; y++) {
            float* src_offset = src + y*ld_src*stride;
            for(int x = 0; x < output_w; x++) {
                dst[x + ld_dst*y] = std::max(
                    std::max(
                        std::max(std::max(src_offset[0], src_offset[1]), src_offset[2]), 
                        std::max(std::max(src_offset[ld_src], src_offset[ld_src+1]), src_offset[ld_src+2])
                    ), 
                    std::max(std::max(src_offset[2*ld_src], src_offset[2*ld_src+1]), src_offset[2*ld_src+2])
                );
                src_offset += 2;
            }
        }
    } else if(kernel==3 && stride==1) {
        for(int y = 0; y < output_h; y++) {
            float* src_offset = src + y*ld_src*stride;
            for(int x = 0; x < output_w; x++) {
                dst[x + ld_dst*y] = std::max(
                    std::max(
                        std::max(std::max(src_offset[0], src_offset[1]), src_offset[2]), 
                        std::max(std::max(src_offset[ld_src], src_offset[ld_src+1]), src_offset[ld_src+2])
                    ), 
                    std::max(std::max(src_offset[2*ld_src], src_offset[2*ld_src+1]), src_offset[2*ld_src+2])
                );
                src_offset += 1;
            }
        }
    } else {
        NOT_IMPLEMENT
    }
}
#endif

int MaxPool::forward(std::vector<Blob*>& bottoms, std::vector<Blob*>& tops) {
    Blob* bottom = bottoms[0];
    Blob* top = tops[0];
    // process padding
#ifndef MAXPOOL_V2
    if(padded != NULL) {
        padding_forward(bottom, padded, pad_left, pad_right, pad_top, pad_down);
        bottom = padded;
    }
#endif
    // calculate pooling
    float* bottom_data = bottom->getPtr();
    float* top_data = top->getPtr();
    int channels = bottom->c;
    for(int i = 0; i < channels; i++) {
#ifdef MAXPOOL_V2
        maxPoolKernel(bottom_data + i * bottom->h * bottom->w, 
                      top_data + i * output_h * output_w, 
                      output_h, 
                      output_w, 
                      bottom->w, 
                      output_w, 
                      kernel, 
                      stride, 
                      pad_left, pad_right, pad_top, pad_down);
#else
        maxPoolKernel(bottom_data + i * bottom->h * bottom->w, 
                      top_data + i * output_h * output_w, 
                      output_h, 
                      output_w, 
                      bottom->w, 
                      output_w, 
                      kernel, 
                      stride);
#endif
    }
}



}   // namespace fcnn