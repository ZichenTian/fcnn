#include "convolution.h"
#include "padding.h"

namespace fcnn {

Convolution::Convolution(std::string name_) : Layer(name_), padded(NULL), conv_weight(NULL), bias_weight(NULL) { type="Convolution"; }

Convolution::~Convolution() {
    if(padded != NULL) {
        delete padded;
    }
    if(conv_weight != NULL) {
        delete conv_weight;
    }
    if(bias_weight != NULL) {
        delete bias_weight;
    }
}

int Convolution::loadModel(std::ifstream& fp) {
    conv_weight = new Blob(kernel, kernel, input_c*output_c);
    fp.read((char*)(conv_weight->getPtr()), conv_weight->size*sizeof(float));
    if(bias) {
        bias_weight = new Blob(output_c);
        fp.read((char*)(bias_weight->getPtr()), bias_weight->size*sizeof(float));
    }
}

int Convolution::loadParam(std::vector<std::string> params, int offset) {
    kernel = string2int(params[offset]);
    stride = string2int(params[offset+1]);
    output_c = string2int(params[offset+2]);
    pad = string2int(params[offset+3]);
    bias = string2int(params[offset+4]);
    return 0;
}

std::vector<std::vector<int>> Convolution::inferShape(std::vector<std::vector<int>> bottom_shapes) {
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
    int padded_c = bottom_shape[2];
    input_c = padded_c;
    if(pad_left || pad_right || pad_top || pad_down) {
        padded = new Blob(padded_h, padded_w, padded_c);
        padded->setValue(0);
    }
    // get output shape
    output_h = (padded_h-kernel)/stride+1;
    output_w = (padded_w-kernel)/stride+1;
    std::vector<int> top_shape {output_h, output_w, output_c};
    std::vector<std::vector<int>> top_shapes {top_shape};
    return top_shapes;
}

int Convolution::forward(std::vector<Blob*>& bottoms, std::vector<Blob*>& tops) {
    Blob* bottom = bottoms[0];
    Blob* top = tops[0];
    // process padding
    if(padded != NULL) {
        padding_forward(bottom, padded, pad_left, pad_right, pad_top, pad_down);
        bottom = padded;
    }
    // start to calculate convolution
    const float* bottom_data = bottom->getPtr();
    const float* conv_weight_data = conv_weight->getPtr();
    const float* bias_weight_data = bias==1?bias_weight->getPtr():NULL;
    float* top_data = top->getPtr();
    const float* src_offset = bottom_data;
    for(int oc = 0; oc < output_c; oc++) {
        for(int oh = 0; oh < output_h; oh++) {
            for(int ow = 0; ow < output_w; ow++) {
                const float* src_offset = bottom_data + oh*stride*bottom->w + ow*stride;
                for(int ic = 0; ic < bottom->c; ic++) {
                    for(int kh = 0; kh < kernel; kh++) {
                        for(int kw = 0; kw < kernel; kw++) {
                            float weight = conv_weight_data[oc*conv_weight->h*conv_weight->w*input_c + ic*conv_weight->h*conv_weight->w + kh*conv_weight->w + kw];
                            *top_data += weight*src_offset[ic*bottom->h*bottom->w + kh*bottom->w + kw];
                        }
                    }
                }
                if(bias)
                    *top_data += bias_weight_data[oc];
                top_data++;
            }
        }
    }
    return 0;

}



}   // namespace fcnn