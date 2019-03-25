#include "inner_product.h"

namespace fcnn {

InnerProduct::InnerProduct(std::string name_) : Layer(name_), multi_weight(NULL), bias_weight(NULL) { type = "InnerProduct"; }

InnerProduct::~InnerProduct() {
    if(multi_weight != NULL) {
        delete multi_weight;
    }
    if(bias_weight != NULL) {
        delete bias_weight;
    }
}

int InnerProduct::loadParam(std::vector<std::string> params, int offset) {
    output_c = string2int(params[offset]);
    return 0;
}

int InnerProduct::loadModel(std::ifstream& fp) {
    fp.read((char*)(multi_weight->getPtr()), multi_weight->size*sizeof(float));
    fp.read((char*)(bias_weight->getPtr()), bias_weight->size*sizeof(float));
    return 0;
}

std::vector<std::vector<int>> InnerProduct::inferShape(std::vector<std::vector<int>> bottom_shapes) {
    input_c = bottom_shapes[0][0]*bottom_shapes[0][1]*bottom_shapes[0][2];
    multi_weight = new Blob(output_c, input_c);
    bias_weight = new Blob(output_c);
    return std::vector<std::vector<int>>{std::vector<int>{1,1,output_c}};
}

int InnerProduct::forward(std::vector<Blob*>& bottoms, std::vector<Blob*>& tops) {
    Blob* bottom = bottoms[0];
    Blob* top = tops[0];

    float* bottom_data = bottom->getPtr();
    float* top_data = top->getPtr();
    float* multi_weight_data = multi_weight->getPtr();
    float* bias_weight_data = bias_weight->getPtr();

    for(int i = 0; i < top->size; i++) {
        float sum = 0;
        for(int j = 0; j < bottom->size; j++) {
            sum += bottom_data[j] * multi_weight_data[i*multi_weight->w+j];
        }
        top_data[i] = sum + bias_weight_data[i];
    }
    
    return 0;
}


}   // namespace fcnn