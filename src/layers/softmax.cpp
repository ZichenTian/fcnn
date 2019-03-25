#include "softmax.h"
#include <math.h>

namespace fcnn {

Softmax::Softmax(std::string name_) : Layer(name_), exps(NULL) { type = "Softmax"; }

Softmax::~Softmax() {
    if(exps != NULL) {
        delete exps;
    }
}


int Softmax::loadParam(std::vector<std::string> params, int offset) {
    return 0;
}

std::vector<std::vector<int>> Softmax::inferShape(std::vector<std::vector<int>> bottom_shapes) {
    exps = new Blob(1, 1, bottom_shapes[0][2]);
    return std::vector<std::vector<int>> {bottom_shapes[0]};
}

int Softmax::loadModel(std::ifstream& fp) {
    return 0;
}

int Softmax::forward(std::vector<Blob*>& bottoms, std::vector<Blob*>& tops) {
    Blob* bottom_blob = bottoms[0];
    Blob* top_blob = tops[0];
    float* bottom_data = bottom_blob->getPtr();
    float* top_data = top_blob->getPtr();
    float* exps_data = exps->getPtr();

    double sum = 0;
    for(int i = 0; i < bottom_blob->size; i++) {
        exps_data[i] = exp(bottom_data[i]);
        sum += exps_data[i];
    }
    for(int i = 0; i < top_blob->size; i++) {
        top_data[i] = exps_data[i] / sum;
    }
    return 0;
}


}   // namespace fcnn