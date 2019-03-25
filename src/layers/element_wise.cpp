#include "element_wise.h"
#include <iostream>

namespace fcnn {

Eltwise::Eltwise(std::string name_) : Layer(name_) { type = "Eltwise"; }

Eltwise::~Eltwise() {}


int Eltwise::loadParam(std::vector<std::string> params, int offset) {
    switch(string2int(params[offset])) {
        case 1: operation = SUM; break;
        default: INVALID_ERROR("Unknown Eltwise Type!\n");
    }
}

template <typename T>
static inline bool vector_is_same(std::vector<T>& a, std::vector<T>& b) {
    if(a.size() != b.size()) {
        return false;
    }
    for(int i = 0; i < a.size(); i++) {
        if(a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

std::vector<std::vector<int>> Eltwise::inferShape(std::vector<std::vector<int>> bottom_shapes) {
    if(bottom_shapes.size() <= 1) {
        INVALID_ERROR("Eltwise must has more than two input blobs!\n");
    }
    for(int i = 0; i < bottom_shapes.size()-1; i++) {
        // std::cout << bottom_shapes[i][0] << " " << bottom_shapes[i][1] << " " << bottom_shapes[i][2] << std::endl;
        // std::cout << bottom_shapes[i+1][0] << " " << bottom_shapes[i+1][1] << " " << bottom_shapes[i+1][2] << std::endl;
        if(vector_is_same<int>(bottom_shapes[i], bottom_shapes[i+1]) == false) {
            INVALID_ERROR("Eltwise input blob shape must be the same!\n");
        }
    }
    return std::vector<std::vector<int>> {bottom_shapes[0]};
}

int Eltwise::loadModel(std::ifstream& fp) {
    return 0;
}

int Eltwise::forward(std::vector<Blob*>& bottoms, std::vector<Blob*>& tops) {
    Blob* top = tops[0];
    float* bottom_data = bottoms[0]->getPtr();
    float* top_data = top->getPtr();
    for(int i = 0; i < top->size; i++) {
        top_data[i] = bottom_data[i];
    }
    if(operation == SUM) {
        for(int index = 1; index < bottoms.size(); index++) {
            float* bottom_data = bottoms[index]->getPtr();
            for(int i = 0; i < top->size; i++) {
                top_data[i] += bottom_data[i];
            }
        }
    } else {
        NOT_IMPLEMENT
    }
    return 0;
}


}   // namespace fcnn