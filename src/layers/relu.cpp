#include "relu.h"

namespace fcnn {

ReLU::ReLU(std::string name_) : Layer(name_) { type = "ReLU"; }

ReLU::~ReLU() {}


int ReLU::loadParam(std::vector<std::string> params, int offset) {
    return 0;
}

std::vector<std::vector<int>> ReLU::inferShape(std::vector<std::vector<int>> bottom_shapes) {
    std::vector<std::vector<int>> top_shapes;
    for(int i = 0; i < bottom_shapes.size(); i++) { // same size
        top_shapes.push_back(bottom_shapes[i]);
    }
    return top_shapes;
}

int ReLU::loadModel(std::ifstream& fp) {
    return 0;
}

int ReLU::forward(std::vector<Blob*>& bottoms, std::vector<Blob*>& tops) {
    Blob* bottom_blob = bottoms[0];
    Blob* top_blob = tops[0];
    float* bottom_data = bottom_blob->data.get();
    float* top_data = top_blob->data.get();
#ifdef __arch64__
    // TODO: Use Neon

#else
    for(int i = 0; i < bottom_blob->size; i++) {
        *(top_data++) = std::max(*(bottom_data++), (float)0.0f);
    }
#endif
    return 0;
}


}   // namespace fcnn