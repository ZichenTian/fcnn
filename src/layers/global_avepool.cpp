#include "global_avepool.h"

namespace fcnn {

GlobalAvePool::GlobalAvePool(std::string name_) : Layer(name_) { type = "GlobalAvePool"; }

GlobalAvePool::~GlobalAvePool() {}

int GlobalAvePool::loadParam(std::vector<std::string> params, int offset) {
    return 0;
}

int GlobalAvePool::loadModel(std::ifstream& fp) {
    return 0;
}

std::vector<std::vector<int>> GlobalAvePool::inferShape(std::vector<std::vector<int>> bottom_shapes) {
    int channels = bottom_shapes[0][2];
    std::vector<std::vector<int>> top_shapes {std::vector<int> {1,1,channels}};
    return top_shapes;
}

int GlobalAvePool::forward(std::vector<Blob*>& bottoms, std::vector<Blob*>& tops) {
    Blob* bottom = bottoms[0];
    Blob* top = tops[0];

    float* bottom_data = bottom->getPtr();
    float* top_data = top->getPtr();

    for(int c = 0; c < bottom->c; c++) {
        float sum = 0;
        for(int h = 0; h < bottom->h; h++) {
            for(int w = 0; w < bottom->w; w++) {
                sum += *(bottom_data++);
            }
        }
        *(top_data++) = sum/(bottom->h*bottom->w);
    }
    return 0;
}


}   // namespace fcnn