#include "scale.h"

namespace fcnn {

Scale::Scale(std::string name_) : Layer(name_), scale(NULL), bias(NULL) { type = "Scale"; }

Scale::~Scale() {
    if(scale != NULL) {
        delete scale;
    }
    if(bias != NULL) {
        delete bias;
    }
}

int Scale::loadModel(std::ifstream& fp) {
    scale = new Blob(channels);
    bias = new Blob(channels);
    fp.read((char*)scale->getPtr(), scale->size*sizeof(float));
    fp.read((char*)bias->getPtr(), bias->size*sizeof(float));
}

int Scale::loadParam(std::vector<std::string> params, int offset) {
    return 0;
}

std::vector<std::vector<int>> Scale::inferShape(std::vector<std::vector<int>> bottom_shapes) {
    channels = bottom_shapes[0][2];
    return std::vector<std::vector<int>> {bottom_shapes[0]};
}

int Scale::forward(std::vector<Blob*>& bottoms, std::vector<Blob*>& tops) {
    Blob* bottom = bottoms[0];
    Blob* top = tops[0];

    float* bottom_data = bottom->getPtr();
    float* top_data = top->getPtr();
    float* scale_data = scale->getPtr();
    float* bias_data = bias->getPtr();

    for(int c = 0; c < bottom->c; c++) {
        for(int h = 0; h < bottom->h; h++) {
            for(int w = 0; w < bottom->w; w++) {
                *(top_data++) = *(bottom_data++)*scale_data[c] + bias_data[c];
            }
        }
    }

    return 0;
}

}   // namespace fcnn