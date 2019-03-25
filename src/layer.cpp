#include "layer.h"

namespace fcnn {

Layer::Layer(std::string name_) : name(name_), type("Abs") {}

Layer::~Layer() {}

int Layer::loadParam(std::vector<std::string> params, int offset) {
    NOT_IMPLEMENT
}

int Layer::loadModel(std::ifstream& fp) {
    NOT_IMPLEMENT
}

std::vector<std::vector<int>> Layer::inferShape(std::vector<std::vector<int>> bottom_shapes) {
    NOT_IMPLEMENT
}

int Layer::forward(std::vector<Blob*>& bottoms, std::vector<Blob*>& tops) {
    NOT_IMPLEMENT
}

}   // namespace fcnn