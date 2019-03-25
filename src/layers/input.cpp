#include "input.h"

namespace fcnn {

Input::Input(std::string name_) : Layer(name_) { type = "Input"; }

Input::~Input() {}


int Input::loadParam(std::vector<std::string> params, int offset) {
    input_shape.push_back(string2int(params[offset]));
    input_shape.push_back(string2int(params[offset+1]));
    input_shape.push_back(string2int(params[offset+2]));
    return 0;
}

std::vector<std::vector<int>> Input::inferShape(std::vector<std::vector<int>> bottom_shapes) {
    std::vector<std::vector<int>> top_shapes {input_shape};
    return top_shapes;
}

int Input::loadModel(std::ifstream& fp) {
    return 0;
}

int Input::forward(std::vector<Blob*>& bottoms, std::vector<Blob*>& tops) {
    return 0;
}

}   // namespace fcnn