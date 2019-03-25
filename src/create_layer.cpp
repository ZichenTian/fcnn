#include "layer.h"
#include "layers/relu.h"
#include "layers/input.h"
#include "layers/maxpool.h"
#include "layers/convolution.h"
#include "layers/global_avepool.h"
#include "layers/inner_product.h"
#include "layers/scale.h"
#include "layers/element_wise.h"
#include "layers/softmax.h"

namespace fcnn {

Layer* createLayer(std::string type, std::string name) {
    if(type == "Input") {
        return new Input(name);
    }
    if(type == "ReLU") {
        return new ReLU(name);
    }
    if(type == "MaxPool") {
        return new MaxPool(name);
    }
    if(type == "Convolution") {
        return new Convolution(name);
    }
    if(type == "GlobalAvePool") {
        return new GlobalAvePool(name);
    }
    if(type == "InnerProduct") {
        return new InnerProduct(name);
    }
    if(type == "Scale") {
        return new Scale(name);
    }
    if(type == "Eltwise") {
        return new Eltwise(name);
    }
    if(type == "Softmax") {
        return new Softmax(name);
    }
    return NULL;
}




}   // namespace fcnn