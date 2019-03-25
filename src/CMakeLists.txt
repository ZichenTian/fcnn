cmake_minimum_required(VERSION 2.8)
set(fcnn_SRCs blob.cpp layer.cpp net.cpp create_layer.cpp 
              layers/relu.cpp layers/input.cpp layers/maxpool.cpp layers/convolution.cpp 
              layers/global_avepool.cpp layers/inner_product.cpp layers/scale.cpp layers/element_wise.cpp 
              layers/softmax.cpp)
add_library(lib_src STATIC ${fcnn_SRCs})
set_target_properties(lib_src PROPERTIES OUTPUT_NAME "fcnn")
