cmake_minimum_required(VERSION 2.8)
find_package(OpenCV REQUIRED)
add_executable(classfication classification.cpp)
target_link_libraries(classfication lib_src ${OpenCV_LIBS})