cmake_minimum_required(VERSION 2.8)
find_package(OpenCV REQUIRED)
add_executable(fcnn test_main.cpp)
target_link_libraries(fcnn lib_src ${OpenCV_LIBS})