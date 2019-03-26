cmake_minimum_required(VERSION 2.8)
set(PROJECT_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${PROJECT_ROOT_PATH})
include_directories(${PROJECT_ROOT_PATH}/src)
add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(tools)
add_subdirectory(test)