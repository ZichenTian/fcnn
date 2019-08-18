cmake_minimum_required(VERSION 2.8)
file(GLOB fcnn_SRCs *.cpp layers/*.cpp)
add_library(lib_src STATIC ${fcnn_SRCs})
set_target_properties(lib_src PROPERTIES OUTPUT_NAME "fcnn")

# add install commands
install(TARGETS lib_src
        ARCHIVE DESTINATION ${CMAKE_BINARY_DIR}/install/lib)
install(FILES blob.h layer.h net.h
        DESTINATION ${CMAKE_BINARY_DIR}/install/include)
