#include <iostream>
#include <vector>

#include "src/blob.h"
#include "src/layer.h"
#include "src/net.h"

#include "opencv2/opencv.hpp"

static inline bool fill_imgdata(std::string filename, fcnn::Blob& dst) {
    cv::Mat srcImage = cv::imread(filename, -1);
    if(srcImage.empty() || srcImage.channels() != 3) {
        return false;
    }
    cv::resize(srcImage, srcImage, cv::Size(dst.w, dst.h));
    cv::imshow("srcImage", srcImage);
    cv::waitKey(0);
    cv::Mat BGR[3];
    uchar mean[3] = {104, 117, 123};
    cv::split(srcImage, BGR);
    for(int c = 0; c < 3; c++) {
        const uchar* src_data = BGR[c].data;
        float* dst_data = dst.getPtr() + c*dst.h*dst.w;
        for(int i = 0; i < dst.h*dst.w; i++) {
            *(dst_data++) = (float)(*(src_data++))-mean[c];
        }
    }
    return true;
}

static inline int argmax(fcnn::Blob& src) {
    float max_score = -FLT_MAX;
    int max_index = -1;
    const float* src_data = src.getPtr();
    int i = 0;
    for(; i < src.size; i++) {
        if(src_data[i] > max_score) {
            max_index = i;
            max_score = src_data[i];
        }
    }
    return max_index;
}

int main(int argc, char** argv) {
    if(argc != 4) {
        std::cerr << "Usage: ./classfication param model img" << std::endl;
        return 1;
    }
    const char* param_file = argv[1];
    const char* model_file = argv[2];
    const char* img_file = argv[3];

    fcnn::Net net;
    std::cout << "start to load param" << std::endl;
    net.loadParam(param_file);
    std::cout << "start to load model" << std::endl;
    net.loadModel(model_file);
    net.printNetInfo();

    fcnn::Blob data_blob = net.getBlob("data");
    std::cout << "filling the data blob" << std::endl;
    if(fill_imgdata(img_file, data_blob) == false) {
        std::cerr << "read image failed!" << std::endl;
        return 1;
    }
    std::cout << "start to forward" << std::endl;
    net.forward();
    std::cout << "run finished" << std::endl;

    fcnn::Blob result_blob = net.getBlob("prob");
    std::cout << argmax(result_blob) << std::endl;
    return 0;
}