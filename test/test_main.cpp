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

void test_resnet_50(void) {
    fcnn::Net net;
    std::cout << "start to load param" << std::endl;
    net.loadParam("/home/tianzichen/Working/fcnn/build/tools/123.param");
    std::cout << "start to load model" << std::endl;
    net.loadModel("/home/tianzichen/Working/fcnn/build/tools/123.model");
    // net.printNetInfo();

    fcnn::Blob data_blob = net.getBlob("data");
    std::cout << "filling the data blob" << std::endl;
    if(fill_imgdata("./test.jpg", data_blob) == false) {
        std::cerr << "read image failed!" << std::endl;
        return;
    }
    std::cout << "start to forward" << std::endl;
    net.forward();
    std::cout << "run finished" << std::endl;

    fcnn::Blob result_blob = net.getBlob("prob");
    std::cout << argmax(result_blob) << std::endl;
    return;
}

void test_load_param(void) {
    fcnn::Net net;
    std::cout << "start to load param" << std::endl;
    net.loadParam("/home/tianzichen/Working/fcnn/build/tools/123.param");
    std::cout << "start to load model" << std::endl;
    net.loadModel("/home/tianzichen/Working/fcnn/build/tools/123.model");
    net.printNetInfo();
    
    fcnn::Blob data_blob = net.getBlob("data");
    fcnn::Blob conv1_blob = net.getBlob("conv1");
    fcnn::Blob pool1_blob = net.getBlob("pool1");

    float* input_data = new float[data_blob.size];
    for(int i = 0; i < data_blob.size; i++) {
        input_data[i] = i<data_blob.size/2?-1:1;
    }
    data_blob.fill(input_data, data_blob.size);

    net.forward();
    return;
    std::cout << data_blob.printData() << std::endl;
    std::cout << conv1_blob.printData() << std::endl;
    std::cout << pool1_blob.printData() << std::endl;
}

int main() {
    // test_load_param();
    test_resnet_50();
    return 0;
}