#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>

#include "test_common.h"

int main(void) {
    using namespace fcnn;
    using namespace std;
    using namespace std::chrono;

    vector<string> params{"MaxPool", "test_maxpool", "1", "1", "input", "output",  // base param
                            "kernel", "stride", "pad"};
    for(int i = 0; i < 30; i++) {
        Layer* layer = createLayer("MaxPool");
        int64_t seed = high_resolution_clock::now().time_since_epoch().count();
        srand(seed);
        int h = rand_num(1, 64);
        int w = rand_num(1, 64);
        int c = rand_num(1, 16);
        int kernel = rand_num(2, 3);
        int stride = rand_num(1, 2);
        int pad = rand_num(0, 1);
        if(h + 2*pad < kernel)  h = kernel - 2*pad;
        if(w + 2*pad < kernel)  w = kernel - 2*pad; 
        fprintf(stderr, "input_h=%d, input_w=%d, input_c=%d, kernel=%d, stride=%d, pad=%d, ", 
                            h, w, c, kernel, stride, pad);

        params[6] = int2string(kernel);
        params[7] = int2string(stride);
        params[8] = int2string(pad);
        layer->loadParam(params, 6);

        vector<int> input_shape {h, w, c};
        vector<int> output_shape = layer->inferShape(vector<vector<int>>{input_shape})[0];
        Blob input_blob(input_shape);
        Blob output_blob(output_shape);
        fill_blob_random(input_blob);
        vector<Blob*> bottoms{&input_blob};
        vector<Blob*> tops{&output_blob};

        int Ops = output_blob.h * output_blob.w * output_blob.c * kernel * kernel;
        int cnt = 1000;

        chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
        for(int i = 0; i < cnt; i++) {
            layer->forward(bottoms, tops);
        }
        chrono::high_resolution_clock::time_point end = chrono::high_resolution_clock::now();
        double forward_time = chrono::duration_cast<chrono::microseconds>(end-start).count();
        double FLOPs = Ops*cnt/forward_time;

        fprintf(stderr, "output_h=%d, output_w=%d, OPs=%.8lfMFLOPs, time=%.8lfms, FLOPs=%.8lfMFLOPs\n", 
                            output_blob.h, output_blob.w, cnt*Ops/1000000.0, forward_time/1000, FLOPs);
        delete layer;
    }

    return 0;
}