#include <iostream>
#include <fstream>
#include <string>
#include <math.h>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>

#include "caffe.pb.h"

#define NOT_IMPLEMENT  { fprintf(stderr, "NOT IMPLEMENT!\n"); exit(1); }
#define INVALID_ERROR(str) { fprintf(stderr, "%s", (str)); exit(1); } 

static bool read_proto_from_text(const char* filepath, google::protobuf::Message* message)
{
    std::ifstream fs(filepath, std::ifstream::in);
    if (!fs.is_open())
    {
        fprintf(stderr, "open failed %s\n", filepath);
        return false;
    }

    google::protobuf::io::IstreamInputStream input(&fs);
    bool success = google::protobuf::TextFormat::Parse(&input, message);

    fs.close();

    return success;
}

static bool read_proto_from_binary(const char* filepath, google::protobuf::Message* message)
{
    std::ifstream fs(filepath, std::ifstream::in | std::ifstream::binary);
    if (!fs.is_open())
    {
        fprintf(stderr, "open failed %s\n", filepath);
        return false;
    }

    google::protobuf::io::IstreamInputStream input(&fs);
    google::protobuf::io::CodedInputStream codedstr(&input);

    codedstr.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);

    bool success = message->ParseFromCodedStream(&codedstr);

    fs.close();

    return success;
}

static inline int find_layer_index_by_name(const caffe::NetParameter& net, const std::string& layer_name) {
    int i = 0;
    for(; i < net.layer_size(); i++) {
        if(net.layer(i).name() == layer_name) {
            break;
        }
    }
    if(i >= net.layer_size()) {
        std::cerr << layer_name << std::endl;
        INVALID_ERROR("prototxt has layer that caffemodel not contain!\n")
    }
    return i;
}

int main(int argc, char** argv) {
    if(argc != 5) {
        fprintf(stderr, "Usage: %s [caffeproto] [caffemodel] [fcnnparam] [fcnnmodel]\n", argv[0]);
        exit(1);
    }

    const char* caffe_proto = argv[1];
    const char* caffe_model = argv[2];
    const char* fcnn_param = argv[3];
    const char* fcnn_model = argv[4];

    caffe::NetParameter proto;
    caffe::NetParameter net;

    bool s0 = read_proto_from_text(caffe_proto, &proto);
    if(!s0) {
        fprintf(stderr, "read caffe proto failed!\n");
        exit(1);
    }
    bool s1 = read_proto_from_binary(caffe_model, &net);
    if(!s1) {
        fprintf(stderr, "read caffe model failed!\n");
        exit(1);
    }

    std::ofstream pp, bp;
    pp.open(fcnn_param, std::ios::out);
    bp.open(fcnn_model, std::ios::out | std::ios::binary);

    int layer_count = proto.layer_size();
    int blob_count = -1;    // not needed now!
    pp << layer_count << " " << blob_count << std::endl;

    // for(int i = 0; i < net.layer_size(); i++) {
    //     const caffe::LayerParameter& layer = net.layer(i);
    //     std::cerr << layer.name() << " : " << layer.type() << std::endl;
    //     std::cerr << "has blobs: " << layer.blobs_size() << std::endl;
    //     for(int j = 0; j < layer.blobs_size(); j++) {
    //         std::cerr << layer.blobs(j).data_size() << " ";
    //     }
    //     std::cerr << std::endl << std::endl;
    //     if(layer.type() == "BatchNorm") {
    //         std::cerr << *(layer.blobs(2).data().data()) << std::endl;
    //     }
    //     if(layer.type() == "Convolution") {
    //         std::cerr << layer.convolution_param().bias_term() << std::endl;
    //     }
    // }
    // return 0;

    char layer_name_buffer[128];


    for(int i = 0; i < layer_count; i++) {

        const caffe::LayerParameter& layer = proto.layer(i);
        std::cerr << "Processing layer: " << layer.name() << std::endl;

        sprintf(layer_name_buffer, "%s", layer.name().c_str());
        std::cerr << "layer_name_buffer: " << layer_name_buffer << std::endl;
        bp.write(layer_name_buffer, 128);

        if(layer.type() == "Convolution") {
            const caffe::ConvolutionParameter& convolution_param = layer.convolution_param();
            if(convolution_param.group() == 1) {
                // normal convolution
                pp << "Convolution" << " "      // layer_type
                    << layer.name() << " "      // layer_name
                    << 1 << " "                 // bottom_size = 1
                    << 1 << " "                 // top_size = 1
                    << layer.bottom(0) << " "   // bottom_name
                    << layer.top(0) << " "      // top_name
                    << convolution_param.kernel_size(0) << " "   // kernel
                    << (convolution_param.stride_size()>0?convolution_param.stride(0):1) << " "  // stride default = 1
                    << convolution_param.num_output() << " "    // output_channels
                    << (convolution_param.pad_size()>0?convolution_param.pad(0):0) << " "    // pad default = 0
                    << convolution_param.bias_term() << std::endl;  // bias_term   

                int index_in_net = find_layer_index_by_name(net, layer.name());
                const caffe::LayerParameter& binlayer = net.layer(index_in_net);
                const caffe::BlobProto& conv_weight = binlayer.blobs(0);
                bp.write((const char*)conv_weight.data().data(), conv_weight.data_size()*sizeof(float));
                if(convolution_param.bias_term()) {
                    const caffe::BlobProto& bias_weight = binlayer.blobs(1);
                    bp.write((const char*)bias_weight.data().data(), bias_weight.data_size()*sizeof(float));
                }
            } else if(convolution_param.group() == convolution_param.num_output()) {
                // depthwise convolution
                NOT_IMPLEMENT
            } else {
                // group convolution
                NOT_IMPLEMENT
            }
        } else if(layer.type() == "Input") {
            const caffe::InputParameter& input_param = layer.input_param();
            const caffe::BlobShape& bs = input_param.shape(0);
            // Input layer_name bottom_size(0) top_size(1) top_name
            pp << "Input" << " " << layer.name() << " 0 1 " << layer.top(0);
            if(bs.dim_size() == 4) {
                // H W C   caffe is NCHW(0123)
                pp << " " << bs.dim(2) << " " << bs.dim(3) << " " << bs.dim(1) << std::endl;
            } else {
                NOT_IMPLEMENT
            }
        } else if(layer.type() == "ReLU") {
            // ReLU layer_name bottom_size(1) top_size(1) bottom_name top_name
            pp << "ReLU" << " " << layer.name() << " 1 1 " << layer.bottom(0) << " " << layer.top(0) << std::endl;
        } else if(layer.type() == "Pooling") {
            const caffe::PoolingParameter& pooling_param = layer.pooling_param();
            if(pooling_param.pool() == caffe::PoolingParameter_PoolMethod_MAX) {
                // max_pooling
                // MaxPool layer_name bottom_size(1) top_size(1) bottom_name top_name kernel stride pad
                pp << "MaxPool" << " "          // layer_type
                    << layer.name() << " "      // layer_name
                    << 1 << " "                 // bottom_size = 1
                    << 1 << " "                 // top_size = 1
                    << layer.bottom(0) << " "   // bottom_name
                    << layer.top(0) << " "      // top_name
                    << pooling_param.kernel_size() << " "   // kernel
                    << pooling_param.stride() << " "        // stride
                    << pooling_param.pad() << std::endl;    // pad
            } else if (pooling_param.pool() == caffe::PoolingParameter_PoolMethod_AVE) {
                // ave_pooling
                pp << "GlobalAvePool" << " "    // layer_type
                    << layer.name() << " "      // layer_name
                    << 1 << " "                 // bottom_size = 1
                    << 1 << " "                 // top_size = 1
                    << layer.bottom(0) << " "   // bottom_name
                    << layer.top(0) << " " << std::endl;     // top_name
            } else {
                // stochastic_pooling
                NOT_IMPLEMENT
            }
        } else if(layer.type() == "InnerProduct") {
            const caffe::InnerProductParameter& inner_product_param = layer.inner_product_param();
            pp << "InnerProduct" << " "     // layer_type
                << layer.name() << " "      // layer_name
                << 1 << " "                 // bottom_size = 1
                << 1 << " "                 // top_size = 1
                << layer.bottom(0) << " "   // bottom_name
                << layer.top(0) << " "      // top_name
                << inner_product_param.num_output() << std::endl;   // num_output
            int index_in_net = find_layer_index_by_name(net, layer.name());
            const caffe::LayerParameter& binlayer = net.layer(index_in_net);
            const caffe::BlobProto& multi_weight = binlayer.blobs(0);
            const caffe::BlobProto& bias_weight = binlayer.blobs(1);
            bp.write((char*)multi_weight.data().data(), multi_weight.data_size()*sizeof(float));
            bp.write((char*)bias_weight.data().data(), bias_weight.data_size()*sizeof(float));
        } else if(layer.type() == "Scale") {
            pp << "Scale" << " "          // layer_type
                << layer.name() << " "      // layer_name
                << 1 << " "                 // bottom_size = 1
                << 1 << " "                 // top_size = 1
                << layer.bottom(0) << " "   // bottom_name
                << layer.top(0) << std::endl;      // top_name
            int index_in_net = find_layer_index_by_name(net, layer.name());
            const caffe::LayerParameter& binlayer = net.layer(index_in_net);
            const caffe::BlobProto& scale_weight = binlayer.blobs(0);
            const caffe::BlobProto& bias_weight = binlayer.blobs(1);
            bp.write((char*)scale_weight.data().data(), scale_weight.data_size()*sizeof(float));
            bp.write((char*)bias_weight.data().data(), bias_weight.data_size()*sizeof(float));
        } else if(layer.type() == "BatchNorm") {
            pp << "Scale" << " "          // layer_type
                << layer.name() << " "      // layer_name
                << 1 << " "                 // bottom_size = 1
                << 1 << " "                 // top_size = 1
                << layer.bottom(0) << " "   // bottom_name
                << layer.top(0) << std::endl;      // top_name
            int index_in_net = find_layer_index_by_name(net, layer.name());
            const caffe::LayerParameter& binlayer = net.layer(index_in_net);
            const caffe::BlobProto& mean_weight = binlayer.blobs(0);
            const caffe::BlobProto& variance_weight = binlayer.blobs(1);
            const float* mean_data = mean_weight.data().data();
            const float* varaince_data = variance_weight.data().data();
            float scale_weight = 1.0 / (*(binlayer.blobs(2).data().data()));    // always has one element

            int data_size = mean_weight.data_size();
            float* mutable_mean_data = new float[data_size];
            float* mutable_varaince_data = new float[data_size];

            for(int i = 0; i < data_size; i++) {
                mutable_mean_data[i] = scale_weight * mean_data[i];
                mutable_varaince_data[i] = sqrt(scale_weight * varaince_data[i] + 1e-5);
            }

            float* scale_data = new float[data_size];
            float* bias_data = new float[data_size];

            for(int i = 0; i < data_size; i++) {
                scale_data[i] = 1.0 / mutable_varaince_data[i];
                bias_data[i] = - 1.0 * mutable_mean_data[i] / mutable_varaince_data[i];
            }
            bp.write((char*)scale_data, data_size*sizeof(float));
            bp.write((char*)bias_data, data_size*sizeof(float));
            delete[] mutable_mean_data;
            delete[] mutable_varaince_data;
            delete[] scale_data;
            delete[] bias_data;
        } else if(layer.type() == "Eltwise") {
            pp << "Eltwise" << " "          // layer_type
                << layer.name() << " "      // layer_name
                << layer.bottom_size() << " "   // bottom_size
                << 1 << " ";                // top_size = 1
            for(int j = 0; j < layer.bottom_size(); j++) {
                pp << layer.bottom(j) << " ";   // bottom_name
            }
            pp << layer.top(0) << " ";      // top_name
            switch(layer.eltwise_param().operation()) {
                case caffe::EltwiseParameter_EltwiseOp_PROD: NOT_IMPLEMENT
                case caffe::EltwiseParameter_EltwiseOp_SUM: pp << 1 << std::endl; break;
                case caffe::EltwiseParameter_EltwiseOp_MAX: NOT_IMPLEMENT
                default: INVALID_ERROR("invalid eltwise type!\n");
            }
        } else if(layer.type() == "Softmax") {
            pp << "Softmax" << " "          // layer_type
                << layer.name() << " "      // layer_name
                << 1 << " "                 // bottom_size = 1
                << 1 << " "                 // top_size = 1
                << layer.bottom(0) << " "   // bottom_name
                << layer.top(0) << std::endl;      // top_name
        } else {
            fprintf(stderr, "layer_type %s ", layer.type().c_str());
            NOT_IMPLEMENT
        }

        

    }

    pp.close();
    bp.close();



    return 0;
}

