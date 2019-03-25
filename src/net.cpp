#include <string.h>
#include <iostream>
#include "net.h"
#include "layers/relu.h"

namespace fcnn {

Net::Net() {}

Net::~Net() {
    clear();
}

int Net::loadParam(std::string filename) {
    std::ifstream fp;
    fp.open(filename, std::ios::in);
    if (!fp) {
        fprintf(stderr, "param file open failed!\n");
        exit(1);
    }
    return loadParam(fp);
}

int Net::loadModel(std::string filename) {
    std::ifstream fp;
    fp.open(filename, std::ios::in | std::ios::binary);
    if (!fp) {
        fprintf(stderr, "model file open failed!\n");
        exit(1);
    }
    return loadModel(fp);
}

static inline std::vector<std::string> split_string(char* s, const char delim) {
    char c = s[0];
    int start = 0;
    int i = 0;
    std::vector<std::string> result;
    while(c) {
        if (c == delim) {
            s[i] = 0;   // replace delim as '\0'
            std::string tmp = s+start;
            if (tmp != "") {
                result.push_back(s+start);
            }
            start = i+1;
        }
        i++;
        c = s[i];
    }
    std::string tmp = s+start;
    if (tmp != "") {
        result.push_back(s+start);
    }
    return result;
}

int Net::loadParam(std::ifstream& fp) {
    char* readBuf = new char[257];
    if (!fp.getline(readBuf, 257)) { // first line read wrong
        fprintf(stderr, "cannot read first line of param!\n");
        exit(1);
    }
    std::vector<std::string> params = split_string(readBuf, ' ');
    if (params.size() != 2) {
        fprintf(stderr, "first line not two number!\n");
        exit(1);
    }
    int layer_num = string2int(params[0]);
    int blob_num = string2int(params[1]);
    for (int i = 0; i < layer_num; i++) {
        if (!fp.getline(readBuf, 257)) {
            fprintf(stderr, "not enough layer in param file\n");
            exit(1);
        }
        // get base params
        fprintf(stderr, "%s\n", readBuf);
        std::vector<std::string> params = split_string(readBuf, ' ');
        std::string type = params[0];
        std::string name = params[1];
        int bottom_num = string2int(params[2]);
        int top_num = string2int(params[3]);
        std::vector<std::string> bottom_names, top_names;
        int index = 4;
        for(int i = 0; i < bottom_num; i++) {
            bottom_names.push_back(params[index++]);
        }
        for(int i = 0; i < top_num; i++) {
            top_names.push_back(params[index++]);
        }
        // create layer and load layer-specific params
        Layer* layer = createLayer(type, name);
        layer->loadParam(params, index);
        // process bottom blobs
        std::vector<std::vector<int>> bottom_shapes;
        for(int i = 0; i < bottom_num; i++) {
            int index = getBlobIndex(bottom_names[i]);
            if(index == -1) {
                INVALID_ERROR("bottom must be found!\n");
            } else {
                layer->bottoms.push_back(index);
                bottom_shapes.push_back(blobs[index]->getShape());
            }
        }
        // get top blobs' shape
        std::vector<std::vector<int>> top_shapes = layer->inferShape(bottom_shapes);
        // process top blobs
        for(int i = 0; i < top_num; i++) {
            int index = getBlobIndex(top_names[i]);
            if(index == -1) {
                index = blobs.size();
                Blob* blob = new Blob(top_shapes[i], top_names[i]);
                blobs.push_back(blob);
                layer->tops.push_back(index);
            } else {
                layer->tops.push_back(index);
            }
        }
        layers.push_back(layer);
    }
    delete[] readBuf;
}

int Net::loadModel(std::ifstream& fp) {
    char layer_name_buffer[128];
    for(int i = 0; i < layers.size(); i++) {
        Layer* layer = layers[i];
        fp.read(layer_name_buffer, 128);
        if(std::string(layer_name_buffer) != layer->name) {
            INVALID_ERROR("not found layer name in modelbin!\n");
        }
        layer->loadModel(fp);
    }
}

void Net::printNetInfo(void) {
    using std::cout;
    using std::endl;
    cout << "\n##################\n" << "NetInfo" << endl;
    cout << "Layer_num: " << layers.size() << endl;
    cout << "Blob_num: " << blobs.size() << endl;
    for(int i = 0; i < layers.size(); i++) {
        Layer* layer = layers[i];
        cout << "######################" << endl;
        cout << "id: " << i << " layer_name: " << layer->name  << " layer_type: " << layer->type << endl;
        cout << "bottoms:" <<endl;
        for(int i = 0; i < layer->bottoms.size(); i++) {
            Blob* blob = blobs[layer->bottoms[i]];
            cout << "blob_name: " << blob->name << " blob_shape(h,w,c):" << blob->printShape() << endl;
        }
        cout << "tops:" << endl;
        for(int i = 0; i < layer->tops.size(); i++) {
            Blob* blob = blobs[layer->tops[i]];
            cout << "blob_name: " << blob->name << " blob_shape(h,w,c):" << blob->printShape() << endl;
        }
    }
}


int Net::forward() {
    for(int i = 0; i < layers.size(); i++) {
        Layer* layer = layers[i];
        std::vector<Blob*> bottoms;
        std::vector<Blob*> tops;
        for(int i = 0; i < layer->bottoms.size(); i++) {
            int blob_index = layer->bottoms[i];
            bottoms.push_back(blobs[blob_index]);
        }
        for(int i = 0; i < layer->tops.size(); i++) {
            int blob_index = layer->tops[i];
            tops.push_back(blobs[blob_index]);
        }
        layer->forward(bottoms, tops);
    }
    return 0;
}

void Net::clear() {

}

Blob& Net::getBlob(std::string name) {
    for(int i = 0; i < blobs.size(); i++) {
        if(blobs[i]->name == name) {
            return *blobs[i];
        }
    }
    fprintf(stderr, "blob not found!\n");
    exit(1);
}

Blob& Net::getBlob(int index) {
    if(index >= blobs.size() || index < 0) {
        fprintf(stderr, "index exceeds blobs\n");
        exit(1);
    }
    return *blobs[index];
}

int Net::getBlobIndex(std::string name) {
    for(int i = 0; i < blobs.size(); i++) {
        if(blobs[i]->name == name) {
            return i;
        }
    }
    return -1;
}

}   // namespace fcnn



