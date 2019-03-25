#include <memory.h>
#include "blob.h"

namespace fcnn {

Blob::Blob(std::string name_) : name(name_), dims(0), h(0), w(0), c(0), size(0), data(NULL), producer(-1) {
}

Blob::Blob(int h_, std::string name_) : name(name_), dims(1), h(h_), w(1), c(1), size(h_), producer(-1) {
    data = std::shared_ptr<float>(new float[size], std::default_delete<float[]>());
    if (!data) {
        fprintf(stderr, "New %ld Data Failed!\n", size);
        exit(1);
    }
}

Blob::Blob(int h_, int w_, std::string name_) : name(name_), dims(2), h(h_), w(w_), c(1), size(h_*w_), producer(-1) {
    data = std::shared_ptr<float>(new float[size], std::default_delete<float[]>());
    if (!data) {
        fprintf(stderr, "New %ld Data Failed!\n", size);
        exit(1);
    }
}

Blob::Blob(int h_, int w_, int c_, std::string name_) : name(name_), dims(3), h(h_), w(w_), c(c_), size(h_*w_*c_), producer(-1) {
    data = std::shared_ptr<float>(new float[size], std::default_delete<float[]>());
    if (!data) {
        fprintf(stderr, "New %ld Data Failed!\n", size);
        exit(1);
    }
}

Blob::Blob(std::vector<int> shape, std::string name_) : name(name_), dims(shape.size()), producer(-1) {
    switch(shape.size()) {
        case 1: h = shape[0]; w = 1; c = 1; size = h; break;
        case 2: h = shape[0]; w = shape[1]; c = 1; size = h*w; break;
        case 3: h = shape[0]; w = shape[1]; c = shape[2]; size = h*w*c; break;
        default: exit(1);
    }
    data = std::shared_ptr<float>(new float[size], std::default_delete<float[]>());
    if (!data) {
        fprintf(stderr, "New %ld Data Failed!\n", size);
        exit(1);
    }
}

void Blob::reshape(int h_) {
    NOT_IMPLEMENT
}

void Blob::reshape(int h_, int w_) {
    NOT_IMPLEMENT
}

void Blob::reshape(int h_, int w_, int c_) {
    NOT_IMPLEMENT
}

std::vector<int> Blob::getShape(void) {
    switch(dims) {
        case 1: { std::vector<int> result {w}; return result; }
        case 2: { std::vector<int> result {w, h}; return result; }
        case 3: { std::vector<int> result {w, h, c}; return result; }
        default: { std::vector<int> result; return result; }
    }
}

std::string Blob::printShape(void) {
    static char str[256];
    switch(dims) {
        case 0: sprintf(str, "()"); break;
        case 1: sprintf(str, "(%d)", h); break;
        case 2: sprintf(str, "(%d, %d)", h, w); break;
        case 3: sprintf(str, "(%d, %d, %d)", h, w, c); break;
        default: INVALID_ERROR("dims should be 0-3!\n"); break;
    }
    return std::string(str);
}

Blob& Blob::copy(void) {
    Blob* new_blob_ptr = new Blob(); // create empty blob
    *new_blob_ptr = *this;  // shallow copy
    new_blob_ptr->data = std::shared_ptr<float>(new float[size], std::default_delete<float[]>());
    if (!new_blob_ptr->data) {
        fprintf(stderr, "New %ld Data Failed", size);
        exit(1);
    }
    memcpy(new_blob_ptr->data.get(), data.get(), sizeof(float)*size);
    return *new_blob_ptr;
}

void Blob::fill(float* src, size_t len) {
    memcpy(data.get(), src, sizeof(float)*std::min(len, size));
}

void Blob::setValue(float value) {
    float* ptr = data.get();
    for(int i = 0; i < size; i++) {
        *(ptr++) = value;
    }
}

float* Blob::getPtr(void) {
    return data.get();
}

std::string Blob::printData(void) {
    size_t index = 0;
    std::string result;
    char tmp[100];
    const float* dataPtr = getPtr();
    for(int i = 0; i < c; i++) {
        result += "[\n";
        for(int j = 0; j < h; j++) {
            for(int k = 0; k < w; k++) {
                sprintf(tmp, "%.8f, ", dataPtr[index]);
                result += tmp;
                index++;
            }
            result += "\n";
        }
        result += "]\n";
    }
    return result;
}

Blob::~Blob() {}


}   // namespace fcnn