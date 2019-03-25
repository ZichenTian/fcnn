#ifndef __LAYER_H
#define __LAYER_H

#include <fstream>
#include <sstream>

#include "blob.h"

namespace fcnn {

static inline int string2int(std::string& s) {
    int num;
    std::stringstream stream(s);
    stream >> num;
    return num;
}

class Layer {

public:
  // functions
  Layer(std::string name="");
  virtual ~Layer();
  virtual int loadParam(std::vector<std::string> params, int offset=0);
  virtual int loadModel(std::ifstream& fp);
  virtual std::vector<std::vector<int>> inferShape(std::vector<std::vector<int>> bottom_shapes);
  virtual int forward(std::vector<Blob*>& bottoms, std::vector<Blob*>& tops);
  // variables
  std::string name;
  std::string type;
  std::vector<int> bottoms;
  std::vector<int> tops;
};

Layer* createLayer(std::string type, std::string name="");

} // namespace fcnn

#endif //__LAYER_H