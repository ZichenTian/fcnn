#ifndef __RELU_H
#define __RELU_H

#include <fstream>
#include <sstream>

#include "layer.h"

namespace fcnn {

class ReLU : public Layer {

public:
  // functions
  ReLU(std::string name="");
  virtual ~ReLU();
  virtual int loadParam(std::vector<std::string> params, int offset=0);
  virtual int loadModel(std::ifstream& fp);
  virtual std::vector<std::vector<int>> inferShape(std::vector<std::vector<int>> bottom_shapes);
  virtual int forward(std::vector<Blob*>& bottoms, std::vector<Blob*>& tops);
  // variable
};


} // namespace fcnn

#endif //__RELU_H