#ifndef __SOFTMAX_H
#define __SOFTMAX_H

#include <fstream>
#include <sstream>

#include "layer.h"

namespace fcnn {

class Softmax : public Layer {

public:
  // functions
  Softmax(std::string name="");
  virtual ~Softmax();
  virtual int loadParam(std::vector<std::string> params, int offset=0);
  virtual int loadModel(std::ifstream& fp);
  virtual std::vector<std::vector<int>> inferShape(std::vector<std::vector<int>> bottom_shapes);
  virtual int forward(std::vector<Blob*>& bottoms, std::vector<Blob*>& tops);
  // variable
  // private
  Blob* exps;
};


} // namespace fcnn

#endif //__SOFTMAX_H