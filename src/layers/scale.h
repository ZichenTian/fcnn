#ifndef __SCALE_H
#define __SCALE_H

#include <fstream>
#include <sstream>

#include "layer.h"

namespace fcnn {

class Scale : public Layer {

public:
  // functions
  Scale(std::string name="");
  virtual ~Scale();
  virtual int loadParam(std::vector<std::string> params, int offset=0);
  virtual int loadModel(std::ifstream& fp);
  virtual std::vector<std::vector<int>> inferShape(std::vector<std::vector<int>> bottom_shapes);
  virtual int forward(std::vector<Blob*>& bottoms, std::vector<Blob*>& tops);
  // variable
  int channels;
  // private
  Blob* scale;
  Blob* bias;
};


} // namespace fcnn

#endif //__SCALE_H