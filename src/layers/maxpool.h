#ifndef __MAXPOOL_H
#define __MAXPOOL_H

#include <fstream>
#include <sstream>

#include "layer.h"

namespace fcnn {

class MaxPool : public Layer {

public:
  // functions
  MaxPool(std::string name="");
  virtual ~MaxPool();
  virtual int loadParam(std::vector<std::string> params, int offset=0);
  virtual int loadModel(std::ifstream& fp);
  virtual std::vector<std::vector<int>> inferShape(std::vector<std::vector<int>> bottom_shapes);
  virtual int forward(std::vector<Blob*>& bottoms, std::vector<Blob*>& tops);
  // variable
  int kernel;
  int stride;
  int pad;
  int pad_left, pad_right, pad_top, pad_down;
  int output_h;
  int output_w;
  // private data
  Blob* padded;
};


} // namespace fcnn

#endif //__MAXPOOL_H