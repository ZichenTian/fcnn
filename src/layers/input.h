#ifndef __INPUT_H
#define __INPUT_H

#include <fstream>
#include <sstream>

#include "layer.h"

namespace fcnn {

class Input : public Layer {

public:
  // functions
  Input(std::string name="");
  virtual ~Input();
  virtual int loadParam(std::vector<std::string> params, int offset=0);
  virtual int loadModel(std::ifstream& fp);
  virtual std::vector<std::vector<int>> inferShape(std::vector<std::vector<int>> bottom_shapes);
  virtual int forward(std::vector<Blob*>& bottoms, std::vector<Blob*>& tops);
  // variables
  std::vector<int> input_shape;
};



} // namespace fcnn

#endif //__INPUT_H