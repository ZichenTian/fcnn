#ifndef __GLOBAL_AVEPOOL_H
#define __GLOBAL_AVEPOOL_H

#include "layer.h"

namespace fcnn {

class GlobalAvePool : public Layer {

public:
  // functions
  GlobalAvePool(std::string name="");
  virtual ~GlobalAvePool();
  virtual int loadParam(std::vector<std::string> params, int offset=0);
  virtual int loadModel(std::ifstream& fp);
  virtual std::vector<std::vector<int>> inferShape(std::vector<std::vector<int>> bottom_shapes);
  virtual int forward(std::vector<Blob*>& bottoms, std::vector<Blob*>& tops);
  // variable

};


}   // namespace fcnn


#endif  // __GLOBAL_AVEPOOL_H