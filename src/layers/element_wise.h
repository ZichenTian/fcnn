#ifndef __ELEMENT_WISE_H
#define __ELEMENT_WISE_H

#include <fstream>
#include <sstream>

#include "layer.h"

namespace fcnn {

class Eltwise : public Layer {

public:
  // functions
  Eltwise(std::string name="");
  virtual ~Eltwise();
  virtual int loadParam(std::vector<std::string> params, int offset=0);
  virtual int loadModel(std::ifstream& fp);
  virtual std::vector<std::vector<int>> inferShape(std::vector<std::vector<int>> bottom_shapes);
  virtual int forward(std::vector<Blob*>& bottoms, std::vector<Blob*>& tops);
  // variable
  typedef enum {
      SUM = 0,
  }Eltwise_Operation_Typedef;

  Eltwise_Operation_Typedef operation;
};


} // namespace fcnn

#endif //__ELEMENT_WISE_H