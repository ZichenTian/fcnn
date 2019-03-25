#ifndef __INNERPRODUCT_H
#define __INNERPRODUCT_H

#include "layer.h"

namespace fcnn {

class InnerProduct : public Layer {

public:
  InnerProduct(std::string name="");
  virtual ~InnerProduct();
  virtual int loadParam(std::vector<std::string> params, int offset=0);
  virtual int loadModel(std::ifstream& fp);
  virtual std::vector<std::vector<int>> inferShape(std::vector<std::vector<int>> bottom_shapes);
  virtual int forward(std::vector<Blob*>& bottoms, std::vector<Blob*>& tops);
  // variable
  int input_c;
  int output_c;
  // private
  Blob* multi_weight;
  Blob* bias_weight;

};


}   // namespace fcnn



#endif  // __INNERPRODUCT_H