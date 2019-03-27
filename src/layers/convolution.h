#ifndef __CONVOLUTION_H
#define __CONVOLUTION_H

#include "layer.h"

#define IM2COL_CONV 1

namespace fcnn {

class Convolution : public Layer {
public:
  // functions
  Convolution(std::string name="");
  virtual ~Convolution();
  virtual int loadParam(std::vector<std::string> params, int offset=0);
  virtual int loadModel(std::ifstream& fp);
  virtual std::vector<std::vector<int>> inferShape(std::vector<std::vector<int>> bottom_shapes);
  virtual int forward(std::vector<Blob*>& bottoms, std::vector<Blob*>& tops);
  // variable
  int kernel;
  int stride;
  int pad;
  int bias;
  int pad_left, pad_right, pad_top, pad_down;
  int output_h;
  int output_w;
  int output_c;
  int input_c;
  // private data
  Blob* padded;
  Blob* conv_weight;
  Blob* bias_weight;
#ifdef IM2COL_CONV
  Blob* col_m;
  void im2col(Blob* bottom);
  void gemm(Blob* top);
#endif
};



}   // namespace fcnn

#endif  // __CONVOLUTION_H