#ifndef __NET_H
#define __NET_H

#include "layer.h"

namespace fcnn{

class Net {

public:
  Net();
  ~Net();
  int loadParam(std::string filename);
  int loadParam(std::ifstream& fp);
  int loadModel(std::string filename);
  int loadModel(std::ifstream& fp);
  int forward();
  void clear();
  Blob& getBlob(std::string name);    // for output
  Blob& getBlob(int index);
  int getBlobIndex(std::string name);
  void printNetInfo();
private:
  std::vector<Layer*> layers;
  std::vector<Blob*> blobs;

};

} // namespace fcnn


#endif //__NET_H