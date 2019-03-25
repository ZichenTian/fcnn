#ifndef __BLOB_H
#define __BLOB_H

#include <vector>
#include <string>
#include <memory>

#define NOT_IMPLEMENT  { fprintf(stderr, "NOT IMPLEMENT!\n"); exit(1); }
#define INVALID_ERROR(str) { fprintf(stderr, "%s", (str)); exit(1); } 

namespace fcnn {

class Blob {

public:
  // functions
  Blob(std::string name=""); // empty
  Blob(int h, std::string name="");  // vector
  Blob(int h, int w, std::string name=""); // image
  Blob(int h, int w, int c, std::string name="");  // tensor
  Blob(std::vector<int> shape, std::string name=""); // unknown shape
  void reshape(int h);
  void reshape(int h, int w);
  void reshape(int h, int w, int c);
  std::string printShape(void);
  std::vector<int> getShape(void);
  Blob& copy(); // deep copy
  void fill(float* src, size_t len);
  void setValue(float value);
  float* getPtr();
  std::string printData();
  ~Blob();
  // variables
  std::string name;
  std::shared_ptr<float> data;
  int dims;
  int h;
  int w;
  int c;
  size_t size;
  int producer;
  std::vector<int> consumers;
};


} // namespace fcnn



#endif // __BLOB_H