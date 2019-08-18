# FCNN

## 说明

仿照NCNN写的深度学习前向推理框架：
* 静态图，网络结构和所有内存在初始化时即确定
* 仅计划开发CPU算子，包括x86和armv8
* 目前仅支持从caffe解析prototxt和caffemodel，转成fcnn认识的格式(*.param, *.model)
* 尚不支持图优化
* 菜的飞起，就酱

------

## 工程架构

- src/：blob、layer、net等核心
  - layers/：所有算子
- test/：测试文件
- examples/:例子
- models/：caffe和fcnn的模型
- tools/：caffe2fcnn

------

## 基本原理

* 执行过程：
  * net.loadparam()：读取param文件，构建网络，确定所有blob的尺寸并静态分配内存
  * net.loadmodel()：建立好网络后，读取model文件，依次填充参数
  * net.getBlob("input_blob_name")：获取输入blob的引用，填入数据
  * net.forward()：前传网络
  * net.getBlob("output_blob_name")：获取输出blob的引用，拿出数据
  * net.clear()：释放layer和blob
* 网络构建
  * 读取param文件的过程中即构建网络，同时确定网络中的尺寸信息(数据和参数)
* 内存管理：
  * 静态内存，在构建网络的过程中，数据和参数的内存即分配完毕
* 资源释放：
  * 需要调用net.clear()手动释放

------

## 已完成功能

* 完成了基本算子
  * convolution：六重for循环233，不支持group和depthwise
  * element_wise：目前仅支持sum
  * global_average_pooling
  * inner_product
  * input
  * max_pooling：仅支持2x2s1、2x2s2、3x3s1、3x3s2
  * relu
  * scale：batchnorm会转成scale
  * softmax
* caffe2fcnn：
  * 读取caffemodel和prototxt，转换成param和model文件
* resnet-18和caffe前传的数值一致

------

## 开发计划

* 算子优化
  * convolution im2col(马上做)
  * convolution winograd
  * maxpool neon
* 算子补充
* 测试代码开发(马上做)
* blob微调：静态内存，不需要智能指针；增加reshape
* 支持onnx
* 图优化(大坑)

------

## 吉祥物

![logo.jpg](examples/logo.jpg)

------

## example使用方法

1. clone下来
2. 编译工程

```
mkdir build
cd build
cmake ..
make
```

3. 读取prototxt和caffemodel，生成param和model文件

```
cd 根目录/build/tools
./caffe2ncnn prototxt caffemodel param model
```

4. 运行分类例子

```
cd 根目录/build/examples
./classification param model img
```

5. 生成install

```
cd 根目录/build
make install
```


