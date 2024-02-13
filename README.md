
# DIY inference framework

## 使用的技术和开发环境
* 开发语言：C++ 17
* 数学库：Armadillo + OpenBlas(或者更快的Intel MKL)
* 加速库：OpenMP
* 单元测试：Google Test
* 性能测试：Google Benchmark


## 已经支持的算子
**总体理念：逐步优化已经有的算子；有需要的时候再对未实现的算子进行开发**

- Convolution 
- AdaptivePooling 
- MaxPooling 
- Expression(抽象语法树)
- Flatten, View(维度展平和变形)
- Sigmoid 
- HardSigmoid 
- HardSwish 
- ReLU
- Linear(矩阵相乘)
- Softmax 
- BatchNorm
- Upsample
- SiLU
- Concat
- ConvTranspose

## 目录
**source**是源码目录

1. **data/** 是张量类Tensor的实现和Tensor初始化方法
2. **layer/** 是算子的实现
3. **parser/** 是Pnnx表达式的解析类(比ONNX有更多的图优化和算子融合)
4. **runtime/** 是计算图结构，解析和运行时相关

**test**是单元测试目录，基本做到public方法单元测试权覆盖

**bench**是google benchmark, 包含对MobilenetV3, Resnet18和yolov5s的性能测试。

## 性能测试
### 测试设备

15 核心的AMD EPYC 7543(霄龙) 32-Core Processor (Docker 容器，宿主机共有32核心)

### 编译环境

gcc (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0

### 性能结果
耗时通过连续五次运行,并以求平均的方式计算

| **input size**         | **模型名称**     | **计算设备**              | **耗时**         |
| ---------------------- | ---------------- | ------------------------- | ---------------- |
| 224×224 batch = 8      | MobileNetV3Small | CPU(armadillo + openblas) | 6.76ms / image   |
| 224×224 batch = 8      | ResNet18         | CPU(armadillo + openblas) | 23.53ms / image  |
| 224×224 batch =16      | ResNet18         | CPU(armadillo + openblas) | 13.52ms / image  |
| 640×640 batch = 8      | Yolov5nano       | CPU(armadillo + openblas) | 78.37ms / image  |
| **640×640** batch = 8  | **Yolov5s**      | CPU(armadillo + openblas) | 177.54ms / image |
| **640×640** batch = 16 | **Yolov5s**      | CPU(armadillo + openblas) | 134.57ms / image |

## 致谢

推理框架NCNN，已经在借鉴的代码中保留了NCNN的BSD协议 https://github.com/Tencent/ncnn

优秀的数学库Openblas: https://github.com/xianyi/OpenBLAS

优秀的数学库Armadillo: https://arma.sourceforge.net/docs.html

给予我灵感的Caffe框架: https://github.com/BVLC/caffe
