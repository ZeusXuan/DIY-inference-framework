# FIVE:构建计算图

## PNNX
PNNX项目[https://github.com/Tencent/ncnn/tree/master/tools/pnnx]PyTorch Neural Network eXchange（PNNX）是PyTorch模型互操作性的开放标准。PNNX为PyTorch提供了一种开源的模型格式，它定义了与Pytorch相匹配的数据流图和运算图，我们的框架在PNNX之上封装了一层更加易用和简单的计算图格式。pytorch训练好一个模型之后，然后模型需要转换到pnnx格式，然后pnnx格式我们再去读取，形成计算图.

PNNX有不同level的图优化、算子融合的工作，所以我们可以读取PNNX的生成(PNNX graph definition和 PNNX model weight) 结构定义，然后在其上构建自己一种易用的计算图结构.

## PNNX的IR

Operand类组成:

1.Producer: 类型是operator, 表示产生了这个操作数的运算符(operator). 

2.Customer: 类型是std::vector<operator>, 表示需要这个操作数下一个操作的的运算符(operator)序列.

3.Name: 类型是std::string, 表示这个操作数的名称.

4.Shape: 类型是std::vector<int>, 用来表示操作数的大小, Add-->values-->Conv 1,values可能为(1x3x320x320).

Operator类组成:

1.Inputs和Outputs: 类型为std::vector<operand*>, 表示这个运算符计算过程中所需要的输入输出操作数(operand).

2.Type和Name: 类型均为std::string, 分别表示运算符号的类型和名称.

3.Params: 类型为std::map, 用于存放该运算符的所有参数(例如对应Conv, params中将存放stride, padding, kernel_size等信息).

4.Attrs: 类型为std::map, 用于存放运算符所需要的具体权重属性(例如对应Conv, 它的 attrs中就存放着卷积的权重参数和偏移量参数).

## 对PNNX的封装

加载PNNX的计算图, 获取PNNX计算图中的运算符(operators), 遍历PNNX计算图中的运算符, 并进行转化(见runtime_ir.cpp)

