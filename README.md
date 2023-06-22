# FOUR:Operator和Layer

## Operator

定义一个Operator类，它是一个父类，其余的Operator，如ReluOperator SigmoidOperator都是其派生类, 
Operator的作用就是会存放节点相关的参数.

定义一个新的Operator, 就是要定义OP的特有的参数(如Relu中的thresh).就是定义其Forward函数如下(![如Relu的计算过程](/assets/relu.PNG))

## Layer

Operator起到了属性存储的作用而具体的运算由Layer类负责, 以ReluLayer为例: 在公共父类Layer中我们用层的名字来完成构造函数, 而在ReluLayer中我们需要根据ReLuOperator类去完成构造函数. 

定义一个新的Layer, 就是要定义Layer的Forward函数(以Relu举例):

```
// begin: transform
    output_data->data().transform([&](float value) {
      float thresh = op_->get_thresh();
      //x >= thresh
      if (value >= thresh) {
        return value; // return x
      } else {
        // x<= thresh return 0.f;
        return 0.f;
      }
    });
    // end: transform
```

## 设计模式: 工厂模式

这里使用了设计模式中的简单工厂模式和单例模式, 不论使用哪种工厂模式其主要目的都是实现类与类之间的解耦合，这样我们在创建对象的时候就变成了拿来主义，使程序更加便于维护.

https://subingwen.cn/design-patterns/simple-factory/

https://subingwen.cn/design-patterns/singleton/

在这里实现的算子注册机制中, 执行时
```
std::shared_ptr<Operator> relu_op = std::make_shared<ReluOperator>(thresh);
std::shared_ptr<Layer> relu_layer = LayerRegisterer::CreateLayer(relu_op);
```

控制流为：
ReluLayer定义完成 ---> LayerRegistererWrapper ---> RegisterCreator --->Registry返回注册表 ---> 在注册表中存入实现方法

CreateLayer ---> Registry返回注册表内容 ---> 取出creator ---> 返回creator生成的Layer实例







