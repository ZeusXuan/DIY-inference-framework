# TWO:定义Tensor

Tensor由两部分组成
```
  std::vector<uint32_t> raw_shapes_;// 形状 (channel,row,col)
  arma::fcube data_;// 数据
```

具体接口的设计见参考:
https://arma.sourceforge.net/docs.html

