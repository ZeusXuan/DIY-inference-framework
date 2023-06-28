# SIX:解析表达式

## 词法解析

词法解析的目的就是将add(@0,mul(@1,@2))拆分为多个token, token依次为:add, (, @0, mul, (, @1, @2), ), 代码如下:

```
enum class TokenType {
  TokenUnknown = -1,
  TokenInputNumber = 0,
  TokenComma = 1,
  TokenAdd = 2,
  TokenMul = 3,
  TokenLeftBracket = 4,
  TokenRightBracket = 5,
};

struct Token {
  TokenType token_type = TokenType::TokenUnknown;
  int32_t start_pos = 0; //词语开始的位置
  int32_t end_pos = 0; // 词语结束的位置
  Token(TokenType token_type, int32_t start_pos, int32_t end_pos): token_type(token_type), start_pos(start_pos), end_pos(end_pos) {

  }
};
```

## 语法解析
词法解析得到token数组之后, 我们对语法进行分析, 并得到最终产物抽象语法树(AST). 语法解析的过程是递归向下的, 定义在Generate_函数中.

```
struct TokenNode {
  int32_t num_index = -1;
  std::shared_ptr<TokenNode> left = nullptr;
  std::shared_ptr<TokenNode> right = nullptr;
  TokenNode(int32_t num_index, std::shared_ptr<TokenNode> left, std::shared_ptr<TokenNode> right);
  TokenNode() = default;
};
```

AST就是一个二叉树, 其中存储它的左子节点和右子节点以及对应的操作编号num_index. num_index为正, 则表明是输入的编号, 例如@0,@1中的num_index依次为1和2. 如果num_index为负数则表明当前的节点是mul或者add等operator.(这里只实现了add和mul), 得到AST后将其转化为逆波兰表达式便于处理.

下面举一个例子:
1.传入Expression:string, 例如add(mul(@0,@1),@2)

2.将add(mul(@0,@1),@2)按照词法分析为多个tokens, 且在拆分的时候需要进行词法校验

3.根据已知的tokens, 通过递归向下遍历的语法分析得到对应的计算二叉树. 二叉树的各个节点为add,mul,@0,@1

4.将计算二叉树进行逆波兰变换, 得到的逆波兰式如下:@0, @1, mul, @2, add.

## Expression Layer

Expression Op:
```
class ExpressionOp : public Operator {
 public:
  explicit ExpressionOp(const std::string &expr);
  std::vector<std::shared_ptr<TokenNode>> Generate();

 private:
  std::unique_ptr<ExpressionParser> parser_;
  std::vector<std::shared_ptr<TokenNode>> nodes_;
  std::string expr_;
};
```

其中expr_表示表达式字符串用于初始化, nodes_表示经过parser_逆波兰变换之后得到的节点.

而Expression Layer的Forward函数就是要batch of cube的add和mul运算. 使用逆波兰表达式配合栈进行处理.

