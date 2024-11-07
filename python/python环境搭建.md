## python环境搭建

### 下载与安装python

python解释器

anaconda环境管理工具

### 运行python

1.交互式解释器编程：python

2.命令行脚本      python script.py文件

3.pycharm，vscode集成开发环境



## python基本语法

#### python脚本的编码方式

```python
"""
\u4f60 是字符的 Unicode 码点表示（2字节）
\xe4\xbd\xa0 是同一个字符的 UTF-8 编码表示（3字节）
这就是为什么同一个汉字会有不同的字节表示方式。Unicode 是字符集（定义了字符和码点的对应关系），而 UTF-8 是编码方案（定义了如何在计算机中存储这些字符）。"""
string="\x41\u0041hello你好\U00000394\u4f60\u597d"
print(string)
bytes_data=string.encode()  #将字符串编码为字节串
print(bytes_data)
string_new=bytes_data.decode() # 将字节串解码为字符串
print(string_new)
```

python脚本文件以UTF-8编码，所有字符都是unicode字符串。



**字符串（string）：**
字符串是由Unicode字符组成的序列。在Python中，字符串用于处理文本数据，可以包含各种字符，包括字母、数字、符号等。字符串也是不可变的。

**字节串（bytes）：**
字节串是由0-255范围内的整数构成的序列，用于在程序中处理8位字节数据。字节串通常用于处理二进制数据，如文件、网络数据等。字节串是不可变的，这意味着一旦创建了一个字节串，就不能修改它。

#### python标识符

1.由数字，字母，下划线组成。

2.不能以数字开头。不能是关键字

3.汉字也能用，但是不推荐

#### python关键字（保留字）

运算符号，常量，程序控制语句

```python
import keyword
print(keyword.kwlist)
'''
['False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield']
'''
```

#### 缩进与换行

1.python使用回车来代替分号。一行一条独立的语句。缩进的空格数是可变的，但是同一个代码块的语句必须包含相同的缩进空格数。

2.python使用缩进来表示{}，相同缩进代表同一个代码块 。

3.python通常用一行写完，如果语句比较长，可以使用\放在句尾来实现多行语句。

在[],{},()中的多行语句不需要使用反斜杠。注意不包括"""\n"""

#### 注释

1.单行注释使用#号

2.多行注释使用三引号或单引号



