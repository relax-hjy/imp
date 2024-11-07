### 配置python环境

#### python环境

python：python解释器  下载python解释器 ，配置环境变量，cmd验证

anaconda：官网填写邮箱下载安装包，配置环境变量。

#### 集成开发环境：

pycharm:jetbrain官网下载安装包，下载激活包，安装激活即可。

终端：python --version -V python执行脚本 python交互式解释器



### python特点

#### 标识符和关键字

标识符（变量名，函数名，类名）：

1. ​	标识符由数字，字母，下划线组成。
2. ​    不能以数字开头。
3. ​	不能是关键字：Ture，False，None，if，else，try。

关键字：

```python
import keyword
print(keyword.kwlist)
```

#### 缩进和行

用行来区别每句代码，即用回车代替分号

用缩进区分代码块

多行语句需要在行末用\联系

{}[]()  “”“ ‘’‘中不需要用反斜杠\

#### 注释

单行用# ，多行用”“” ’‘’ 包裹的字符串常量

