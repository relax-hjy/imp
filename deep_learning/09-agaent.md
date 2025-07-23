## agent

通过工程化让agent完成一些复杂的问题

属于大模型应用



人需要适应微信，若果是agent会适应人类。

o1是用来解决一些复杂任务的。

agent也是用来解决复杂问题的。





### function call

函数调用

以前大预言耐磨性处理文本

用api function训练大模型，让大模型使用api 和function

### 什么是function call





2023 6 23 openai 发布了functioncall 功能，旨在大预言模型中集成调用外部api的能力。

![image-20250722192601568](./assets/image-20250722192601568.png)



使用领域

信息实时性  天气 新闻 股价

数据局限性 获得特定领域的信息  比如全网搜索

功能扩展性 代码运行能力 计算器等等





### function 工作原理

client   server    gpt api

![image-20250722193746631](./assets/image-20250722193746631.png)





### 调用单一函数

部署模型，获取api  定义function_call  输入prompt到大模型，模型输出函数参数，调用本地函数得到返回，融入prompt，再次送入模型，获得结果。

3.10   pip install zhipuai

查询外部天气的函数，描述函数功能，解析模型参数调用函数

![image-20250723112504697](./assets/image-20250723112504697.png)



![image-20250723112737583](./assets/image-20250723112737583.png)



定义模型应用函数



### 调用多个函数





调用数据库



连接数据库查询的聊天机器人

database  表结构  字段



定义查询数据库的函数   描述函数功能tools 

## gpts

什么是gpts

再不写大模型的情况下，创建属于自己的gpt版本，通过prompt构建个人的ai助手。

coze

利用coze平台基于本地知识库快速搭建一个学习答疑的bot，

收集数据，基于coze搭建知识库，搭建bot，将coze连到平台实现应用。

![image-20250723164358423](./assets/image-20250723164358423.png)

多个gpts的应用

https://github.com/chatchat-space/Langchain-Chatchat



assistant api

gpts肯定不是维护多个大模型

几百万个gpts都是通过assistant api 维护的

什么是assitant api



agent

