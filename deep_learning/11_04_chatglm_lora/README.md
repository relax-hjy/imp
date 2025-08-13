本项目目标是实现文本的信息抽取和文本分类

LLM：chatglm
peft微调方法：lora

数据格式：
instruction+ question（任务+文本输入）+ answer（答案是json格式）

单卡：48G
Python：3.9
Pytorch：1.11.0
Cuda：11.3.1