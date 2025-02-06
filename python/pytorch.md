cnn rnn transformer  自编码器 对抗生成网络



nvidia-smi

https://developer.nvidia.com/cuda-toolkit-archive

https://developer.nvidia.com/rdp/cudnn-archive



linux：/usr/local/cuda



```python
# torch.tensor()指定数据创建张量 数字，列表，数组，只接受一个参数
scalar=torch.tensor(7)
vector=torch.tensor([7,7])
matrix=torch.tensor([[7,8],[9,10]])
tensor=torch.tensor([[[1,2,3],[3,6,9],[2,4,5]],[[1,2,3],[3,6,9],[2,4,5]]])
```

```python
tensor.ndim  #张量的轴
```

```python
# torch.Tensor()指定数据或形状创建张量，指定数据的时候一定有方括号，包括列表和ndarray数组。
import numpy as np
nnn=np.random.randn(2,3)
ttt=torch.tensor(nnn)
print(ttt)
x=torch.Tensor(2,3)
x=torch.Tensor(nnn)
print(x)
y=torch.IntTensor([10])
z=torch.FloatTensor([11,12])
u=torch.DoubleTensor([11,12,13])
```