import torch
print(torch.__version__)
import torch.distributed as dist
print(dist.__file__)
# import torch.distributed.device_mesh device_mesh 2.2 引入  高版本accelerate会调用这个函数初始化分布式。
import os
import time
import copy
import argparse
from functools import partial
import peft
# autocast是PyTorch中一种混合精度的技术，可在保持数值精度的情况下提高训练速度和减少显存占用。
# 该方法混合精度训练，如果在CPU环境中不起任何作用
from torch.cuda.amp import autocast as autocast
from transformers import AutoTokenizer, AutoConfig, AutoModel, get_scheduler,AutoModelForCausalLM
from utils.common_utils import *
from data_handle.data_loader import *
from project_config import *
import os
def init_distributed():
    # 可以通过环境变量读取 rank 和 world_size 等信息
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    # 选择后端：
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank, world_size

torch.distributed.init_process_group(backend='nccl')
    # 获得当前进程使用的gpu号
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
print('local-->', local_rank, 'device-->', device)
if __name__ == '__main__':
    #rank, world_size = init_distributed()


    tokenizer = AutoTokenizer.from_pretrained(pc.pre_model, trust_remote_code=True)

    config = AutoConfig.from_pretrained(pc.pre_model, trust_remote_code=True)
    print('结束')
    # python - m
    # torch.distributed.launch - -nproc_per_node = 4 - -nnodes = 1 - -node_rank = 0 - -master_addr = 127.0
    # .0
    # .1 - -master_port = 29500
    # scan.py