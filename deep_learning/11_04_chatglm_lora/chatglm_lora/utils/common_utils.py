import torch
import torch.nn as nn
from project_config import *
import copy
pc = ProjectConfig()

class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def second2time(seconds: int):
    """
    将秒转换成时分秒。

    Args:
        seconds (int): _description_
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)


def save_model(
        model,
        cur_save_dir: str
    ):
    """
    存储当前模型。

    Args:
        cur_save_path (str): 存储路径。
    """
    if pc.use_lora:                       # merge lora params with origin model
    #     merged_model = copy.deepcopy(model)
    #     # 如果直接保存，只保存的是adapter也就是lora模型的参数
    #     merged_model = merged_model.merge_and_unload()
    #     merged_model.save_pretrained(cur_save_dir)
    # 先清理显存
        torch.cuda.empty_cache()
    # 在CPU上进行模型合并
        with torch.no_grad():
            # 复制到CPU上
            cpu_model = copy.deepcopy(model.cpu())
            # 合并LoRA权重
            merged_model = cpu_model.merge_and_unload()
            # 保存模型
            merged_model.save_pretrained(cur_save_dir)
            # 将原模型移回GPU
            model.to(pc.device)
            # 再次清理显存
            torch.cuda.empty_cache()
    else:
        model.save_pretrained(cur_save_dir)