# -*- coding:utf-8 -*-
import torch
# import rich


class ProjectConfig(object):
    def __init__(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # 硬件配置
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 模型和数据路径
        #self.pre_model = r'D:\02-weights\chatglm2-6b-int4'
        self.pre_model = '/ai/data/dev/pre_models/ChatGLM'
        self.train_path = '/ai/data/dev/chatglm_lora/data/mixed_train_dataset.jsonl'
        self.dev_path = '/ai/data/dev/chatglm_lora/data/mixed_dev_dataset.jsonl'
        # 使用lora不适用ptuning
        self.use_lora = True
        self.use_ptuning = False
        # lora的秩选择8
        self.lora_rank = 8
        # 批次大小，轮次大小
        self.batch_size = 1
        self.epochs = 2
        # 学习率（控制模型参数更新的步长大小）
        # 权重衰减（权重衰减(即L2正则化)的系数，设为0表示不使用权重衰减），
        # 预热比例（表示在前6%的训练步骤中，学习率将从0线性增加到设定值），
        # 最大源序列长度，最大目标序列长度，日志步数，保存频率，预训练模型长度，前缀投影
        self.learning_rate = 3e-5
        self.weight_decay = 0
        self.warmup_ratio = 0.06
        self.max_source_seq_len = 100
        self.max_target_seq_len = 100
        self.logging_steps = 10
        self.save_freq = 200
        self.pre_seq_len = 128  # p-tuning前缀，应该不会用到
        self.prefix_projection = False # 默认为False,即p-tuning,如果为True，即p-tuning-v2
        self.save_dir = '/ai/data/dev/chatglm_lora/models_saved'


if __name__ == '__main__':
    pc = ProjectConfig()
    print(pc.save_dir)