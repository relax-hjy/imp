# coding:utf-8
from torch.utils.data import DataLoader
from transformers import default_data_collator, AutoTokenizer
from data_handle.data_preprocess import *
from project_config import *
from torch.utils.data.distributed import DistributedSampler

from functools import partial

pc = ProjectConfig() # 实例化项目配置文件

tokenizer = AutoTokenizer.from_pretrained(pc.pre_model, trust_remote_code=True)


def get_data():
    dataset = load_dataset('text', data_files={'train': pc.train_path,
                                               'dev': pc.dev_path})


    new_func = partial(convert_example2features,
                       tokenizer=tokenizer,
                       max_source_seq_len=200,
                       max_target_seq_len=150)

    dataset = dataset.map(new_func, batched=True)
    train_dataset = dataset["train"]
    dev_dataset = dataset["dev"]
    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  collate_fn=default_data_collator,
                                  batch_size=pc.batch_size)
    dev_dataloader = DataLoader(dev_dataset,
                                collate_fn=default_data_collator,
                                batch_size=pc.batch_size)
    return train_dataloader, dev_dataloader

def get_data_dist():
    dataset = load_dataset('text', data_files={'train': pc.train_path,
                                               'dev': pc.dev_path})


    new_func = partial(convert_example2features,
                       tokenizer=tokenizer,
                       max_source_seq_len=100,
                       max_target_seq_len=100)

    dataset = dataset.map(new_func, batched=True)
    train_dataset = dataset["train"]
    dev_dataset = dataset["dev"]
    train_sampler = DistributedSampler(train_dataset)
    dev_sampler = DistributedSampler(dev_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  #shuffle=True,
                                  collate_fn=default_data_collator,
                                  batch_size=pc.batch_size,
                                  sampler=train_sampler)
    dev_dataloader = DataLoader(dev_dataset,
                                collate_fn=default_data_collator,
                                batch_size=pc.batch_size,
                                sampler=dev_sampler)
    return train_dataloader, dev_dataloader
if __name__ == '__main__':
    train_dataloader, dev_dataloader = get_data()
    print(len(train_dataloader))
    print(len(dev_dataloader))
    for i, value in enumerate(train_dataloader):
        print(i)
        print(value)
        print(value['input_ids'].shape)
        print(value['labels'].shape)
        break
