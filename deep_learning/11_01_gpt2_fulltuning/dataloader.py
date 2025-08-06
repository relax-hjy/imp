import torch.nn.urils.rnn as rnn_tuils
from torch.utils.data import DataLoader
import pickle
import torch
from dataset import *
from parameter_config import *

params=ParameterConfig()

def load_dataset(train_path,valid_path):
    with open(train_path,"rb") as f:
        train_data = pickle.load(f)
    with open(valid_path,"rb") as f:
        train_data = pickle.load(f)

    train_dataset=MyDataset(train_data)
    valid_dataset=MyDataset(valid_data)
    return train_dataset,valid_dataset

def collate_fn(batch):
    input_ids=rnn_tuils.pad_sequence(batch,
                                     batch_first=True,
                                     padding_value=0)

    labels=rnn_tuils.pad_sequence(batch,
                                  batch_first=True,
                                  padding_value=-100)
    return input_ids,labels
def get_dataloader(train_path,valid_path):
    train_dataset,valid_dataset=load_dataset(train_path,valid_path)

    train_dataloader=DataLoader(train_dataset,
                                batch_size=params.batch_size,
                                shuffle=True,
                                collate_fn=collate_fn,
                                drop_last=True)
    valid_dataloader=DataLoader(valid_dataset,
                                batch_size=params.batch_size,
                                shuffle=True,
                                collate_fn=collate_fn,
                                drop_last=True)

    return train_dataloader,valid_dataloader
    )