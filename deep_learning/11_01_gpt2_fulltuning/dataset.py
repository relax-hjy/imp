from torch.utils.data import Dataset
import torch
import pickle

class MyDataset(Dataset):

    def __init__(self, input_list,max_len):
        super().__init__()
        self.input_list=input_list
        self.max_len=max_len

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self,index):
        input_ids=self.input_list[index]
        input_ids=input_ids[:self.max_len]
        input_ids=torch.tensor(input_ids,dtype=torch.long)
        return input_ids


if __name__ == '__main__':
    with open("") as f:
        train_input_list = pickle.load(f)

    mydataset = MyDataset(train_input_list, max_len=300)