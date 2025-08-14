from datasets import load_dataset
from transformers import AutoTokenizer
from project_config import ProjectConfig
import numpy as np

import json
import traceback
import sys

from tqdm import tqdm
from rich import print as rprint

sys.path.append('..')


def convert_example2features(examples: dict,
                             tokenizer,
                             max_source_seq_len: int,
                             max_target_seq_len: int,
                             )-> dict:
    """

    :param examples:
    :param tokenizer:
    :param max_source_seq_len: prompt最大长度
    :param max_target_seq_len: target最大长度
    :return:
    dict(str:np.ndarray)
    """
    """
    将样本数据转换为Prompt-tuning模型接收的输入数据。

    Args:
        examples (dict): 训练数据样本, e.g. -> {
                                                "text": [
                                                            '{"context": "年基准利率4.35%。从实际看...", "target": "2017年银行贷款基准利率"}',
                                                            ...
                                                ]
                                            }
        max_source_seq_len (int): prompt最大长度
        max_target_seq_len (int): 答案最大长度

    Returns:
        dict (str: np.array) -> tokenized_output = {
                            'input_ids': [[1525, 10, ...], [758, 2345, ...]],
                            'labels': [[822, 10, ...], [125, 58...]]
                        }
    """
    tokenized_output = {
        'input_ids': [],   # 输入是文本id
        'labels': []       # 输出应该是错位
    }

    max_seq_length = max_source_seq_len + max_target_seq_len

    for example in tqdm(examples['text']):
        try:
            examples = json.loads(example)   # 将字符串转为字典
            context = examples['context']    # 获取prompt
            target = examples['target']      # 获取答案
            print(f"context-->\n{context}")
            print(f"target-->\n{target}")

            prompts_ids = tokenizer.encode(
                text=context,
                add_special_tokens=False
            )

            target_ids = tokenizer.encode(
                text=target,
                add_special_tokens=False
            )

            if len(prompts_ids) >= max_source_seq_len:  # 裁剪，方便添加gmax
                prompts_ids = prompts_ids[:max_source_seq_len - 1]
            if len(target_ids) >= max_target_seq_len - 1:  # 裁剪，添加sop eop
                target_ids = target_ids[:max_target_seq_len - 2]
            # print(f'new_prompts_ids--》{prompts_ids}\n{len(prompts_ids)}')
            # print(f'new_target_ids--》{target_ids}\n{len(target_ids)}')
            # a = tokenizer.convert_tokens_to_string(target_ids)
            # print(a)

            inputs_ids = tokenizer.build_inputs_with_special_tokens(prompts_ids, target_ids)

            # 回答开始的地方
            context_length = inputs_ids.index(tokenizer.bos_token_id)

            mask_position = context_length - 1

            labels = [-100] * context_length + inputs_ids[mask_position + 1:]

            pad_len= max_seq_length - len(inputs_ids)

            inputs_ids = inputs_ids + [tokenizer.pad_token_id] * pad_len

            labels = labels + [-100] * pad_len
            print(inputs_ids,labels)
            tokenized_output['input_ids'].append(inputs_ids)
            tokenized_output['labels'].append(labels)

        except:
            print(f'"{example}" -> {traceback.format_exc()}')
            continue

    for k,v in tokenized_output.items():
        tokenized_output[k] = np.array(v)


    return tokenized_output

def get_max_length(tokenizer,dataset_file):
    source_seq_len_list=[]
    target_seq_len_list=[]
    with open(dataset_file,'r') as f:
        for line in tqdm(f.readlines()):
            line=json.loads(line)

            source_len=tokenizer.encode(line['context'])
            source_seq_len_list.append(len(source_len))

            target_len=tokenizer.encode(line['target'])
            target_seq_len_list.append(len(target_len))

    print(dataset_file)
    print(f"【Source Sequence】 Max: {max(source_seq_len_list)}, Avg: {int(sum(source_seq_len_list) / len(source_seq_len_list))}, Middle: {sorted(source_seq_len_list)[int(len(source_seq_len_list) / 2)]}.")
    print(f"【Target Sequence】 Max: {max(target_seq_len_list)}, Avg: {int(sum(target_seq_len_list) / len(target_seq_len_list))}, Middle: {sorted(target_seq_len_list)[int(len(target_seq_len_list) / 2)]}.")



if __name__ == '__main__':
    pc = ProjectConfig()
    '''DatasetDict({
        train: Dataset({
             features: ['text'],
             num_rows: 902
        })
    })'''
    train_dataset = load_dataset('text', data_files={'train': pc.train_path})
    print(type(train_dataset))
    rprint(train_dataset)
    '''Dataset({
        features: ['text'],
        num_rows: 902
    })'''
    rprint(train_dataset['train'])
    # 字符串的列表
    rprint(train_dataset['train']['text'])

    tokenizer = AutoTokenizer.from_pretrained(pc.pre_model, trust_remote_code=True)
    tokenized_output = convert_example2features(train_dataset['train'],
                                                  tokenizer=tokenizer,
                                                  max_source_seq_len=30,
                                                  max_target_seq_len=20,
                                                  )

    print(len(tokenized_output["input_ids"][0]))
    print(len(tokenized_output["labels"][0]))


    rprint(get_max_length(tokenizer, pc.train_path))
