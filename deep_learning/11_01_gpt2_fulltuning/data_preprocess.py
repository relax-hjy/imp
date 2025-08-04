from transformers import BertTokenizerFast
import pickle
from tqdm import tqdm


def data_preprocess(train_txt_path, train_pkl_path):
    """
    将训练数据处理成向量后保存
    :param train_txt_path:
    :param train_pkl_path:
    :return:
    """
    tokenizer = BertTOkenizerFast(
        '', sep_token='[SEP]', cls_token='[CLS]', pad_token='[PAD]'
    )

    sep_id = tokenizer.sep_token_id
    cls_id = tokenizer.cls_token_id
    print(sep_id, cls_id)

    with open(train_txt_path, "rb") as f:
        data = f.read().decode("utf-8")

    if "\r\n" in data:
        data = data.replace("\r\n", "\n\n")
        train_data = data.split("\n\n")
    else:
        train_data = data.split("\n\n")

    dialogue_len = []
    dialogue_list = []

    for index, dialogue in enumerate(tqdm(train_data)):
        sequences = dialogue.split("\n")
        print(sequences)

        input_ids = [cls_id]
        for sequence in sequences:
            input_ids += tokenizer.encode(sequence, add_special_tokens=False)
            input_ids.append(sep_id)

        dialogue_len.append(len(input_ids))
        dialogue_list.append(input_ids)
    with open(train_pkl_path, "wb") as f:
        pickle.dump(dialogue_list, f)

