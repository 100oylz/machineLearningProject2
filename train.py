import numpy as np
import torch.utils.data
import transformers.modeling_outputs

import LLM
import datastruct
from config import *
import utils
from PromptGenerate import PromptGenerate


def transform_data(data, token_format):
    # 获取batch_size
    batch_size = data.size(0)

    # 将二维tensor展平成一维
    flattened_data = data.view(batch_size, -1)

    # 使用格式化字符串对所有元素进行格式化
    formatted_data = [[token_format.format(item.item()) for item in sublist] for sublist in flattened_data]

    return formatted_data


def train(dataset: datastruct.datastruct, config: trainConfig):
    model, tokenizer = LLM.getLLM()
    data, label, labelmap = dataset.discrete(slicenum=trainConfig.slice_num)
    # LLM.tokenizer_add_new_tokens(tokenizer, LEVEL_TOKEN_FORMAT,  [str(i) for i in range(dataset.slicenum)])
    # LLM.tokenizer_add_new_tokens(tokenizer, LABEL_TOKEN_FORMAT, label)
    tokens_num = tokenizer.vocab_size
    promptModel = PromptGenerate(config.init_shape, config.emb_dim, config.embLength, config.output_length)
    for param in model.parameters():
        param.requires_grad = False
    promptModel.to(config.device)
    model.to(config.device)
    dataset = utils.CustomDataset(data, label)
    utils.setup_seed(config.seed)
    dataloader = torch.utils.data.dataloader.DataLoader(dataset, shuffle=True, batch_size=config.batch_size)
    # print(model)
    for epoch in range(1, config.num_epochs + 1):
        for batch in dataloader:
            data = batch['data']
            label = batch['label']
            data, maskpos = addDataAndMaskToPrompt(data, promptModel, tokenizer, tokens_num)
            # print(data.shape)
            data = data.to(config.device)
            # print(data.shape)
            with torch.no_grad():
                output = model(data)
                last_hidden_state, _ = output.to_tuple()
                maskItem: torch.Tensor = last_hidden_state[:, maskpos, :]
                logmaskItem = maskItem.logit()
                logmaskItem[torch.isnan(logmaskItem)] = -torch.inf
                print(logmaskItem)
                print(logmaskItem.shape)

                # mask_tensor=pooler_output[]

            break
        break


def encode_data_with_special_symbols(data, tokenizer):
    encoded_data = []
    # vocabs=tokenizer.get_added_vocab()
    # print(set(vocabs))
    for item in data:
        # 将特殊符号组成的矩阵转换为字符串
        item_str = "".join(item)  # 假设特殊符号之间用空格分隔
        encoded_item = tokenizer.encode(item_str, add_special_tokens=False)
        encoded_data.append(encoded_item[0])
    return encoded_data


def addDataAndMaskToPrompt(data, promptModel, tokenizer, tokens_num):
    data = transform_data(data, LEVEL_TOKEN_FORMAT)
    # data = [subdata.__repr__().replace("'", "") for subdata in data]
    encoded_data = []
    for item in data:
        encoded_item = encode_data_with_special_symbols(item, tokenizer)
        encoded_data.append(encoded_item)
    datalength = len(encoded_item) + trainConfig.output_length
    # print(datalength)
    prompt, maskpos = promptModel(tokens_num, datalength, tokenizer)
    # print(prompt, maskpos)
    # 将 prompt 和 encoded_data 拼接成一个列表

    combined_data = [prompt.tolist() + data for data in
                     encoded_data]
    assert all([len(data) == datalength for data in combined_data]), "Length Error"
    for data in combined_data:
        data.insert(tokenizer.mask_token_id, maskpos)

    # 将重新编码后的数据转换为 torch.Tensor
    reencoded_data_tensor = torch.LongTensor(combined_data)
    print('Add Success!')
    # print(reencoded_data_tensor)
    print(tokens_num)
    return reencoded_data_tensor, maskpos


if __name__ == '__main__':
    train(utils.ADNI, ADNIconfig)
