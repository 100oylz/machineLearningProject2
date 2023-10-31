import numpy as np
import torch.utils.data
import transformers.modeling_outputs
import logging
import LLM
import datastruct
from config import *
import utils
from PromptGenerate import PromptGenerate
from maskInfo import maskinfo
from colorlog import ColoredFormatter

def transform_data(data, token_format):
    # 获取batch_size
    batch_size = data.size(0)

    # 将二维tensor展平成一维
    flattened_data = data.view(batch_size, -1)

    # 使用格式化字符串对所有元素进行格式化
    formatted_data = [[token_format.format(item.item()) for item in sublist] for sublist in flattened_data]

    return formatted_data


def train(dataset: datastruct.datastruct, config: trainConfig):
    utils.setup_seed(config.seed)
    logConfig(config)
    model, tokenizer = LLM.getLLM()
    data, label, labelmap = dataset.discrete(slicenum=trainConfig.slice_num)
    traindata, trainlabel, validdata, validlabel, testdata, testlabel = datastruct.split_train_valid_test(data, label,
                                                                                                          randomstate=1)
    # LLM.tokenizer_add_new_tokens(tokenizer, LEVEL_TOKEN_FORMAT,  [str(i) for i in range(dataset.slicenum)])
    # LLM.tokenizer_add_new_tokens(tokenizer, LABEL_TOKEN_FORMAT, label)
    tokens_num = tokenizer.vocab_size
    promptModel = PromptGenerate(config.init_shape, config.emb_dim, config.embLength, config.output_length)
    maskModel = maskinfo(1024, len(labelmap), config.dropout)
    for param in model.parameters():
        param.requires_grad = False
    promptModel.to(config.device)
    model.to(config.device)
    maskModel.to(config.device)
    traindataset = utils.CustomDataset(traindata, trainlabel)
    validdataset = utils.CustomDataset(validdata, validlabel)

    traindataloader = torch.utils.data.dataloader.DataLoader(traindataset, shuffle=True, batch_size=config.batch_size)
    validdataloader = torch.utils.data.dataloader.DataLoader(validdataset, batch_size=config.batch_size)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.NAdam(list(promptModel.parameters()) + list(maskModel.parameters()), lr=config.lr,
                                  weight_decay=config.weight_decay)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)
    best_valid_loss = float('inf')
    for epoch in range(1, config.num_epochs + 1):
        # 重置计数器和累积值
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        valid_epoch_loss = 0.0
        valid_correct_predictions = 0
        total_valid_samples = 0
        for batch in traindataloader:
            data = batch['data']
            label = batch['label']
            data, maskpos = addDataAndMaskToPrompt(data, promptModel, tokenizer, tokens_num)
            data = data.to(config.device)
            label = label.to(config.device)
            with torch.no_grad():
                output = model(data)

            last_hidden_state, _ = output.to_tuple()
            maskItem = last_hidden_state[:, maskpos, :]
            out = maskModel(maskItem)

            loss = criterion(out, label.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算准确率
            _, predicted = torch.max(out, 1)
            correct_predictions += (predicted == label).sum().item()
            total_samples += label.size(0)

            # 累积epoch损失
            epoch_loss += loss.item()

        promptModel.eval()
        maskModel.eval()

        for batch in validdataloader:
            data = batch['data']
            label = batch['label']
            data, maskpos = addDataAndMaskToPrompt(data, promptModel, tokenizer, tokens_num)
            data = data.to(config.device)
            label = label.to(config.device)
            with torch.no_grad():
                output = model(data)

            last_hidden_state, _ = output.to_tuple()
            maskItem = last_hidden_state[:, maskpos, :]
            out = maskModel(maskItem)

            loss = criterion(out, label.long())

            # 计算准确率
            _, predicted = torch.max(out, 1)
            valid_correct_predictions += (predicted == label).sum().item()
            total_valid_samples += label.size(0)

            # 累积epoch损失
            valid_epoch_loss += loss.item()

        promptModel.train()
        maskModel.train()
        scheduler.step()
        # 计算epoch平均损失和准确率
        epoch_loss /= len(traindataloader)
        accuracy = correct_predictions / total_samples
        valid_epoch_loss /= len(validdataloader)
        valid_accuracy = valid_correct_predictions / total_valid_samples

        if valid_epoch_loss < best_valid_loss:
            best_valid_loss = valid_epoch_loss
            # 保存模型
            torch.save(promptModel, f'checkpoint/promptModel/{config.name}_promptModel.pt')
            torch.save(maskModel, f'checkpoint/maskModel/{config.name}_maskModel.pt')
        # 输出epoch损失和准确率
        # print(
        #     f'Epoch [{epoch}/{config.num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy * 100:.2f}%,Valid_Loss:{valid_epoch_loss:.4f},Valid_Accuracy:{valid_accuracy * 100:.2f}%')
        logging.info(
            f'Epoch [{epoch}/{config.num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {accuracy * 100:.2f}%, Valid Loss: {valid_epoch_loss:.4f}, Valid Accuracy: {valid_accuracy * 100:.2f}%')


def logConfig(config):
    # 创建一个ColoredFormatter
    formatter = ColoredFormatter(
        "%(white)s%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
            'DEBUG': 'white',
            'INFO': 'white',
            'WARNING': 'white',
            'ERROR': 'white',
            'CRITICAL': 'white,bg_red',
        },
        secondary_log_colors={},
        style='%'
    )

    logging.basicConfig(filename=f'journal/{config.name}.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

    # 创建一个用于在控制台输出的处理程序
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # 将处理程序添加到日志器
    logging.getLogger('').addHandler(console_handler)


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

    return reencoded_data_tensor, maskpos


if __name__ == '__main__':
    train(utils.ADNI, ADNIconfig)
    train(utils.PPMI,PPMIconfig)
