import time

import torch.utils.data
import LLM
import config
import customdataset

from config import *
import utils
from PromptGenerate import PromptGenerate
from maskInfo import maskmodel
from datastruct import split_train_valid_test
from dataclasses import dataclass
from utils import logConfig


@dataclass
class dim3Dict():
    num: int
    last_item: int


def ensembleLearning(output: torch.Tensor, label: torch.Tensor, labelmap, datalength):
    output = output.view(-1, datalength, len(labelmap))
    itemlist = []
    sortedkeyslist = []
    for j in range(output.shape[0]):
        itemdict = {}
        for i in range(len(labelmap)):
            itemdict[i] = dim3Dict(num=0, last_item=-1)
        for i in range(output.shape[1]):
            # print(output.shape)
            item = output[j, i, :]
            # print(item.shape)
            _, predicted = torch.max(item, dim=0)

            itemdict[predicted.item()].num += 1
            itemdict[predicted.item()].last_item = i

        # 找到num最大的几个类别的键
        max_num = max(itemdict.values(), key=lambda x: x.num).num
        max_num_keys = [key for key, value in itemdict.items() if value.num == max_num]
        sorted_keys = max(max_num_keys, key=lambda x: itemdict[x].last_item)

        eps = 1e-5

        ensembleOutput = torch.tensor(
            [itemdict[i].num / output.shape[0] if i != sorted_keys else itemdict[i].num / output.shape[0] + eps for i
             in
             range(len(labelmap))],
            requires_grad=True).to(label.device)
        sortedkeyslist.append(sorted_keys)
        itemlist.append(ensembleOutput)
    return torch.stack(itemlist), torch.Tensor(sortedkeyslist).to(
        label.device)


def batchProcess(config, correct_predictions, criterion, epoch_loss, maskModel, model, optimizer, promptModel,
                 tokenizer, tokens_num, total_samples, dataloader, train: bool, dim: int, labelmap: List[int]):
    """
    进行批处理。

    Args:
        config: 配置信息。
        correct_predictions (int): 当前正确预测的数量。
        criterion: 损失函数。
        epoch_loss (float): 当前epoch的损失值。
        maskModel: 掩码模型。
        model: 主模型。
        optimizer: 优化器。
        promptModel: 提示模型。
        tokenizer: 分词器。
        tokens_num: 标记的数量。
        total_samples (int): 当前已处理的样本数量。
        dataloader: 数据加载器。
        train (bool): 是否在训练模式下。

    Returns:
        Tuple[int, float, int]: 更新后的正确预测数量、epoch损失和样本总数。
    """
    for batch in dataloader:
        if(train):
            optimizer.zero_grad()
        torch.cuda.empty_cache()
        time.sleep(0.5)
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))

        data = batch['data']
        label = batch['label']
        datalength = data.shape[1]
        if (dim == 3):
            data = data.view(data.shape[0] * data.shape[1], data.shape[2])
        elif (dim == 2):
            pass
        else:
            raise NotImplementedError("dim!=2 And dim!=3 Not Implement!")
        data, maskpos = addDataAndMaskToPrompt(
            data, promptModel, tokenizer, tokens_num)
        data = data.to(config.device)
        label = label.to(config.device)
        with torch.no_grad():
            output = model(data)
        last_hidden_state = output.last_hidden_state
        maskItem = last_hidden_state[:, maskpos, :]
        out = maskModel(maskItem)
        if (dim == 3):
            ensembleoutput, predicted = ensembleLearning(out, label, labelmap, datalength)
            # print(ensembleoutput.shape)
            ensembleoutput = ensembleoutput.view(config.batch_size, -1)
            loss = criterion(ensembleoutput, label.long())
        elif (dim == 2):
            # print(out.shape)
            loss = criterion(out, label.long())
            _, predicted = torch.max(out, 1)
        else:
            raise NotImplementedError("dim!=2 And dim!=3 Not Implement!")
        # exit(0)
        if train:
            # optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 计算准确率
        correct_predictions += (predicted == label).sum().item()
        total_samples += label.size(0)
        # 累积epoch损失
        epoch_loss += loss.item()
        del data, output, last_hidden_state, maskItem, label, loss, predicted, batch

    return correct_predictions, epoch_loss, total_samples


def encode_data_with_special_symbols(data, tokenizer):
    """
    使用特殊符号编码数据。

    Args:
        data: 需要编码的数据。
        tokenizer: 分词器。

    Returns:
        List: 编码后的数据。
    """
    encoded_data = []

    for item in data:
        # 将特殊符号组成的矩阵转换为字符串
        item_str = "".join(item)  # 假设特殊符号之间用空格分隔
        encoded_item = tokenizer.encode(item_str, add_special_tokens=False)
        encoded_data.append(encoded_item[0])
    return encoded_data


def addDataAndMaskToPrompt(data, promptModel, tokenizer, tokens_num):
    """
    向数据中添加特殊标记和掩码。

    Args:
        data: 待处理的数据。
        promptModel: 提示模型。
        tokenizer: 分词器。
        tokens_num: 标记的数量。

    Returns:
        Tuple: 包含重新编码后的数据和掩码位置的元组。
    """
    data = transform_data(data, LEVEL_TOKEN_FORMAT)

    encoded_data = []
    for item in data:
        encoded_item = encode_data_with_special_symbols(item, tokenizer)
        encoded_data.append(encoded_item)
    datalength = len(encoded_item) + trainConfig.output_length

    beforePrompt, afterPrompt, maskpos = promptModel(
        tokens_num, datalength, tokenizer)

    batch_size = len(encoded_data)
    beforePrompt = beforePrompt.expand(batch_size, -1)
    afterPrompt = afterPrompt.expand(batch_size, -1)
    data = torch.Tensor(encoded_data).to(beforePrompt.device)

    data_tensor = torch.cat((beforePrompt, data, afterPrompt), dim=1)

    return data_tensor.long(), maskpos


def transform_data(data, token_format):
    """
    将数据转换成特定格式的字符串。

    Args:
        data (Tensor): 待转换的数据。
        token_format (str): 格式化字符串。

    Returns:
        List[List[str]]: 格式化后的数据。
    """
    # 获取batch_size
    batch_size = data.size(0)

    # 将二维tensor展平成一维
    flattened_data = data.view(batch_size, -1)

    # 使用格式化字符串对所有元素进行格式化
    formatted_data = [[token_format.format(
        item.item()) for item in sublist] for sublist in flattened_data]

    return formatted_data


def train(dataset: datastruct, config: trainConfig):
    """
    训练函数。

    Args:
        dataset (datastruct): 数据集。
        config (trainConfig): 训练配置。
    """
    utils.setup_seed(config.seed)
    data, label, labelmap = dataset.discrete(slicenum=trainConfig.slice_num)
    for randomstate in config.random_state:

        logger = logConfig(config, randomstate)
        model, tokenizer = LLM.getLLM()

        traindata, trainlabel, validdata, validlabel, testdata, testlabel = split_train_valid_test(data, label,
                                                                                                   randomstate=randomstate)
        # LLM.tokenizer_add_new_tokens(tokenizer, LEVEL_TOKEN_FORMAT,  [str(i) for i in range(dataset.slicenum)])
        # LLM.tokenizer_add_new_tokens(tokenizer, LABEL_TOKEN_FORMAT, label)
        tokens_num = tokenizer.vocab_size
        promptModel = PromptGenerate(
            config.init_shape, config.emb_dim, config.embLength, config.output_length, config.device)
        hidden_size = model.config.hidden_size
        maskModel = maskmodel(
            hidden_size, config.hidden_features, len(labelmap), config.dropout)
        for param in model.parameters():
            param.requires_grad = False
        promptModel.to(config.device)
        model.to(config.device)
        maskModel.to(config.device)
        traindataset = customdataset.CustomDataset(traindata, trainlabel)
        validdataset = customdataset.CustomDataset(validdata, validlabel)

        traindataloader = torch.utils.data.dataloader.DataLoader(
            traindataset, shuffle=True, batch_size=config.batch_size)
        validdataloader = torch.utils.data.dataloader.DataLoader(
            validdataset, batch_size=config.batch_size)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.NAdam(list(promptModel.parameters()) + list(maskModel.parameters()), lr=config.lr,
                                      weight_decay=config.weight_decay)

        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=0.1)

        best_valid_loss = float('inf')
        best_train_loss = float('inf')
        for epoch in range(1, config.num_epochs + 1):
            # 重置计数器和累积值
            epoch_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            valid_epoch_loss = 0.0
            valid_correct_predictions = 0
            total_valid_samples = 0
            correct_predictions, epoch_loss, total_samples = batchProcess(config, correct_predictions, criterion,
                                                                          epoch_loss, maskModel, model, optimizer,
                                                                          promptModel, tokenizer, tokens_num,
                                                                          total_samples,
                                                                          traindataloader, train=True, dim=config.dim,
                                                                          labelmap=labelmap)

            promptModel.eval()
            maskModel.eval()

            valid_correct_predictions, valid_epoch_loss, total_valid_samples = batchProcess(config,
                                                                                            valid_correct_predictions,
                                                                                            criterion,
                                                                                            valid_epoch_loss, maskModel,
                                                                                            model,
                                                                                            optimizer,
                                                                                            promptModel, tokenizer,
                                                                                            tokens_num,
                                                                                            total_valid_samples,
                                                                                            validdataloader,
                                                                                            train=False,
                                                                                            dim=config.dim,
                                                                                            labelmap=labelmap)

            promptModel.train()
            maskModel.train()

            # scheduler.step()

            # 计算epoch平均损失和准确率
            epoch_loss /= len(traindataloader)
            accuracy = correct_predictions / total_samples
            valid_epoch_loss /= len(validdataloader)
            valid_accuracy = valid_correct_predictions / total_valid_samples

            logger.info(
                f'Epoch [{epoch}/{config.num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {accuracy * 100:.2f}%, Valid Loss: {valid_epoch_loss:.4f}, Valid Accuracy: {valid_accuracy * 100:.2f}%')

            if config.saved_by_valid_loss and valid_epoch_loss < best_valid_loss:
                best_valid_loss = valid_epoch_loss
                # 保存模型，以valid_epoch_loss为标准
                torch.save(
                    promptModel, f'checkpoint/promptModel/{config.name}_{randomstate}_promptModel.pt')
                torch.save(
                    maskModel, f'checkpoint/maskModel/{config.name}_{randomstate}_maskModel.pt')
                logger.info(
                    f'Epoch [{epoch}/{config.num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {accuracy * 100:.2f}%, Valid Loss: {valid_epoch_loss:.4f}, Valid Accuracy: {valid_accuracy * 100:.2f}% Saved!')
            elif not config.saved_by_valid_loss and epoch_loss < best_train_loss:
                best_train_loss = epoch_loss
                # 保存模型，以epoch_loss为标准
                torch.save(
                    promptModel, f'checkpoint/promptModel/{config.name}_{randomstate}_promptModel.pt')
                torch.save(
                    maskModel, f'checkpoint/maskModel/{config.name}_{randomstate}_maskModel.pt')
                logger.info(
                    f'Epoch [{epoch}/{config.num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {accuracy * 100:.2f}%, Valid Loss: {valid_epoch_loss:.4f}, Valid Accuracy: {valid_accuracy * 100:.2f}% Saved!')


if __name__ == '__main__':
    # train(config.ADNI, ADNIconfig)
    # train(config.PPMI, PPMIconfig)
    train(config.ADNI_fMRI, config.ADNI_fMRIconfig)
    train(config.OCD_fMRI, config.OCD_fMRIconfig)
    train(config.FTD_fMRI, config.FTD_fMRIconfig)
    # ensembleLearning(torch.Tensor([1, 1]), torch.Tensor([1]), [0, 1, 2])
